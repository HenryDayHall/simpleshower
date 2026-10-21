"""
The SimpleShower model: what a detector array sees of one subshower, predicted
from the particle that starts it.

Input is the incident particle: its PDG, position (x/y/z) and momentum
(px/py/pz). Output is a list of energy deposits, each one a detector, a PDG
and an energy. Following the README:

1. Every detector is a token, made from its position (plus an embedding
   learned for that detector alone). They are worked out once per call, and
   are the same for every subshower.
2. One multiplicity token, made from the incident particle, attends to all
   the detectors, and predicts how many deposits there are. Stop if 0.
3. One token per deposit, made from the incident particle, the multiplicity
   and the deposit's rank, attends to all the detectors for N rounds. The
   deposits are updated; the detectors stay the same.
4. Each deposit's attention weights over the detectors say which detector it
   reaches, and its token predicts its PDG and energy.

Deposits are ranked by energy, highest first, and the rank is what tells the
deposit tokens of one subshower apart. They don't attend to each other, so
each deposit is predicted independently, given the incident particle, the
multiplicity and its rank.
"""

import copy
import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CHECKPOINT_VERSION = 1

# names the encoders' inputs lists can use
INCIDENT_INPUTS = [
    "incident_PDG",
    "incident_x",
    "incident_y",
    "incident_z",
    "incident_px",
    "incident_py",
    "incident_pz",
    # where the incident particle's track reaches the array's mean height;
    # worked out from x/y/z and px/py/pz, so no new information, but it's
    # nearly where the subshower lands
    "incident_ground_x",
    "incident_ground_y",
]
DEPOSIT_INPUTS = ["deposit_multiplicity", "deposit_rank"]
MOMENTUM_INPUTS = {"incident_px": 0, "incident_py": 1, "incident_pz": 2}


def detector_array(configs):
    """The array the config's detectors section describes (see subshowers.detector)."""
    from subshowers.detector import from_config as detector_from_config

    return detector_from_config(configs)


def count_detectors(configs):
    return len(detector_array(configs))


def _section(configs, name):
    return (configs.get("model") or {}).get(name) or {}


# ------------------------------------------------------------------- pieces


def _dropout(configs):
    return float((configs.get("model") or {}).get("dropout", 0.0))


class MLP(nn.Sequential):
    """Linear layers of the given widths, with GELU (and dropout) between them."""

    def __init__(self, in_dim, layers, out_dim, dropout=0.0):
        modules = []
        for width in layers or []:
            modules += [nn.Linear(in_dim, width), nn.GELU()]
            if dropout:
                modules.append(nn.Dropout(dropout))
            in_dim = width
        modules.append(nn.Linear(in_dim, out_dim))
        super().__init__(*modules)


def fourier(u, n_frequencies):
    """sin and cos of pi 2^k u for k < n_frequencies; u (...) -> (..., 2 n)."""
    frequencies = math.pi * 2.0 ** torch.arange(
        n_frequencies, device=u.device, dtype=u.dtype
    )
    angles = u[..., None] * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ArrayFrame(nn.Module):
    """
    Coordinates relative to the detector array: positions are measured from
    the array's mean position, in units of its half width, so the array
    fills [-1, 1] in x and y. Incident particles and detectors are both put
    in this frame, so their position encodings can be matched.
    """

    def __init__(self, detector_centers):
        super().__init__()
        centers = torch.as_tensor(np.asarray(detector_centers), dtype=torch.float32)
        center = centers.mean(dim=0)
        half_width = (centers[:, :2] - center[:2]).abs().max().clamp_min(1.0)
        self.register_buffer("center", center)
        self.register_buffer("half_width", half_width)

    def normalise(self, value, axis):
        return (value - self.center[axis]) / self.half_width

    def ground(self, incident, limit=10.0):
        """
        (x, y), in the frame, where the incident particle's straight track
        reaches the array's mean height. Tracks within 3 degrees of
        horizontal are taken at 3 degrees, and points are kept within
        `limit` half widths of the array.
        """
        momentum = torch.stack([incident["px"], incident["py"], incident["pz"]], -1)
        direction = momentum / momentum.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        down = (-direction[..., 2]).clamp_min(0.05)
        height = (incident["z"] - self.center[2]).clamp_min(0.0)
        distance = height / down
        ground_x = incident["x"] + direction[..., 0] * distance
        ground_y = incident["y"] + direction[..., 1] * distance
        return (
            self.normalise(ground_x, 0).clamp(-limit, limit),
            self.normalise(ground_y, 1).clamp(-limit, limit),
        )


class PDGVocabulary(nn.Module):
    """
    The PDG codes the model knows, each with a learned embedding. Index 0 is
    for any other code.
    """

    def __init__(self, codes, embedding_dim):
        super().__init__()
        codes = sorted({int(code) for code in codes})
        if not codes:
            raise ValueError("the model needs at least one PDG code in model: pdgs")
        self.register_buffer("codes", torch.tensor(codes, dtype=torch.long))
        self.n_classes = len(codes) + 1
        self.dim = embedding_dim
        self.embedding = nn.Embedding(self.n_classes, embedding_dim)

    def index(self, pdg):
        """Vocabulary index of each PDG code; 0 for codes not in it."""
        pdg = pdg.long()
        position = torch.searchsorted(self.codes, pdg).clamp_max(len(self.codes) - 1)
        found = self.codes[position] == pdg
        return torch.where(found, position + 1, torch.zeros_like(position))

    def code(self, index):
        """PDG code of each vocabulary index; 0 for index 0."""
        padded = torch.cat([self.codes.new_zeros(1), self.codes])
        return padded[index]

    def forward(self, index):
        return self.embedding(index)


class InputFeatures(nn.Module):
    """
    One vector per token from the named inputs (see INCIDENT_INPUTS and
    DEPOSIT_INPUTS):

    - incident_PDG is a learned embedding;
    - positions are measured in the array's frame (ArrayFrame), and the
      horizontal ones are also given as Fourier features, as the detectors'
      positions are, so the two can be matched in attention;
    - momentum components go in as the direction (p_i / |p|), with log10 |p|
      added once, so the energy scale isn't lost;
    - deposit_multiplicity goes in as log(1 + multiplicity), and
      deposit_rank as (i + 1/2) / multiplicity for the i-th deposit by energy.

    Everything but the embedding and the Fourier features is standardised
    with the means and widths of the training set (fit).
    """

    def __init__(self, inputs, frame, pdgs, n_frequencies, allow_deposit_inputs):
        super().__init__()
        allowed = INCIDENT_INPUTS + (DEPOSIT_INPUTS if allow_deposit_inputs else [])
        unknown = [name for name in inputs if name not in allowed]
        if unknown:
            raise ValueError(f"unknown inputs {unknown}; inputs can be from {allowed}")
        self.inputs = list(inputs)
        self.frame = frame
        self.pdgs = pdgs
        self.n_frequencies = n_frequencies
        self.use_pdg = "incident_PDG" in inputs
        self.incident_names = [
            name
            for name in inputs
            if name in INCIDENT_INPUTS and name != "incident_PDG"
        ]
        if any(name in MOMENTUM_INPUTS for name in inputs):
            self.incident_names.append("incident_log_momentum")
        self.fourier_names = [
            name
            for name in inputs
            if name
            in ("incident_x", "incident_y", "incident_ground_x", "incident_ground_y")
        ]
        self.deposit_names = [name for name in inputs if name in DEPOSIT_INPUTS]
        self.register_buffer("incident_mean", torch.zeros(len(self.incident_names)))
        self.register_buffer("incident_std", torch.ones(len(self.incident_names)))
        self.register_buffer("deposit_mean", torch.zeros(len(self.deposit_names)))
        self.register_buffer("deposit_std", torch.ones(len(self.deposit_names)))
        self.incident_dim = (
            len(self.incident_names)
            + 2 * n_frequencies * len(self.fourier_names)
            + (pdgs.dim if self.use_pdg else 0)
        )
        self.dim = self.incident_dim + len(self.deposit_names)
        if self.dim == 0:
            raise ValueError("inputs is empty")

    def _incident_values(self, incident):
        """The incident inputs, in the array's frame, before standardising."""
        values = {
            "incident_x": lambda: self.frame.normalise(incident["x"], 0),
            "incident_y": lambda: self.frame.normalise(incident["y"], 1),
            "incident_z": lambda: self.frame.normalise(incident["z"], 2),
        }
        momentum = torch.stack([incident["px"], incident["py"], incident["pz"]], -1)
        size = momentum.norm(dim=-1).clamp_min(1e-12)
        for name, axis in MOMENTUM_INPUTS.items():
            values[name] = lambda axis=axis: momentum[..., axis] / size
        values["incident_log_momentum"] = lambda: torch.log10(size)
        if any(name.startswith("incident_ground") for name in self.inputs):
            ground_x, ground_y = self.frame.ground(incident)
            values["incident_ground_x"] = lambda: ground_x
            values["incident_ground_y"] = lambda: ground_y
        return {name: values[name]() for name in self.incident_names + self.fourier_names}

    def _deposit_values(self, multiplicity, rank):
        values = {
            "deposit_multiplicity": lambda: torch.log1p(multiplicity.float())[
                :, None
            ].expand_as(rank),
            "deposit_rank": lambda: rank,
        }
        return torch.stack([values[name]() for name in self.deposit_names], -1)

    def forward(self, incident, multiplicity=None, rank=None):
        """
        incident: dict of (B,) tensors -> (B, dim). For deposit tokens, also
        multiplicity (B,) and rank (B, K) -> (B, K, dim).
        """
        values = self._incident_values(incident)
        parts = []
        if self.incident_names:
            scalars = torch.stack([values[name] for name in self.incident_names], -1)
            parts.append((scalars - self.incident_mean) / self.incident_std)
        for name in self.fourier_names:
            parts.append(fourier(values[name], self.n_frequencies))
        if self.use_pdg:
            parts.append(self.pdgs(self.pdgs.index(incident["pdg"])))
        if rank is None:
            if self.deposit_names:
                raise ValueError(f"inputs {self.deposit_names} need multiplicity and rank")
            return torch.cat(parts, -1)
        n_tokens = rank.shape[1]
        pieces = []
        if parts:
            pieces.append(torch.cat(parts, -1)[:, None, :].expand(-1, n_tokens, -1))
        if self.deposit_names:
            deposit = self._deposit_values(multiplicity, rank)
            pieces.append((deposit - self.deposit_mean) / self.deposit_std)
        return torch.cat(pieces, -1)

    @torch.no_grad()
    def fit(self, incident, multiplicity):
        """
        Standardise with the training set: incident, dict of (S,) tensors, and
        multiplicity (S,) for every training subshower. Deposit inputs are
        standardised over the deposits of the subshowers that have some.
        """
        if self.incident_names:
            values = self._incident_values(incident)
            scalars = torch.stack([values[name] for name in self.incident_names], -1)
            self.incident_mean.copy_(scalars.mean(0))
            self.incident_std.copy_(scalars.std(0).nan_to_num(1.0).clamp_min(1e-6))
        if self.deposit_names:
            counts = multiplicity[multiplicity > 0].long()
            if len(counts) == 0:
                return
            per_deposit = torch.repeat_interleave(counts, counts)
            first = torch.cumsum(counts, 0) - counts
            ranks = torch.arange(len(per_deposit)) - torch.repeat_interleave(
                first, counts
            )
            rank = ((ranks + 0.5) / per_deposit)[:, None]
            values = self._deposit_values(per_deposit, rank)[:, 0]
            self.deposit_mean.copy_(values.mean(0))
            self.deposit_std.copy_(values.std(0).nan_to_num(1.0).clamp_min(1e-6))


class DetectorTokens(nn.Module):
    """
    One token per detector, from its position in the array's frame (and its
    Fourier features), plus an embedding learned for that detector alone,
    which starts at zero.
    """

    def __init__(self, configs, frame, detector_centers, n_frequencies):
        super().__init__()
        section = _section(configs, "detector_tokens")
        self.frame = frame
        self.n_frequencies = n_frequencies
        self.dim = section.get("embedding_dim", 128)
        self.register_buffer(
            "centers", torch.as_tensor(np.asarray(detector_centers), dtype=torch.float32)
        )
        self.mlp = MLP(
            3 + 4 * n_frequencies, section.get("layers", [128]), self.dim, _dropout(configs)
        )
        self.own = None
        if section.get("learned", True):
            self.own = nn.Parameter(torch.zeros(len(self.centers), self.dim))

    def forward(self):
        u = torch.stack(
            [self.frame.normalise(self.centers[:, axis], axis) for axis in range(3)], -1
        )
        features = torch.cat(
            [
                u,
                fourier(u[:, 0], self.n_frequencies),
                fourier(u[:, 1], self.n_frequencies),
            ],
            -1,
        )
        tokens = self.mlp(features)
        if self.own is not None:
            tokens = tokens + self.own
        return tokens


class DetectorAttention(nn.Module):
    """
    One round: tokens attend to all the detectors, and are updated (pre-norm
    multi-head cross-attention, then a feed-forward layer, both residual).
    The detectors aren't changed.
    """

    def __init__(self, dim, detector_dim, heads, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"embedding_dim {dim} must divide into {heads} heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm_tokens = nn.LayerNorm(dim)
        self.norm_detectors = nn.LayerNorm(detector_dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(detector_dim, dim)
        self.value = nn.Linear(detector_dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = MLP(dim, [2 * dim], dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, detectors):
        """tokens (..., dim), detectors (N, detector_dim) -> (..., dim)"""
        lead = tokens.shape[:-1]
        query = self.query(self.norm_tokens(tokens)).view(*lead, self.heads, -1)
        normed = self.norm_detectors(detectors)
        key = self.key(normed).view(len(detectors), self.heads, -1)
        value = self.value(normed).view(len(detectors), self.heads, -1)
        weights = torch.einsum("...hd,nhd->...hn", query, key) / math.sqrt(
            self.head_dim
        )
        weights = weights.softmax(dim=-1)
        mixed = torch.einsum("...hn,nhd->...hd", weights, value).reshape(*lead, -1)
        tokens = tokens + self.dropout(self.out(mixed))
        return tokens + self.dropout(self.ff(self.norm_ff(tokens)))


class DetectorPointer(nn.Module):
    """
    The deposit tokens' final attention logits over the detectors; their
    softmax is the probability of each detector being the one reached.
    """

    def __init__(self, dim, detector_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(detector_dim, dim)
        self.scale = 1 / math.sqrt(dim)

    def forward(self, tokens, detectors):
        """tokens (..., dim), detectors (N, detector_dim) -> logits (..., N)"""
        query = self.query(self.norm(tokens))
        return torch.einsum("...d,nd->...n", query, self.key(detectors)) * self.scale


def multiplicity_edges(max_multiplicity, exact_up_to=16, growth=1.25):
    """
    Edges of the multiplicity classes: class c holds multiplicities from
    edges[c] up to, not including, edges[c + 1]. Below exact_up_to every
    multiplicity has a class of its own; above it they widen by a factor
    growth, until max_multiplicity is covered. Larger multiplicities are put
    in the last class.
    """
    edges = list(range(exact_up_to + 1))
    while edges[-1] <= max_multiplicity:
        edges.append(max(edges[-1] + 1, int(math.ceil(edges[-1] * growth))))
    return edges


# ------------------------------------------------ the parts the README names


class MultiplicityEncoder(nn.Module):
    """The multiplicity token, made from the incident particle."""

    def __init__(self, configs, frame, pdgs, n_frequencies):
        super().__init__()
        section = _section(configs, "multiplicity_encoder")
        self.dim = _section(configs, "multiplicity_token").get("embedding_dim", 128)
        self.features = InputFeatures(
            section.get("inputs", INCIDENT_INPUTS[:7]),
            frame,
            pdgs,
            n_frequencies,
            allow_deposit_inputs=False,
        )
        self.mlp = MLP(
            self.features.dim, section.get("layers", [64, 128]), self.dim, _dropout(configs)
        )

    def forward(self, incident):
        return self.mlp(self.features(incident))


class MultiplicityPredictor(nn.Module):
    """Logits of the multiplicity classes (see multiplicity_edges)."""

    def __init__(self, configs, dim):
        super().__init__()
        section = _section(configs, "multiplicity_predictor")
        edges = multiplicity_edges(
            section.get("max_multiplicity", 1000),
            section.get("exact_up_to", 16),
            section.get("growth", 1.25),
        )
        self.register_buffer("edges", torch.tensor(edges, dtype=torch.long))
        self.n_classes = len(edges) - 1
        self.mlp = MLP(dim, section.get("layers", [64]), self.n_classes, _dropout(configs))

    def forward(self, token):
        return self.mlp(token)

    def class_of(self, multiplicity):
        classes = torch.searchsorted(self.edges, multiplicity.long(), right=True) - 1
        return classes.clamp(0, self.n_classes - 1)

    def draw(self, logits, generator=None, greedy=False):
        """
        A multiplicity for each row of logits: a class drawn from them, then a
        multiplicity drawn uniformly within it. If greedy, the most likely
        multiplicity instead, taking each class's probability to be spread
        evenly over its multiplicities.
        """
        widths = (self.edges[1:] - self.edges[:-1]).to(logits.dtype)
        if greedy:
            classes = (logits - widths.log()).argmax(-1)
            low = self.edges[classes]
            return low + (self.edges[classes + 1] - low - 1) // 2
        classes = torch.multinomial(logits.softmax(-1), 1, generator=generator)[:, 0]
        low = self.edges[classes]
        width = self.edges[classes + 1] - low
        draw = torch.rand(classes.shape, generator=generator, device=classes.device)
        return low + (draw * width).long().clamp_max(width - 1)


class DepositEncoder(nn.Module):
    """One token per deposit: the incident particle, multiplicity and rank."""

    def __init__(self, configs, frame, pdgs, n_frequencies):
        super().__init__()
        section = _section(configs, "deposit_encoder")
        self.dim = _section(configs, "deposit_token").get("embedding_dim", 128)
        self.features = InputFeatures(
            section.get("inputs", INCIDENT_INPUTS[:7] + DEPOSIT_INPUTS),
            frame,
            pdgs,
            n_frequencies,
            allow_deposit_inputs=True,
        )
        # Without deposit_rank, every deposit token of a subshower is the same,
        # so its deposits are drawn independently from one distribution.
        self.mlp = MLP(
            self.features.dim, section.get("layers", [64, 128]), self.dim, _dropout(configs)
        )

    def forward(self, incident, multiplicity, rank):
        return self.mlp(self.features(incident, multiplicity, rank))


class DepositPDGPredictor(nn.Module):
    """Logits of each deposit's PDG, over the PDG vocabulary."""

    def __init__(self, configs, dim, pdgs):
        super().__init__()
        layers = _section(configs, "deposit_PDG_predictor").get("layers", [64])
        self.mlp = MLP(dim, layers, pdgs.n_classes, _dropout(configs))

    def forward(self, tokens):
        return self.mlp(tokens)


class DepositEnergyPredictor(nn.Module):
    """
    A normal distribution for log10(energy / GeV) of each deposit, given its
    token and its PDG: (mean, width).
    """

    def __init__(self, configs, dim, pdgs):
        super().__init__()
        layers = _section(configs, "deposit_energy_predictor").get("layers", [64])
        self.pdgs = pdgs
        self.mlp = MLP(dim + pdgs.dim, layers, 2, _dropout(configs))

    def forward(self, tokens, pdg_index):
        out = self.mlp(torch.cat([tokens, self.pdgs(pdg_index)], -1))
        return out[..., 0], F.softplus(out[..., 1]) + 1e-3


# -------------------------------------------------------------- the model


class Model(nn.Module):
    def __init__(self, configs, detector_centers=None):
        """
        configs as read from a yaml (see configs/default.yaml). The detector
        array is built from its detectors section, unless detector_centers
        (N, 3) are given. model: pdgs must list the PDG codes the model
        knows; train.py fills it in from the data if it's empty.
        """
        super().__init__()
        self.configs = copy.deepcopy(configs)
        model_configs = self.configs.setdefault("model", {})
        if detector_centers is None:
            detector_centers = detector_array(self.configs).centers
        self.n_detectors = len(detector_centers)
        if not model_configs.get("pdgs"):
            raise ValueError(
                "model: pdgs is empty; list the PDG codes the model should know,"
                " or let train.py fill them in from the data"
            )
        n_frequencies = model_configs.get("position_frequencies", 8)

        self.frame = ArrayFrame(detector_centers)
        self.pdgs = PDGVocabulary(
            model_configs["pdgs"], model_configs.get("pdg_embedding_dim", 16)
        )
        self.detector_tokens = DetectorTokens(
            self.configs, self.frame, detector_centers, n_frequencies
        )
        detector_dim = self.detector_tokens.dim

        self.multiplicity_encoder = MultiplicityEncoder(
            self.configs, self.frame, self.pdgs, n_frequencies
        )
        dim = self.multiplicity_encoder.dim
        dropout = _dropout(self.configs)
        token = _section(self.configs, "multiplicity_token")
        self.multiplicity_rounds = nn.ModuleList(
            DetectorAttention(dim, detector_dim, token.get("heads", 4), dropout)
            for _ in range(token.get("rounds", 1))
        )
        self.multiplicity_predictor = MultiplicityPredictor(self.configs, dim)

        self.deposit_encoder = DepositEncoder(
            self.configs, self.frame, self.pdgs, n_frequencies
        )
        dim = self.deposit_encoder.dim
        token = _section(self.configs, "deposit_token")
        self.deposit_rounds = nn.ModuleList(
            DetectorAttention(dim, detector_dim, token.get("heads", 4), dropout)
            for _ in range(token.get("rounds", 3))
        )
        self.detector_pointer = DetectorPointer(dim, detector_dim)
        self.deposit_PDG_predictor = DepositPDGPredictor(self.configs, dim, self.pdgs)
        self.deposit_energy_predictor = DepositEnergyPredictor(
            self.configs, dim, self.pdgs
        )

    # ----------------------------------------------------------- pieces

    def multiplicity_logits(self, incident, detectors):
        """incident: dict of (B,) tensors -> logits (B, multiplicity classes)"""
        token = self.multiplicity_encoder(incident)
        for block in self.multiplicity_rounds:
            token = block(token, detectors)
        return self.multiplicity_predictor(token)

    def deposit_tokens(self, incident, multiplicity, rank, detectors):
        """incident (B,), multiplicity (B,), rank (B, K) -> tokens (B, K, dim)"""
        tokens = self.deposit_encoder(incident, multiplicity, rank)
        for block in self.deposit_rounds:
            tokens = block(tokens, detectors)
        return tokens

    @torch.no_grad()
    def fit_normalisation(self, incident, multiplicity):
        """Standardise the inputs with the training set (see InputFeatures.fit)."""
        self.multiplicity_encoder.features.fit(incident, multiplicity)
        self.deposit_encoder.features.fit(incident, multiplicity)

    # ------------------------------------------------------------- loss

    LOSSES = ["multiplicity", "detector", "pdg", "energy"]

    def total_loss(self, losses):
        """The losses summed, weighted by the config's training: loss_weights."""
        weights = (self.configs.get("training") or {}).get("loss_weights") or {}
        return sum(
            weights.get(name, 1.0) * losses[name] for name in self.LOSSES if name in losses
        )

    def loss(self, incident, multiplicity, deposits=None, metrics=False):
        """
        Losses for a batch, as a dict of scalars, with "total" their weighted
        sum (see total_loss).

        incident: dict of (B,) tensors (pdg, x, y, z, px, py, pz)
        multiplicity: (B,) true number of deposits
        deposits: for the deposit losses, a dict with
            incident, multiplicity: as above, for B' subshowers with deposits
            rank (B', K), mask (B', K): the deposits trained on, padded
            weight (B', K): weight of each in the loss
            detector, pdg (B', K): true detector index and PDG code
            log_energy (B', K): true log10(energy / GeV)
        metrics: also give accuracies, which aren't in the total
        """
        detectors = self.detector_tokens()
        losses = self.multiplicity_losses(incident, multiplicity, detectors, metrics)
        if deposits is not None:
            losses.update(self.deposit_losses(deposits, detectors, metrics))
        losses["total"] = self.total_loss(losses)
        return losses

    def multiplicity_losses(self, incident, multiplicity, detectors, metrics=False):
        logits = self.multiplicity_logits(incident, detectors)
        target = self.multiplicity_predictor.class_of(multiplicity)
        losses = {"multiplicity": F.cross_entropy(logits, target)}
        if metrics:
            # is P(no deposits) < 1/2 when, and only when, there are some?
            predicted = logits.softmax(-1)[:, 0] < 0.5
            losses["detected_accuracy"] = (predicted == (multiplicity > 0)).float().mean()
        return losses

    def deposit_losses(self, deposits, detectors, metrics=False):
        tokens = self.deposit_tokens(
            deposits["incident"], deposits["multiplicity"], deposits["rank"], detectors
        )
        weight = deposits["weight"] * deposits["mask"]
        total_weight = weight.sum().clamp_min(1e-12)

        def mean(values):
            return (values * weight).sum() / total_weight

        def cross_entropy(logits, target):
            values = F.cross_entropy(
                logits.flatten(0, 1), target.flatten(), reduction="none"
            )
            return mean(values.view_as(weight))

        pointer = self.detector_pointer(tokens, detectors)
        pdg_logits = self.deposit_PDG_predictor(tokens)
        pdg_index = self.pdgs.index(deposits["pdg"])
        mean_log_e, width = self.deposit_energy_predictor(tokens, pdg_index)
        z = (deposits["log_energy"] - mean_log_e) / width
        energy_nll = 0.5 * z**2 + width.log() + 0.5 * math.log(2 * math.pi)
        losses = {
            "detector": cross_entropy(pointer, deposits["detector"]),
            "pdg": cross_entropy(pdg_logits, pdg_index),
            "energy": mean(energy_nll),
        }
        if metrics:
            # the highest attention weight on the right detector?
            hit = (pointer.argmax(-1) == deposits["detector"]).float()
            losses["detector_accuracy"] = mean(hit)
            losses["pdg_accuracy"] = mean((pdg_logits.argmax(-1) == pdg_index).float())
        return losses

    # ------------------------------------------------------- generation

    @torch.no_grad()
    def generate(self, incident, greedy=False, generator=None, max_tokens=8192):
        """
        Predict the deposits of one subshower for each incident particle.

        incident: dict of (B,) tensors (pdg, x, y, z, px, py, pz)
        greedy: take the most likely multiplicity, detector (the highest
            attention weight) and PDG, and the median energy, rather than
            drawing them
        generator: torch.Generator for the draws
        max_tokens: deposit tokens worked on at once, to bound memory

        Returns a list of B dicts of numpy arrays, detector, pdg and energy
        (GeV), one entry per deposit, in order of rank.
        """
        was_training = self.training
        self.eval()  # no dropout while generating
        try:
            return self._generate(incident, greedy, generator, max_tokens)
        finally:
            self.train(was_training)

    def _generate(self, incident, greedy, generator, max_tokens):
        detectors = self.detector_tokens()
        logits = self.multiplicity_logits(incident, detectors)
        multiplicity = self.multiplicity_predictor.draw(logits, generator, greedy)
        counts = multiplicity.tolist()
        results = [
            {
                "detector": np.zeros(0, dtype=np.int64),
                "pdg": np.zeros(0, dtype=np.int64),
                "energy": np.zeros(0),
            }
            for _ in counts
        ]
        # stop where there are no deposits; the rest go in chunks of at
        # most max_tokens deposit tokens (padded to the chunk's largest)
        chunk, largest = [], 0
        for b in [b for b, count in enumerate(counts) if count > 0]:
            if chunk and max(largest, counts[b]) * (len(chunk) + 1) > max_tokens:
                self._generate_deposits(
                    incident, multiplicity, chunk, detectors, greedy, generator, results
                )
                chunk, largest = [], 0
            chunk.append(b)
            largest = max(largest, counts[b])
        if chunk:
            self._generate_deposits(
                incident, multiplicity, chunk, detectors, greedy, generator, results
            )
        return results

    def _generate_deposits(
        self, incident, multiplicity, rows, detectors, greedy, generator, results
    ):
        """Fill in results[b] for each b of rows, which all have deposits."""
        index = torch.as_tensor(rows, device=multiplicity.device)
        incident = {name: value[index] for name, value in incident.items()}
        multiplicity = multiplicity[index]
        n_tokens = int(multiplicity.max())
        rank = (torch.arange(n_tokens, device=multiplicity.device) + 0.5)[None, :] / (
            multiplicity[:, None]
        )
        tokens = self.deposit_tokens(incident, multiplicity, rank, detectors)
        pointer = self.detector_pointer(tokens, detectors)
        pdg_logits = self.deposit_PDG_predictor(tokens)
        # index 0 is "any other code", which isn't a PDG to give a deposit
        pdg_logits[..., 0] = -math.inf
        if greedy:
            detector = pointer.argmax(-1)
            pdg_index = pdg_logits.argmax(-1)
        else:
            detector = _draw(pointer, generator)
            pdg_index = _draw(pdg_logits, generator)
        mean_log_e, width = self.deposit_energy_predictor(tokens, pdg_index)
        log_energy = mean_log_e
        if not greedy:
            noise = torch.randn(
                mean_log_e.shape, generator=generator, device=mean_log_e.device
            )
            log_energy = mean_log_e + width * noise
        pdg = self.pdgs.code(pdg_index)
        for row, (b, count) in enumerate(zip(rows, multiplicity.tolist())):
            results[b] = {
                "detector": detector[row, :count].cpu().numpy(),
                "pdg": pdg[row, :count].cpu().numpy(),
                "energy": (10.0 ** log_energy[row, :count].double()).cpu().numpy(),
            }

    # ------------------------------------------------------ checkpoints

    @classmethod
    def _load_from_checkpoint(cls, checkpoint):
        version = checkpoint.get("version")
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"checkpoint version {version}, but this model reads {CHECKPOINT_VERSION}"
            )
        model = cls(checkpoint["configs"], detector_centers=checkpoint["detector_centers"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()

    @classmethod
    def load(cls, path, map_location="cpu"):
        """A model written by write, ready to generate with (in eval mode)."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        return cls._load_from_checkpoint(checkpoint)

    def write(self, path, **extra):  # include both weights and configs
        """
        Save the weights, the configs and the detector positions, so load
        can rebuild the model without the array. extra is saved alongside,
        e.g. the epoch.
        """
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "configs": self.configs,
            "detector_centers": self.detector_tokens.centers.detach().cpu(),
            "state_dict": self.state_dict(),
            **extra,
        }
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        temporary = path + ".tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)


def _draw(logits, generator=None):
    """One index from each row of logits (..., C)."""
    probabilities = logits.softmax(-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    return torch.multinomial(flat, 1, generator=generator).view(logits.shape[:-1])
