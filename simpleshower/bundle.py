"""
Export a trained SimpleShower as a bundle that CORSIKA 8 can run:

    python -m simpleshower.bundle CHECKPOINT BUNDLE_DIR [--station-radius 10] [--check]

A bundle is a directory:

    manifest.yaml      corsika: the contract with CORSIKA (frame, units, thresholds,
                         PDG codes, stations, the training runs' physics), which
                         CORSIKA checks before it runs;
                       plugin: what plugin/ (or sampling.py) needs;
                       source: the checkpoint it came from
    multiplicity.onnx  (pdg_index, kinematics, noise) -> multiplicity, p_none
    deposits.onnx      (pdg_index, kinematics, multiplicity, noise)
                         -> detector, deposit_pdg_index, log10_energy
    stations.csv       index, id, x, y, z, radius: the detectors, in the model's frame

The graphs take uniform noise in and give samples out, so they are deterministic and
the seeds CORSIKA gives each particle decide every draw (see sampling.py for how noise
is made from a seed). The model is split in two where the work depends on a draw: how
many deposit tokens each subshower needs. Detector tokens are worked out once here and
stored in the graphs as constants; ONNX Runtime folds their key and value projections
when it loads them. PDG codes go in and out as indices into the model's vocabulary
(plugin: pdgs), as the graphs can't do torch.searchsorted.
"""

import argparse
import copy
import hashlib
import math
import os
import shutil
import sys

import numpy as np
import torch
import yaml
from torch import nn

from simpleshower.model import Model

BUNDLE_FORMAT = 1  # of the graphs' inputs and outputs, and of the noise
NOISE_MULTIPLICITY = 2  # uniforms per particle: class, place within the class
NOISE_PER_DEPOSIT = 4  # uniforms per deposit: detector, PDG, two for a normal

# what c8_air_shower(_with_history) does by default, which is how subshowers'
# generation scripts make the training showers; CORSIKA warns where a run differs
TRAINING_PHYSICS = {
    "hadronic_model": "SIBYLL-2.3d",
    "low_energy_model": "UrQMD",
    "emcut": 5.0e-4,
    "hadcut": 0.3,
    "mucut": 0.3,
    "emthin": 1.0e-6,
    "atmosphere": "USStdBK",
    "magnetic_field_uT": [50.0, 0.0, 0.0],
}


def inverse_cdf(probabilities, u):
    """
    One index from each row of probabilities (..., C), given uniforms u (...): the first
    index whose cumulative probability exceeds u times the total. Summed in double, so
    the draw doesn't depend on how the runtime rounds; zero-probability entries are never
    drawn.
    """
    cdf = probabilities.double().cumsum(-1)
    threshold = u.double().unsqueeze(-1) * cdf[..., -1:]
    index = (cdf <= threshold).sum(-1)
    return index.clamp(max=probabilities.shape[-1] - 1)


def _incident(pdg_index, kinematics):
    """The model's incident dict, from vocabulary indices and (x, y, z, px, py, pz)."""
    names = ["x", "y", "z", "px", "py", "pz"]
    incident = {name: kinematics[:, i] for i, name in enumerate(names)}
    incident["pdg"] = pdg_index
    return incident


def _for_export(model):
    """
    A copy of the model in eval mode that takes vocabulary indices where it took PDG
    codes (the exported graphs get indices, and give indices back).
    """
    model = copy.deepcopy(model).eval()
    model.pdgs.index = lambda index: index  # instance attribute, shadows the method
    return model


class MultiplicityGraph(nn.Module):
    """
    pdg_index (B,) int64, kinematics (B, 6) float32, noise (B, 2) uniforms
    -> multiplicity (B,) int64, p_none (B,) float32, the probability of no deposits
    """

    def __init__(self, model):
        super().__init__()
        self.model = _for_export(model)
        with torch.no_grad():
            self.register_buffer("detectors", self.model.detector_tokens().detach())
        edges = self.model.multiplicity_predictor.edges
        self.register_buffer("low", edges[:-1].clone())
        self.register_buffer("width", (edges[1:] - edges[:-1]).clone())

    def forward(self, pdg_index, kinematics, noise):
        incident = _incident(pdg_index, kinematics)
        logits = self.model.multiplicity_logits(incident, self.detectors)
        probabilities = logits.softmax(-1)
        classes = inverse_cdf(probabilities, noise[:, 0])
        low, width = self.low[classes], self.width[classes]
        # a multiplicity drawn uniformly within the class, as MultiplicityPredictor.draw
        within = torch.floor(noise[:, 1].double() * width.double()).long()
        multiplicity = low + torch.minimum(within, width - 1)
        return multiplicity, probabilities[:, 0]


class DepositGraph(nn.Module):
    """
    pdg_index (B,) int64, kinematics (B, 6) float32, multiplicity (B,) int64,
    noise (B, K, 4) uniforms, for K deposit tokens per subshower
    -> detector (B, K) int64, deposit_pdg_index (B, K) int64, log10_energy (B, K) float32

    Token j of row b is the deposit of rank j; tokens j >= multiplicity[b] are padding,
    to be ignored. A token's outputs depend only on its row and its noise: deposit
    tokens attend to the detectors, not to each other or to other rows.
    """

    def __init__(self, model):
        super().__init__()
        self.model = _for_export(model)
        with torch.no_grad():
            self.register_buffer("detectors", self.model.detector_tokens().detach())
        other = torch.zeros(self.model.pdgs.n_classes, dtype=torch.bool)
        other[0] = True  # "any other code" isn't a PDG to give a deposit
        self.register_buffer("other", other)

    def forward(self, pdg_index, kinematics, multiplicity, noise):
        incident = _incident(pdg_index, kinematics)
        n_tokens = noise.shape[1]
        rank = (torch.arange(n_tokens, device=noise.device, dtype=torch.float32) + 0.5)[
            None, :
        ] / multiplicity[:, None].clamp_min(1).float()
        tokens = self.model.deposit_tokens(incident, multiplicity, rank, self.detectors)
        pointer = self.model.detector_pointer(tokens, self.detectors)
        detector = inverse_cdf(pointer.softmax(-1), noise[..., 0])
        pdg_logits = self.model.deposit_PDG_predictor(tokens).masked_fill(
            self.other, float("-inf")
        )
        pdg_index = inverse_cdf(pdg_logits.softmax(-1), noise[..., 1])
        mean, width = self.model.deposit_energy_predictor(tokens, pdg_index)
        # Box-Muller: a standard normal from the other two uniforms
        normal = torch.sqrt(-2.0 * torch.log(1.0 - noise[..., 2])) * torch.cos(
            2.0 * math.pi * noise[..., 3]
        )
        return detector, pdg_index, mean + width * normal


def example_inputs(n_detectors, batch=3, tokens=2, n_pdgs=1, seed=0):
    """Inputs of the right shapes and types, for tracing and checks."""
    g = torch.Generator().manual_seed(seed)
    pdg_index = torch.randint(0, n_pdgs + 1, (batch,), generator=g)
    position = torch.rand(batch, 3, generator=g) * torch.tensor([2000.0, 2000.0, 3000.0])
    direction = torch.nn.functional.normalize(
        torch.randn(batch, 3, generator=g) - torch.tensor([0.0, 0.0, 3.0]), dim=-1
    )
    momentum = direction * (10.0 ** (torch.rand(batch, 1, generator=g) * 4 - 2))
    kinematics = torch.cat([position, momentum], -1).float()
    multiplicity = torch.randint(1, tokens + 1, (batch,), generator=g)
    return {
        "pdg_index": pdg_index,
        "kinematics": kinematics,
        "noise_multiplicity": torch.rand(batch, NOISE_MULTIPLICITY, generator=g),
        "multiplicity": multiplicity,
        "noise_deposits": torch.rand(batch, tokens, NOISE_PER_DEPOSIT, generator=g),
    }


def export_graphs(model, folder, opset=18):
    """Write multiplicity.onnx and deposits.onnx; returns their paths."""
    batch = torch.export.Dim("batch")
    tokens = torch.export.Dim("tokens")
    x = example_inputs(model.n_detectors, n_pdgs=len(model.pdgs.codes))
    paths = {}

    graph = MultiplicityGraph(model).eval()
    paths["multiplicity"] = os.path.join(folder, "multiplicity.onnx")
    torch.onnx.export(
        graph,
        (x["pdg_index"], x["kinematics"], x["noise_multiplicity"]),
        paths["multiplicity"],
        input_names=["pdg_index", "kinematics", "noise"],
        output_names=["multiplicity", "p_none"],
        dynamic_shapes={
            "pdg_index": {0: batch},
            "kinematics": {0: batch},
            "noise": {0: batch},
        },
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )

    graph = DepositGraph(model).eval()
    paths["deposits"] = os.path.join(folder, "deposits.onnx")
    torch.onnx.export(
        graph,
        (x["pdg_index"], x["kinematics"], x["multiplicity"], x["noise_deposits"]),
        paths["deposits"],
        input_names=["pdg_index", "kinematics", "multiplicity", "noise"],
        output_names=["detector", "deposit_pdg_index", "log10_energy"],
        dynamic_shapes={
            "pdg_index": {0: batch},
            "kinematics": {0: batch},
            "multiplicity": {0: batch},
            "noise": {0: batch, 1: tokens},
        },
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    return paths


def _tag_graphs(paths, metadata):
    """Store what the plugin needs to read the graphs in their own metadata."""
    import onnx

    for path in paths.values():
        graph = onnx.load(path)
        del graph.metadata_props[:]
        for key, value in metadata.items():
            entry = graph.metadata_props.add()
            entry.key, entry.value = key, str(value)
        onnx.save(graph, path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_graphs(model, folder, trials=5, atol=1e-4):
    """
    Run the graphs with ONNX Runtime and the wrappers with PyTorch on the same random
    inputs. Integer draws must agree, except where a uniform falls within rounding of a
    boundary between two outcomes; log energies must agree to atol. Returns the worst
    fraction of integer draws that differ.
    """
    import onnxruntime as ort

    sessions = {
        name: ort.InferenceSession(os.path.join(folder, f"{name}.onnx"))
        for name in ["multiplicity", "deposits"]
    }
    graphs = {
        "multiplicity": MultiplicityGraph(model).eval(),
        "deposits": DepositGraph(model).eval(),
    }
    worst = 0.0
    for trial in range(trials):
        x = example_inputs(
            model.n_detectors, batch=64, tokens=7, n_pdgs=len(model.pdgs.codes), seed=trial
        )
        with torch.no_grad():
            m_torch, p_torch = graphs["multiplicity"](
                x["pdg_index"], x["kinematics"], x["noise_multiplicity"]
            )
            d_torch = graphs["deposits"](
                x["pdg_index"], x["kinematics"], x["multiplicity"], x["noise_deposits"]
            )
        m_ort, p_ort = sessions["multiplicity"].run(
            None,
            {
                "pdg_index": x["pdg_index"].numpy(),
                "kinematics": x["kinematics"].numpy(),
                "noise": x["noise_multiplicity"].numpy(),
            },
        )
        d_ort = sessions["deposits"].run(
            None,
            {
                "pdg_index": x["pdg_index"].numpy(),
                "kinematics": x["kinematics"].numpy(),
                "multiplicity": x["multiplicity"].numpy(),
                "noise": x["noise_deposits"].numpy(),
            },
        )
        np.testing.assert_allclose(p_ort, p_torch.numpy(), atol=atol)
        differ = [np.mean(m_ort != m_torch.numpy())]
        differ += [np.mean(d_ort[i] != d_torch[i].numpy()) for i in range(2)]
        same = (d_ort[0] == d_torch[0].numpy()) & (d_ort[1] == d_torch[1].numpy())
        np.testing.assert_allclose(d_ort[2][same], d_torch[2].numpy()[same], atol=atol)
        worst = max([worst] + differ)
    return worst


def write_bundle(
    checkpoint,
    folder,
    station_radius=10.0,
    observation_level=0.0,
    threshold=None,
    physics=None,
    check=True,
):
    """Export the checkpoint as a bundle in folder (made if needed; must be empty)."""
    model = Model.load(checkpoint)
    configs = model.configs
    os.makedirs(folder, exist_ok=True)
    if os.listdir(folder):
        raise FileExistsError(f"{folder} isn't empty")

    if threshold is None:
        cuts = (configs.get("subshowers") or {}).get("energy_cuts")
        if not cuts:
            raise ValueError("give --threshold: the config lists no subshowers: energy_cuts")
        threshold = float(max(cuts))  # starts below any of them were trained on

    paths = export_graphs(model, folder)
    pdgs = [int(code) for code in model.pdgs.codes.tolist()]
    _tag_graphs(
        paths,
        {
            "bundle_format": BUNDLE_FORMAT,
            "pdgs": ",".join(map(str, pdgs)),
            "noise_multiplicity": NOISE_MULTIPLICITY,
            "noise_per_deposit": NOISE_PER_DEPOSIT,
            "n_detectors": model.n_detectors,
        },
    )

    centers = model.detector_tokens.centers.cpu().numpy().astype(np.float64)
    with open(os.path.join(folder, "stations.csv"), "w") as f:
        f.write("index,id,x,y,z,radius\n")
        for i, (x, y, z) in enumerate(centers):
            f.write(f"{i},{i},{x:.3f},{y:.3f},{z:.3f},{station_radius}\n")

    manifest = {
        "corsika": {
            "interface_version": 1,
            # the training showers' observation plane, which their histories use
            "frame": {
                "observation_level": float(observation_level),
                "x_axis": [1.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
            },
            "units": {"length": "m", "momentum": "GeV", "time": "s"},
            # kinetic energies below which the model takes a particle, at most
            "trigger": {"em": threshold, "muon": threshold, "hadron": threshold},
            "incident_pdgs": list(pdgs),
            "stations": {"file": "stations.csv", "count": int(len(centers))},
            "physics": dict(physics or TRAINING_PHYSICS),
        },
        "plugin": {
            "name": "simpleshower",
            "bundle_format": BUNDLE_FORMAT,
            "graphs": {name: os.path.basename(path) for name, path in paths.items()},
            # vocabulary index i + 1 is pdgs[i]; index 0 is any other code
            "pdgs": list(pdgs),
            "noise": {
                "rng": "splitmix64",
                "multiplicity": NOISE_MULTIPLICITY,
                "per_deposit": NOISE_PER_DEPOSIT,
            },
            "outputs": ["station", "pdg", "kinetic_energy"],  # no time or direction yet
        },
        "source": {
            "checkpoint": os.path.abspath(checkpoint),
            "sha256": _sha256(checkpoint),
            "run_name": configs.get("run_name"),
            "showers": configs.get("showers"),
            "subshowers": configs.get("subshowers"),
            "detectors": configs.get("detectors"),
        },
    }
    with open(os.path.join(folder, "manifest.yaml"), "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    if check:
        worst = check_graphs(model, folder)
        if worst > 1e-3:
            raise RuntimeError(f"ONNX Runtime and PyTorch disagree on {worst:.2%} of draws")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export a SimpleShower as a CORSIKA bundle.")
    parser.add_argument("checkpoint")
    parser.add_argument("bundle", help="output directory, empty or new")
    parser.add_argument("--station-radius", type=float, default=10.0, help="m, as trained")
    parser.add_argument(
        "--observation-level",
        type=float,
        default=0.0,
        help="m: the training showers' --observation-level, which fixes the frame",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="GeV: largest kinetic energy to take (default: the config's largest cut)",
    )
    parser.add_argument("--physics", help="yaml of the training runs' settings")
    parser.add_argument("--no-check", action="store_true", help="skip the ORT check")
    parser.add_argument("--force", action="store_true", help="empty the bundle first")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.force and os.path.isdir(args.bundle):
        shutil.rmtree(args.bundle)
    physics = None
    if args.physics:
        with open(args.physics) as f:
            physics = yaml.safe_load(f)
    write_bundle(
        args.checkpoint,
        args.bundle,
        station_radius=args.station_radius,
        observation_level=args.observation_level,
        threshold=args.threshold,
        physics=physics,
        check=not args.no_check,
    )
    print(f"Wrote {args.bundle}")


if __name__ == "__main__":
    sys.exit(main())
