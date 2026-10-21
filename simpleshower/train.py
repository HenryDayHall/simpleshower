"""
Train SimpleShower on the subshowers a config describes, as its detector
array sees them.

    python -m simpleshower.train configs/default.yaml

The config's storage_path, showers and subshowers sections say which
subshower sets to use, as in subshowers' configs (subshowers.datasets finds
the ones that are made), and its detectors section the array. Everything is
written to training: output_dir, by default <showers folder>/simpleshower/<run_name>:

    examples/                the training examples of each subshower set, cached
    config.yaml              the config as trained, with model: pdgs filled in
    checkpoint_best.pt       the model after the epoch with the lowest validation loss
    checkpoint_last.pt       the model after the last epoch
    validation_examples.npz  the held-out subshowers (see generate.py --examples)
    log.csv                  losses and accuracies after each epoch
"""

import argparse
import copy
import csv
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

from simpleshower.data import MODEL_COLUMNS, DataError, examples_from_config
from simpleshower.model import Model

# training settings that the config's training section can change
DEFAULTS = {
    "output_dir": None,
    "validation_fraction": 0.1,
    "max_subshowers_per_set": None,  # subshowers taken at random from each set
    "epochs": 20,
    "steps_per_epoch": None,  # default: one pass over the subshowers with deposits
    # subshowers per step for the multiplicity loss: enough for one pass over
    # the training subshowers each epoch, but no more than this
    "batch_size": 256,
    "deposit_batch_size": 32,  # subshowers with deposits per step, for the rest
    "max_deposits_per_subshower": 64,  # deposits trained on per subshower per step
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-2,
    "warmup_steps": 100,
    "gradient_clip": 1.0,
    "loss_weights": {"multiplicity": 1.0, "detector": 1.0, "pdg": 1.0, "energy": 1.0},
    "seed": 0,
    "device": "auto",
}


def training_settings(config):
    settings = copy.deepcopy(DEFAULTS)
    settings.update((config.get("training") or {}))
    return settings


def output_folder(config, settings):
    if settings["output_dir"]:
        return os.path.abspath(settings["output_dir"])
    from subshowers.generation import showers_folder

    run_name = config.get("run_name") or "default"
    return os.path.join(showers_folder(config), "simpleshower", str(run_name))


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Tensors:
    """Examples as torch tensors, and the batches made from them."""

    def __init__(self, examples, device):
        self.device = device
        self.n = len(examples)
        self.incident = {
            name: torch.as_tensor(
                examples.incident[name],
                dtype=torch.long if name == "pdg" else torch.float32,
                device=device,
            )
            for name in MODEL_COLUMNS
        }
        self.counts = examples.multiplicity
        self.multiplicity = torch.as_tensor(self.counts, device=device)
        self.offsets = examples.deposit_offsets
        self.detected = np.flatnonzero(self.counts > 0)
        self.detector = torch.as_tensor(examples.deposit_detector, device=device)
        self.pdg = torch.as_tensor(examples.deposit_pdg, device=device)
        self.log_energy = torch.as_tensor(
            np.log10(np.maximum(examples.deposit_energy, 1e-30)),
            dtype=torch.float32,
            device=device,
        )

    def incident_of(self, indices):
        index = torch.as_tensor(indices, device=self.device)
        return {name: column[index] for name, column in self.incident.items()}

    def deposits_of(self, indices, max_deposits, rng):
        """
        The deposit batch Model.loss takes, for subshowers that have deposits.
        A subshower with more than max_deposits contributes a random
        max_deposits of them, weighted up so it counts as much as all of them.
        """
        counts = self.counts[indices]
        chosen = [
            np.arange(count)
            if max_deposits is None or count <= max_deposits
            else np.sort(rng.choice(count, max_deposits, replace=False))
            for count in counts
        ]
        shape = (len(indices), max(len(ranks) for ranks in chosen))
        positions = np.zeros(shape, dtype=np.int64)
        rank = np.zeros(shape, dtype=np.float32)
        weight = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        for row, (i, count, ranks) in enumerate(zip(indices, counts, chosen)):
            k = len(ranks)
            positions[row, :k] = self.offsets[i] + ranks
            rank[row, :k] = (ranks + 0.5) / count
            weight[row, :k] = count / k
            mask[row, :k] = True
        positions = torch.as_tensor(positions, device=self.device)
        return {
            "incident": self.incident_of(indices),
            "multiplicity": torch.as_tensor(counts, device=self.device),
            "rank": torch.as_tensor(rank, device=self.device),
            "weight": torch.as_tensor(weight, device=self.device),
            "mask": torch.as_tensor(mask, device=self.device),
            "detector": self.detector[positions],
            "pdg": self.pdg[positions],
            "log_energy": self.log_energy[positions],
        }


def shuffled_batches(indices, batch_size, rng):
    """Batches of indices, reshuffled after each pass, for ever."""
    while True:
        order = rng.permutation(indices)
        for start in range(0, len(order), batch_size):
            yield order[start : start + batch_size]


@torch.no_grad()
def evaluate(model, tensors, settings):
    """Losses and accuracies over every subshower of tensors."""
    model.eval()
    sums, norms = defaultdict(float), defaultdict(float)
    everything = np.arange(tensors.n)
    for start in range(0, tensors.n, settings["batch_size"]):
        indices = everything[start : start + settings["batch_size"]]
        detectors = model.detector_tokens()
        out = model.multiplicity_losses(
            tensors.incident_of(indices), tensors.multiplicity[indices], detectors, True
        )
        for name, value in out.items():
            sums[name] += value.item() * len(indices)
            norms[name] += len(indices)
    rng = np.random.default_rng(0)
    for start in range(0, len(tensors.detected), settings["deposit_batch_size"]):
        indices = tensors.detected[start : start + settings["deposit_batch_size"]]
        deposits = tensors.deposits_of(
            indices, settings["max_deposits_per_subshower"], rng
        )
        weight = (deposits["weight"] * deposits["mask"]).sum().item()
        out = model.deposit_losses(deposits, model.detector_tokens(), True)
        for name, value in out.items():
            sums[name] += value.item() * weight
            norms[name] += weight
    model.train()
    result = {name: sums[name] / norms[name] for name in sums}
    result["total"] = model.total_loss(result)
    return result


def learning_rate_factor(step, warmup, total):
    """Linear warmup, then a cosine down to a tenth."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def describe(losses, names=None):
    names = names or [n for n in Model.LOSSES + ["total"] if n in losses]
    return " ".join(f"{name} {losses[name]:.3f}" for name in names)


def train(config, output_dir=None, log=print):
    """Train a model as the config says; returns the output folder."""
    config = copy.deepcopy(config)
    settings = training_settings(config)
    if output_dir is not None:
        settings["output_dir"] = output_dir
    folder = output_folder(config, settings)
    config["training"] = settings
    os.makedirs(folder, exist_ok=True)
    seed = settings["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device(settings["device"])
    log(f"Output: {folder}")
    log(f"Device: {device}")

    # ------------------------------------------------------------ data
    examples = examples_from_config(
        config,
        os.path.join(folder, "examples"),
        max_per_set=settings["max_subshowers_per_set"],
        seed=seed,
        log=log,
    )
    training, validation = examples.split(settings["validation_fraction"], seed)
    log(
        f"{len(examples)} subshowers, {examples.detected.sum()} with deposits,"
        f" {len(examples.deposit_energy)} deposits; {len(training)} for training,"
        f" {len(validation)} held out"
    )
    if not training.detected.any():
        log("None of the training subshowers have deposits: only multiplicity is learnt")
    validation.save(os.path.join(folder, "validation_examples.npz"))

    model_configs = config.setdefault("model", {})
    if not model_configs.get("pdgs"):
        model_configs["pdgs"] = [int(code) for code in training.pdg_codes()]
        log(f"PDG codes, from the data: {model_configs['pdgs']}")
    max_multiplicity = int(examples.multiplicity.max())
    limit = (model_configs.get("multiplicity_predictor") or {}).get(
        "max_multiplicity", 1000
    )
    if max_multiplicity > limit:
        log(
            f"Some subshowers have {max_multiplicity} deposits, more than"
            f" model: multiplicity_predictor: max_multiplicity ({limit});"
            " they're counted in its last class"
        )
    with open(os.path.join(folder, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # ----------------------------------------------------------- model
    model = Model(config)
    on_cpu = Tensors(training, torch.device("cpu"))
    model.fit_normalisation(on_cpu.incident, on_cpu.multiplicity)
    model.to(device)
    train_tensors = Tensors(training, device)
    validation_tensors = Tensors(validation, device)
    n_parameters = sum(p.numel() for p in model.parameters())
    log(f"Model: {n_parameters} parameters, {model.n_detectors} detectors")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    detected = train_tensors.detected
    steps_per_epoch = settings["steps_per_epoch"] or max(
        1,
        math.ceil(len(detected) / settings["deposit_batch_size"])
        if len(detected)
        else math.ceil(len(training) / settings["batch_size"]),
    )
    total_steps = steps_per_epoch * settings["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, settings["warmup_steps"], total_steps),
    )
    # seeing each subshower about as often for the multiplicity as for the
    # deposits; many more passes over them would overfit the multiplicity
    batch_size = max(
        1, min(settings["batch_size"], math.ceil(len(training) / steps_per_epoch))
    )
    subshower_batches = shuffled_batches(np.arange(len(training)), batch_size, rng)
    deposit_batches = (
        shuffled_batches(detected, settings["deposit_batch_size"], rng)
        if len(detected)
        else None
    )
    log(
        f"Training: {settings['epochs']} epochs of {steps_per_epoch} steps, each with"
        f" {batch_size} subshowers for the multiplicity and"
        f" {settings['deposit_batch_size'] if len(detected) else 0} with deposits"
        " for the rest"
    )

    # ---------------------------------------------------------- epochs
    log_path = os.path.join(folder, "log.csv")
    log_columns = None
    best = math.inf
    started = time.monotonic()
    for epoch in range(1, settings["epochs"] + 1):
        sums = defaultdict(float)
        for _ in range(steps_per_epoch):
            indices = next(subshower_batches)
            deposits = None
            if deposit_batches is not None:
                deposits = train_tensors.deposits_of(
                    next(deposit_batches), settings["max_deposits_per_subshower"], rng
                )
            losses = model.loss(
                train_tensors.incident_of(indices),
                train_tensors.multiplicity[indices],
                deposits,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if settings["gradient_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            scheduler.step()
            for name, value in losses.items():
                sums[name] += value.item()
        train_losses = {name: value / steps_per_epoch for name, value in sums.items()}
        validation_losses = (
            evaluate(model, validation_tensors, settings) if len(validation) else {}
        )
        row = {
            "epoch": epoch,
            "step": epoch * steps_per_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": round(time.monotonic() - started, 1),
            **{f"train_{name}": value for name, value in train_losses.items()},
            **{f"validation_{name}": value for name, value in validation_losses.items()},
        }
        if log_columns is None:
            log_columns = list(row)
            with open(log_path, "w", newline="") as f:
                csv.DictWriter(f, log_columns).writeheader()
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, log_columns, extrasaction="ignore").writerow(row)

        line = f"epoch {epoch:3d}  train: {describe(train_losses)}"
        if validation_losses:
            accuracies = [
                f"{name.replace('_accuracy', '')} {validation_losses[name]:.1%}"
                for name in ["detected_accuracy", "detector_accuracy", "pdg_accuracy"]
                if name in validation_losses
            ]
            line += f"  |  validation: {describe(validation_losses)}"
            line += f"  |  accuracy: {', '.join(accuracies)}"
        log(line + f"  ({row['seconds']:.0f} s)")

        score = validation_losses.get("total", train_losses["total"])
        details = {"epoch": epoch, "validation": validation_losses}
        if score < best:
            best = score
            model.write(os.path.join(folder, "checkpoint_best.pt"), **details)
        model.write(os.path.join(folder, "checkpoint_last.pt"), **details)
    log(f"Done in {time.monotonic() - started:.0f} s; best validation loss {best:.3f}")
    log(f"Checkpoints in {folder}")
    return folder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config_file", help="yaml config, e.g. configs/default.yaml")
    parser.add_argument("--output-dir", help="instead of the config's training: output_dir")
    parser.add_argument("--epochs", type=int, help="instead of the config's")
    parser.add_argument("--device", help="cpu, cuda, ... instead of the config's")
    parser.add_argument(
        "--max-subshowers-per-set",
        type=int,
        help="use this many subshowers at random from each set, for a quick run",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    training = config.setdefault("training", None) or {}
    for key in ["epochs", "device", "max_subshowers_per_set"]:
        if getattr(args, key) is not None:
            training[key] = getattr(args, key)
    config["training"] = training
    from subshowers.detector import DetectorConfigError
    from subshowers.generation import ConfigError

    try:
        train(config, output_dir=args.output_dir)
    except (DataError, DetectorConfigError, ConfigError) as error:
        sys.exit(f"Can't train from {args.config_file}: {error}")
    except KeyError as error:
        sys.exit(f"Can't train from {args.config_file}: it has no {error} entry")


if __name__ == "__main__":
    main()
