"""
Predict single subshowers with a trained SimpleShower.

    python -m simpleshower.generate CHECKPOINT --pdg 11 --position X Y Z --momentum PX PY PZ

prints the deposits that a subshower started by that particle leaves in the
detector array: which detector, which PDG and how much energy. Positions are
in m and momenta in GeV/c, as in the shower histories. To see a held-out
subshower next to the model's prediction for it,

    python -m simpleshower.generate CHECKPOINT --examples OUTPUT_DIR/validation_examples.npz --index 3

--samples N draws N subshowers from the same particle, --greedy takes the
most likely one (highest attention weights, most likely PDG, median energy),
--seed fixes the draws, and --output writes them to a json file as well.

From Python:

    from simpleshower.generate import Generator
    generator = Generator.load("checkpoint_best.pt")
    subshower = generator.subshower(pdg=11, position=(x, y, z), momentum=(px, py, pz))
    subshower.deposit_detector, subshower.deposit_pdg, subshower.deposit_energy
    subshower.energy_per_detector  # as subshowers' sampler gives it

A GeneratedSubshower has the attributes that subshowers'
Responses.from_samples reads, so the model's subshowers can be compared with
the dataset's with subshowers.observables.
"""

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
import torch

from simpleshower.data import Examples, kinetic_energy
from simpleshower.model import Model


@dataclass
class GeneratedSubshower:
    """One subshower: its start particle and its deposits."""

    start_pdg: int
    start_position: np.ndarray  # (3,) m
    # (3,) the start particle's momentum, GeV/c, as the sampler's DetectedSample has it
    start_direction: np.ndarray
    start_kinetic_energy: float  # GeV
    start_time: float
    deposit_detector: np.ndarray  # (M,) index into the array
    deposit_pdg: np.ndarray  # (M,)
    deposit_energy: np.ndarray  # (M,) GeV
    n_detectors: int
    source: str = "model"

    @property
    def multiplicity(self):
        return len(self.deposit_detector)

    @property
    def n_detected_leaves(self):
        # one deposit per leaf and detector it hits
        return self.multiplicity

    @property
    def energy_per_detector(self):
        return np.bincount(
            self.deposit_detector, weights=self.deposit_energy, minlength=self.n_detectors
        )

    @property
    def total_detected_energy(self):
        return float(self.deposit_energy.sum())

    @property
    def n_hit_detectors(self):
        return len(np.unique(self.deposit_detector))

    def as_dict(self):
        return {
            "source": self.source,
            "start_pdg": int(self.start_pdg),
            "start_position": [float(v) for v in self.start_position],
            "start_momentum": [float(v) for v in self.start_direction],
            "start_kinetic_energy": float(self.start_kinetic_energy),
            "start_time": float(self.start_time),
            "deposits": [
                {"detector": int(d), "pdg": int(p), "energy": float(e)}
                for d, p, e in zip(
                    self.deposit_detector, self.deposit_pdg, self.deposit_energy
                )
            ],
        }


class Generator:
    """A trained model, and what's needed to make subshowers with it."""

    def __init__(self, model, device=None):
        self.model = model.eval()
        self.device = torch.device(device) if device else next(model.parameters()).device
        self.detector_centers = model.detector_tokens.centers.cpu().numpy()

    @classmethod
    def load(cls, path, device="cpu"):
        model = Model.load(path, map_location=device).to(device)
        return cls(model, device)

    @property
    def n_detectors(self):
        return self.model.n_detectors

    def random_generator(self, seed):
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def predict(
        self,
        pdg,
        position,
        momentum,
        greedy=False,
        seed=None,
        start_kinetic_energy=None,
        start_time=None,
        batch_size=256,
    ):
        """
        One subshower for each of B incident particles: pdg (B,), position
        (B, 3) in m, momentum (B, 3) in GeV/c. The kinetic energies are
        worked out from the momenta and PDGs unless given; they aren't used
        by the model.
        """
        pdg = np.asarray(pdg, dtype=np.int64).reshape(-1)
        position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
        momentum = np.asarray(momentum, dtype=np.float64).reshape(-1, 3)
        if start_kinetic_energy is None:
            start_kinetic_energy = [kinetic_energy(p, m) for p, m in zip(pdg, momentum)]
        start_kinetic_energy = np.broadcast_to(start_kinetic_energy, pdg.shape)
        start_time = np.broadcast_to(0.0 if start_time is None else start_time, pdg.shape)
        generator = self.random_generator(seed)
        subshowers = []
        for start in range(0, len(pdg), batch_size):
            chunk = slice(start, start + batch_size)

            def tensor(values, dtype=torch.float32):
                return torch.as_tensor(values, dtype=dtype, device=self.device)

            incident = {"pdg": tensor(pdg[chunk], torch.long)}
            for axis, name in enumerate(["x", "y", "z"]):
                incident[name] = tensor(position[chunk, axis])
            for axis, name in enumerate(["px", "py", "pz"]):
                incident[name] = tensor(momentum[chunk, axis])
            predicted = self.model.generate(incident, greedy=greedy, generator=generator)
            for i, deposits in zip(range(start, len(pdg)), predicted):
                subshowers.append(
                    GeneratedSubshower(
                        start_pdg=int(pdg[i]),
                        start_position=position[i].copy(),
                        start_direction=momentum[i].copy(),
                        start_kinetic_energy=float(start_kinetic_energy[i]),
                        start_time=float(start_time[i]),
                        deposit_detector=deposits["detector"],
                        deposit_pdg=deposits["pdg"],
                        deposit_energy=deposits["energy"],
                        n_detectors=self.n_detectors,
                    )
                )
        return subshowers

    def subshower(self, pdg, position, momentum, greedy=False, seed=None, **kwargs):
        """One subshower, started by one particle."""
        return self.subshowers(pdg, position, momentum, 1, greedy, seed, **kwargs)[0]

    def subshowers(self, pdg, position, momentum, n=1, greedy=False, seed=None, **kwargs):
        """n subshowers, all started by the same particle."""
        return self.predict(
            np.full(n, pdg),
            np.tile(np.asarray(position, dtype=np.float64), (n, 1)),
            np.tile(np.asarray(momentum, dtype=np.float64), (n, 1)),
            greedy,
            seed,
            **kwargs,
        )


def from_examples(examples, i, n_detectors):
    """Subshower i of some Examples, as a GeneratedSubshower."""
    detector, pdg, energy = examples.deposits(i)
    incident = {name: column[i] for name, column in examples.incident.items()}
    dx, dy = examples.shift[i]
    # the incident position is the shifted subshower's, not the history's
    shifted = f", shifted ({dx:.1f}, {dy:.1f}) m" if dx or dy else ""
    return GeneratedSubshower(
        start_pdg=int(incident["pdg"]),
        start_position=np.array([incident[c] for c in ["x", "y", "z"]]),
        start_direction=np.array([incident[c] for c in ["px", "py", "pz"]]),
        start_kinetic_energy=float(incident["kinetic_energy"]),
        start_time=float(incident["time"]),
        deposit_detector=detector,
        deposit_pdg=pdg,
        deposit_energy=energy,
        n_detectors=n_detectors,
        source=(
            f"dataset ({examples.source[i]}, start particle"
            f" {examples.start_index[i]}{shifted})"
        ),
    )


# ---------------------------------------------------------------- printing


def describe_incident(subshower):
    x, y, z = subshower.start_position
    px, py, pz = subshower.start_direction
    return (
        f"PDG {subshower.start_pdg} at ({x:.1f}, {y:.1f}, {z:.1f}) m,"
        f" momentum ({px:.4g}, {py:.4g}, {pz:.4g}) GeV/c,"
        f" kinetic energy {subshower.start_kinetic_energy:.4g} GeV"
    )


def describe_deposits(subshower, detector_centers, max_rows=40):
    if subshower.multiplicity == 0:
        return ["  no deposits"]
    lines = [
        f"  {subshower.multiplicity} deposits in {subshower.n_hit_detectors} detectors,"
        f" {subshower.total_detected_energy:.4g} GeV in all",
        f"  {'detector':>8} {'x [m]':>10} {'y [m]':>10} {'z [m]':>8} {'PDG':>6} {'energy [GeV]':>13}",
    ]
    order = np.lexsort((-subshower.deposit_energy, subshower.deposit_detector))
    for row in order[:max_rows]:
        detector = subshower.deposit_detector[row]
        x, y, z = detector_centers[detector]
        lines.append(
            f"  {detector:>8} {x:>10.1f} {y:>10.1f} {z:>8.1f}"
            f" {subshower.deposit_pdg[row]:>6} {subshower.deposit_energy[row]:>13.4g}"
        )
    if subshower.multiplicity > max_rows:
        lines.append(f"  ... and {subshower.multiplicity - max_rows} more")
    return lines


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict single subshowers with a trained SimpleShower."
    )
    parser.add_argument("checkpoint", help="a checkpoint written by train.py")
    parser.add_argument("--pdg", type=int, help="PDG code of the incident particle")
    parser.add_argument("--position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="m")
    parser.add_argument(
        "--momentum", type=float, nargs=3, metavar=("PX", "PY", "PZ"), help="GeV/c"
    )
    parser.add_argument("--time", type=float, default=0.0, help="start time, kept as given")
    parser.add_argument(
        "--examples", help="Examples file (e.g. validation_examples.npz) to take a particle from"
    )
    parser.add_argument("--index", type=int, default=0, help="which subshower of --examples")
    parser.add_argument("--samples", type=int, default=1, help="subshowers to draw")
    parser.add_argument("--greedy", action="store_true", help="the most likely subshower")
    parser.add_argument("--seed", type=int, help="for the draws")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="also write the subshowers to this json file")
    parser.add_argument("--max-rows", type=int, default=40, help="deposits printed per subshower")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    generator = Generator.load(args.checkpoint, args.device)
    truth = None
    if args.examples:
        examples = Examples.load(args.examples)
        if not 0 <= args.index < len(examples):
            sys.exit(f"--index must be below {len(examples)}, the number of examples")
        truth = from_examples(examples, args.index, generator.n_detectors)
        pdg, position, momentum = truth.start_pdg, truth.start_position, truth.start_direction
        extra = {
            "start_kinetic_energy": truth.start_kinetic_energy,
            "start_time": truth.start_time,
        }
    else:
        if args.pdg is None or args.position is None or args.momentum is None:
            sys.exit("Give --pdg, --position and --momentum, or --examples")
        pdg, position, momentum = args.pdg, args.position, args.momentum
        extra = {"start_time": args.time}

    n = 1 if args.greedy else args.samples
    subshowers = generator.subshowers(
        pdg, position, momentum, n=n, greedy=args.greedy, seed=args.seed, **extra
    )
    print(f"Incident particle: {describe_incident(subshowers[0])}")
    if truth is not None:
        print(f"\nIn the {truth.source}:")
        print("\n".join(describe_deposits(truth, generator.detector_centers, args.max_rows)))
    for number, subshower in enumerate(subshowers, start=1):
        title = "Most likely subshower" if args.greedy else f"Predicted subshower {number}"
        print(f"\n{title}:")
        print(
            "\n".join(describe_deposits(subshower, generator.detector_centers, args.max_rows))
        )
    if args.output:
        records = [s.as_dict() for s in ([truth] if truth else []) + subshowers]
        with open(args.output, "w") as f:
            json.dump(records, f, indent=1)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
