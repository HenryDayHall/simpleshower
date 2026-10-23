"""
Compare a trained SimpleShower with the dataset, on the held-out subshowers.

    python scripts/compare_with_dataset.py CHECKPOINT [--examples FILE] [--output-dir DIR]

For each held-out subshower (by default validation_examples.npz, next to the
checkpoint), the model predicts a subshower from the same incident particle.
It prints numbers comparing the two populations, and writes

    distributions.png  subshowers' observables.distributions of both, as
                       subshowers' README describes for generative models
    examples.png       a few held-out subshowers next to the model's
                       predictions for them

into --output-dir (by default the checkpoint's folder).
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

# run as scripts/compare_with_dataset.py, simpleshower is the folder above
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simpleshower.data import Examples  # noqa: E402
from simpleshower.generate import Generator, from_examples  # noqa: E402


class ArrayPositions:
    """What subshowers.observables needs of an array: where its detectors are."""

    def __init__(self, centers):
        self.centers = np.asarray(centers)

    def __len__(self):
        return len(self.centers)


def landing_point(subshower, height):
    """Where the start particle's straight track reaches the given height."""
    direction = subshower.start_direction / np.linalg.norm(subshower.start_direction)
    distance = max(subshower.start_position[2] - height, 0.0) / max(-direction[2], 0.05)
    return subshower.start_position[:2] + direction[:2] * distance


def summary(truth, predicted):
    """Rows of (what, dataset, model); nan where there's nothing to show."""

    def mean(values):
        return float(np.mean(values)) if len(values) else float("nan")

    def median_deposited(population):
        totals = [s.total_detected_energy for s in population if s.multiplicity]
        return float(np.median(totals)) if totals else float("nan")

    rows = [
        (name, mean([get(s) for s in truth]), mean([get(s) for s in predicted]))
        for name, get in [
            ("fraction with deposits", lambda s: s.multiplicity > 0),
            ("deposits per subshower", lambda s: s.multiplicity),
            ("detectors hit per subshower", lambda s: s.n_hit_detectors),
            ("energy deposited per subshower [GeV]", lambda s: s.total_detected_energy),
        ]
    ]
    rows.append(
        (
            "median energy deposited, of those with deposits [GeV]",
            median_deposited(truth),
            median_deposited(predicted),
        )
    )
    # where both have deposits: are the model's in the detectors the dataset's hit?
    shared = [
        np.isin(made.deposit_detector, real.deposit_detector).mean()
        for real, made in zip(truth, predicted)
        if real.multiplicity and made.multiplicity
    ]
    rows.append(
        ("of the model's deposits, in a detector the dataset hit", float("nan"), mean(shared))
    )
    return rows


def pdg_fractions(population):
    pdgs = np.concatenate([s.deposit_pdg for s in population] or [np.zeros(0, int)])
    codes, counts = np.unique(pdgs, return_counts=True)
    return {int(c): n / counts.sum() for c, n in zip(codes, counts)}


def plot_examples(truth, predictions, centers, height, path, n_examples=4):
    """A row per held-out subshower: the dataset's, then the model's draws."""
    chosen = [i for i, s in enumerate(truth) if s.multiplicity > 1][:n_examples]
    if not chosen:
        chosen = [i for i, s in enumerate(truth) if s.multiplicity][:n_examples]
    if not chosen:
        return False
    n_draws = len(predictions)
    fig, axarr = plt.subplots(
        len(chosen), 1 + n_draws, figsize=(4.2 * (1 + n_draws), 4 * len(chosen)), squeeze=False
    )
    for row, i in enumerate(chosen):
        panels = [("dataset", truth[i])] + [
            (f"model, draw {k + 1}", draws[i]) for k, draws in enumerate(predictions)
        ]
        energies = [s.energy_per_detector for _, s in panels]
        positive = np.concatenate([e[e > 0] for e in energies])
        norm = LogNorm(positive.min(), positive.max()) if len(positive) else None
        landing = landing_point(truth[i], height)
        hit_anywhere = np.flatnonzero(np.sum(energies, axis=0) > 0)
        focus = np.vstack([centers[hit_anywhere, :2], landing[None, :]])
        middle = (focus.min(0) + focus.max(0)) / 2
        half = max((focus.max(0) - focus.min(0)).max() / 2 * 1.3, 600.0)
        for column, (title, subshower) in enumerate(panels):
            ax = axarr[row, column]
            energy = subshower.energy_per_detector
            hit = energy > 0
            ax.scatter(centers[~hit, 0], centers[~hit, 1], s=12, c="lightgrey")
            points = ax.scatter(
                centers[hit, 0], centers[hit, 1], s=60, c=energy[hit], cmap="viridis",
                norm=norm, edgecolors="k", linewidths=0.5,
            )
            ax.scatter(*landing, s=160, marker="*", c="white", edgecolors="k", zorder=3)
            ax.set_xlim(middle[0] - half, middle[0] + half)
            ax.set_ylim(middle[1] - half, middle[1] + half)
            ax.set_aspect("equal")
            ax.set_title(
                f"{title}\n{subshower.multiplicity} deposits, {subshower.n_hit_detectors} detectors,"
                f" {subshower.total_detected_energy:.3g} GeV",
                fontsize=9,
            )
            ax.set_xlabel("x [m]")
            if column == 0:
                s = truth[i]
                ax.set_ylabel(
                    f"PDG {s.start_pdg}, {s.start_kinetic_energy:.3g} GeV,"
                    f" from {s.start_position[2] / 1000:.1f} km\ny [m]"
                )
            if hit.any() and column == len(panels) - 1:
                fig.colorbar(points, ax=ax, label="energy deposited [GeV]")
    fig.suptitle(
        "Held-out subshowers and the model's predictions from the same particle"
        " (star: where its track reaches the array)",
        y=0.998,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--examples", help="default: validation_examples.npz by the checkpoint")
    parser.add_argument("--output-dir", help="default: the checkpoint's folder")
    parser.add_argument("--draws", type=int, default=2, help="model draws per example in examples.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    folder = os.path.dirname(os.path.abspath(args.checkpoint))
    examples_path = args.examples or os.path.join(folder, "validation_examples.npz")
    output_dir = args.output_dir or folder
    os.makedirs(output_dir, exist_ok=True)
    generator = Generator.load(args.checkpoint)
    examples = Examples.load(examples_path)
    n_detectors = generator.n_detectors
    truth = [from_examples(examples, i, n_detectors) for i in range(len(examples))]

    def predict(seed):
        return generator.predict(
            examples.incident["pdg"],
            np.stack([examples.incident[c] for c in ["x", "y", "z"]], -1),
            np.stack([examples.incident[c] for c in ["px", "py", "pz"]], -1),
            seed=seed,
            start_kinetic_energy=examples.incident["kinetic_energy"],
            start_time=examples.incident["time"],
        )

    predictions = [predict(args.seed + k) for k in range(max(1, args.draws))]
    predicted = predictions[0]

    print(f"{len(examples)} held-out subshowers from {examples_path}")
    print(f"{'':56} {'dataset':>10} {'model':>10}")
    for name, real, made in summary(truth, predicted):
        real_text = "" if np.isnan(real) else f"{real:.4g}"
        print(f"{name:56} {real_text:>10} {made:>10.4g}")
    real_pdgs, made_pdgs = pdg_fractions(truth), pdg_fractions(predicted)
    print("PDG of deposits, fraction:")
    for code in sorted(set(real_pdgs) | set(made_pdgs)):
        print(f"  {code:>8} {real_pdgs.get(code, 0):>49.3f} {made_pdgs.get(code, 0):>10.3f}")

    from subshowers import observables
    from subshowers.responses import Responses

    array = ArrayPositions(generator.detector_centers)
    by_group = {
        "dataset": observables.record_of(
            Responses.from_samples(truth, n_detectors, source="dataset"), array
        ),
        "model": observables.record_of(
            Responses.from_samples(predicted, n_detectors, source="model"), array
        ),
    }
    fig = observables.distributions(
        by_group, title="Held-out subshowers: the dataset and SimpleShower, from the same particles"
    )
    path = os.path.join(output_dir, "distributions.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Wrote {path}")

    height = float(np.mean(generator.detector_centers[:, 2]))
    path = os.path.join(output_dir, "examples.png")
    if plot_examples(truth, predictions, generator.detector_centers, height, path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
