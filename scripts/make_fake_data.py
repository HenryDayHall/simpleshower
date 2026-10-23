"""
Fake showers, for trying SimpleShower out without CORSIKA.

    python scripts/make_fake_data.py configs/example.yaml

makes the showers and subshower sets the config names, in the layout
subshowers expects: each shower in the showers section is grown with
subshowers' test/fake_data.py, with the energy (GeV) and seed in its folder
name (e<energy>_<seed>), and split with each of subshowers: energy_cuts.
The config's fake_showers section sets where the showers start and how
much they spread.

fake_data gives every particle a unit vector for its momentum; here the
momenta are scaled to the size their kinetic energy and PDG give, as in a
real history, since the model works energy out from momentum and PDG.
Showers and sets already made are left alone.
"""

import argparse
import importlib.util
import os
import sys
import warnings

import numpy as np
import yaml

# run as scripts/make_fake_data.py, simpleshower is the folder above
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simpleshower.data import mass  # noqa: E402

DEFAULT_FAKE_SHOWERS = {"root_height": [15000.0, 30000.0], "lateral": 0.05}


def load_fake_data():
    """subshowers' test/fake_data.py, from the subshowers repository in use."""
    from subshowers import detector

    repository = os.path.dirname(os.path.dirname(os.path.abspath(detector.__file__)))
    path = os.path.join(repository, "test", "fake_data.py")
    if not os.path.isfile(path):
        raise SystemExit(
            f"Can't find subshowers' test/fake_data.py at {path}: use subshowers"
            " from a clone of its repository (on PYTHONPATH, or pip install -e)"
        )
    spec = importlib.util.spec_from_file_location("subshowers_fake_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_momenta(data):
    """Scale each particle's unit momentum to |p| = sqrt(T (T + 2m))."""
    masses = np.array([mass(pdg) or 0.0 for pdg in data["pdg"]])
    energy = np.asarray(data["kinetic_energy"], dtype=float)
    size = np.sqrt(energy * (energy + 2 * masses))
    for column in ["px", "py", "pz"]:
        data[column] = np.asarray(data[column], dtype=float) * size
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Make the fake showers a config names.")
    parser.add_argument("config_file")
    args = parser.parse_args(argv)
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    from subshowers import generation, subshowers

    fake_data = load_fake_data()
    settings = dict(DEFAULT_FAKE_SHOWERS)
    settings.update(config.get("fake_showers") or {})
    low, high = settings["root_height"]

    for number, shower in enumerate(generation.wanted_showers(config)):
        history = generation.history_path(shower.folder)
        if os.path.isfile(history):
            print(f"{shower.subfolder}: already made")
        else:
            energy = shower.energy or 3.0e4
            seed = number if shower.seed is None else shower.seed
            rng = np.random.default_rng([seed, int(energy)])
            height = rng.uniform(low, high)
            data = fake_data.grow_cascade(rng, energy, height, lateral=settings["lateral"])
            if shower.pdg is not None:
                data["pdg"][0] = shower.pdg
            fake_data.fake_history(shower.folder, physical_momenta(data))
            print(
                f"{shower.subfolder}: {len(data['label'])} particles,"
                f" {energy:.3g} GeV from {height / 1000:.1f} km"
            )

    for subshower_set in generation.wanted_subshower_sets(config):
        wanted = (subshower_set.energy_cut, subshower_set.leaves_only)
        label = f"{subshower_set.shower.subfolder}, energy cut {subshower_set.energy_cut} GeV"
        if generation.saved_settings(subshower_set.output) == wanted:
            print(f"{label}: already made")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            subshowers.run(
                subshower_set.shower.folder,
                subshower_set.energy_cut,
                leaves_only=subshower_set.leaves_only,
                output_path=subshower_set.output,
            )
        print(f"{label}: made")


if __name__ == "__main__":
    main()
