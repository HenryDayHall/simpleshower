"""
Training examples of subshowers the sampler shifts across the ground: their
deposits, the shifts they record, the settings for them, and their cache.
These don't need torch.

    pytest test/test_data.py

subshowers has to be importable from a clone of its repository, as
scripts/make_fake_data.py uses its test/fake_data.py.
"""

import copy
import dataclasses
import os
import sys

import numpy as np
import pytest
import yaml

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY)
sys.path.insert(0, os.path.join(REPOSITORY, "scripts"))

import make_fake_data  # noqa: E402
from simpleshower.data import (  # noqa: E402
    DataError,
    Examples,
    deposits_of,
    examples_from_config,
    folder_configs,
)


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    """configs/example.yaml, for two small fake showers."""
    with open(os.path.join(REPOSITORY, "configs", "example.yaml")) as f:
        config = yaml.safe_load(f)
    folder = tmp_path_factory.mktemp("data")
    config["storage_path"] = f"{folder}/"
    config["showers"]["subfolders"] = ["e5e3_0", "e8e3_1"]
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    make_fake_data.main([str(path)])
    return config


def history_starts(config, examples):
    """Where each example's start particle is in its shower's history."""
    from subshowers.showers import Reader

    folder = os.path.join(config["storage_path"], config["showers"]["folders"][0])
    readers = {}
    positions = np.zeros((len(examples), 3))
    for i, (source, start) in enumerate(zip(examples.source, examples.start_index)):
        if source not in readers:
            readers[source] = Reader(os.path.join(folder, source))
        positions[i] = readers[source].data.loc[start, ["x", "y", "z"]]
    return positions


def made(config, cache_folder, **options):
    """The examples of the config, and what examples_from_config logged."""
    lines = []
    examples = examples_from_config(config, str(cache_folder), log=lines.append, **options)
    return examples, lines


def test_deposits_of_shifted_samples(config):
    """
    The deposits are those of the leaves where the sampler moved them, so
    they add up to the sampler's energy_per_detector; where the history has
    them, they don't.
    """
    from subshowers import datasets, detector
    from subshowers.sampling import SubshowerSampler

    array = detector.from_config(config)
    folder_config = folder_configs(config)["PDG22"]
    check = next(check for check in datasets.check_subshower_sets(folder_config) if check.ready)
    sampler = SubshowerSampler([check.source()], array, seed=0)  # shifting, by default
    n_detected, n_wrong_unshifted = 0, 0
    for sample in sampler.iter_epoch(shuffle=False):
        assert sample.shift[:2].any()
        detector_index, _, energy = deposits_of(sample, array)
        np.testing.assert_allclose(
            np.bincount(detector_index, weights=energy, minlength=len(array)),
            sample.energy_per_detector,
        )
        if sample.n_hit_detectors:
            n_detected += 1
            # as data.py did before it moved the leaves by the shift
            unshifted = dataclasses.replace(sample, shift=np.zeros(3))
            detector_index, _, energy = deposits_of(unshifted, array)
            n_wrong_unshifted += not np.allclose(
                np.bincount(detector_index, weights=energy, minlength=len(array)),
                sample.energy_per_detector,
            )
    assert n_detected >= 5
    assert n_wrong_unshifted >= n_detected / 2


def test_examples_of_shifted_subshowers(config, tmp_path):
    from subshowers import detector

    examples, lines = made(config, tmp_path / "examples")
    furthest = detector.max_shift(detector.from_config(config))  # half the array's width
    assert f"shifted across the ground by up to {furthest:.6g} m" in "\n".join(lines)
    distances = np.hypot(*examples.shift.T)
    assert np.all((0 < distances) & (distances <= furthest))
    assert np.median(distances) > furthest / 2  # spread over the disc
    # each starts where its start particle is in the history, moved by its shift
    starts = history_starts(config, examples)
    np.testing.assert_allclose(examples.incident["x"], starts[:, 0] + examples.shift[:, 0])
    np.testing.assert_allclose(examples.incident["y"], starts[:, 1] + examples.shift[:, 1])
    np.testing.assert_allclose(examples.incident["z"], starts[:, 2])
    assert examples.detected.sum() >= 20

    # each set's shifts are its own
    sets = [Examples.load(str(path)) for path in sorted((tmp_path / "examples").iterdir())]
    assert len(sets) == 4
    first = [tuple(part.shift[0]) for part in sets]
    assert len(set(first)) == 4
    # and the same each time
    again, _ = made(config, tmp_path / "again")
    np.testing.assert_array_equal(again.shift, examples.shift)
    np.testing.assert_array_equal(again.deposit_detector, examples.deposit_detector)
    # but not with another seed
    other, _ = made(config, tmp_path / "other", seed=1)
    assert not np.array_equal(other.shift, examples.shift)


def test_shift_settings(config, tmp_path):
    # where they were simulated
    examples, lines = made(config, tmp_path / "unshifted", shift=False)
    assert "Subshowers aren't shifted" in "\n".join(lines)
    assert not examples.shift.any()
    starts = history_starts(config, examples)
    for axis, name in enumerate("xyz"):
        np.testing.assert_allclose(examples.incident[name], starts[:, axis])

    # or no further than asked
    examples, lines = made(config, tmp_path / "near", max_shift=100.0)
    distances = np.hypot(*examples.shift.T)
    assert np.all((0 < distances) & (distances <= 100.0))

    for wrong in [{"shift": "yes"}, {"max_shift": -5.0}, {"max_shift": "far"}]:
        with pytest.raises(DataError):
            made(config, tmp_path / "wrong", **wrong)


def test_cache_follows_the_shift_settings(config, tmp_path):
    cache = tmp_path / "examples"

    def how(**options):
        """How each set's examples were got: {"sampled"} or {"cached"}."""
        _, lines = made(config, cache, **options)
        return {
            line.rsplit("(", 1)[-1].rstrip(")").split()[0]
            for line in lines
            if "subshowers," in line
        }

    assert how() == {"sampled"}
    assert how() == {"cached"}
    assert how(max_shift=500.0) == {"sampled"}
    assert how(max_shift=500.0) == {"cached"}
    assert how(shift=False) == {"sampled"}
    assert how(shift=False) == {"cached"}


def test_examples_files(config, tmp_path):
    examples, _ = made(config, tmp_path / "examples")
    path = str(tmp_path / "examples.npz")
    examples.save(path)
    loaded = Examples.load(path)
    np.testing.assert_array_equal(loaded.shift, examples.shift)
    picked = examples.select([3, 0])
    np.testing.assert_array_equal(picked.shift, examples.shift[[3, 0]])
    both = Examples.concatenate([picked, examples])
    np.testing.assert_array_equal(both.shift[2:], examples.shift)

    # files from before examples kept shifts are of unshifted subshowers
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files if name != "shift"}
    arrays["format_version"] = np.array(1)
    old = str(tmp_path / "old.npz")
    np.savez_compressed(old, **arrays)
    loaded = Examples.load(old)
    assert loaded.shift.shape == (len(examples), 2) and not loaded.shift.any()
    np.testing.assert_array_equal(loaded.deposit_detector, examples.deposit_detector)
    # and files from later versions aren't read
    arrays["format_version"] = np.array(3)
    np.savez_compressed(old, **arrays)
    with pytest.raises(ValueError, match="format version 3"):
        Examples.load(old)


def test_subshowers_without_shifts_refused(config, tmp_path, monkeypatch):
    from subshowers import detector

    monkeypatch.delattr(detector, "max_shift")
    with pytest.raises(DataError, match="too old"):
        made(copy.deepcopy(config), tmp_path / "examples")
