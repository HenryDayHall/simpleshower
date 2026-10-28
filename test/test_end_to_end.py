"""
Make two small fake showers, train a small model on them for a few steps,
and predict subshowers with it, checking each stage gives what the next needs;
then train one on the same showers in two showers folders. The runs share
one examples cache.

    pytest test

subshowers has to be importable from a clone of its repository, as
scripts/make_fake_data.py uses its test/fake_data.py.
"""

import copy
import os
import sys

import numpy as np
import pytest
import torch
import yaml

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY)
sys.path.insert(0, os.path.join(REPOSITORY, "scripts"))

import make_fake_data  # noqa: E402
from simpleshower import generate, train  # noqa: E402
from simpleshower.data import (  # noqa: E402
    DataError,
    Examples,
    examples_from_config,
    shower_folders,
)
from simpleshower.model import Model  # noqa: E402


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    with open(os.path.join(REPOSITORY, "configs", "example.yaml")) as f:
        config = yaml.safe_load(f)
    folder = tmp_path_factory.mktemp("simpleshower")
    config["storage_path"] = f"{folder}/data/"
    config["showers"]["subfolders"] = ["e5e3_0", "e8e3_1"]
    model = config["model"]
    for section in ["multiplicity_token", "detector_tokens", "deposit_token"]:
        model[section]["embedding_dim"] = 32
    model["deposit_token"]["rounds"] = 2
    config["training"].update(
        output_dir=str(folder / "run"), epochs=2, steps_per_epoch=5, device="cpu"
    )
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    make_fake_data.main([str(path)])
    return config


@pytest.fixture(scope="module")
def run(config):
    return train.train(config, log=lambda text: None)


def test_examples_match_the_sampler(run):
    # data.examples_from_samples checks that each subshower's deposits add up
    # to the sampler's energy_per_detector; here, that they're in order
    examples = Examples.load(os.path.join(run, "validation_examples.npz"))
    assert len(examples) > 0
    for i in np.flatnonzero(examples.detected):
        _, _, energy = examples.deposits(i)
        assert np.all(np.diff(energy) <= 0)


def test_checkpoints_reload(run):
    for name in ["checkpoint_best.pt", "checkpoint_last.pt"]:
        model = Model.load(os.path.join(run, name))
        assert model.n_detectors == 400
    with open(os.path.join(run, "config.yaml")) as f:
        saved = yaml.safe_load(f)
    assert saved["model"]["pdgs"], "train.py should fill in the PDG codes"
    # subshowers are shifted by up to half the array's width, unless told otherwise
    assert saved["training"]["shift"] is True and saved["training"]["max_shift"] is None
    with open(os.path.join(run, "log.csv")) as f:
        assert len(f.readlines()) == 3  # header and two epochs
    # kept: before training, and after epoch 2
    kept = sorted(os.listdir(os.path.join(run, "checkpoints")))
    assert kept == ["epoch_0000.pt", "epoch_0002.pt"]
    assert Model.load(os.path.join(run, "checkpoints", "epoch_0000.pt")).n_detectors == 400


def test_training_until_annealed(config, tmp_path, monkeypatch):
    """
    With epochs: -1, training stops once the learning rate has annealed: here
    the held-out losses never fall after the first epoch, so it's fixed for 3
    epochs, then anneals over 3.
    """
    import csv

    flat = {"multiplicity": 1.0, "detector": 2.0, "pdg": 1.6, "energy": 0.1, "total": 4.7}
    monkeypatch.setattr(train, "evaluate", lambda *args: dict(flat))
    until = copy.deepcopy(config)
    until["training"].update(
        output_dir=str(tmp_path / "run"),
        epochs=-1,
        plateau_epochs=2,
        cosine_epochs=3,
        warmup_steps=2,
    )
    lines = []
    folder = train.train(until, log=lines.append)
    with open(os.path.join(folder, "log.csv")) as f:
        rows = list(csv.DictReader(f))
    assert [row["phase"] for row in rows] == ["fixed"] * 3 + ["annealing"] * 3
    rates = [float(row["learning_rate"]) for row in rows]
    learning_rate = until["training"]["learning_rate"]
    assert rates[:3] == [learning_rate] * 3
    assert rates[3] > rates[4] > rates[5] > 0.1 * learning_rate
    assert any("annealing the learning rate over the next 3" in line for line in lines)
    assert any("has annealed, after 6 epochs" in line for line in lines)
    kept = sorted(os.listdir(os.path.join(folder, "checkpoints")))
    assert kept == ["epoch_0000.pt", "epoch_0002.pt", "epoch_0004.pt"]
    last = torch.load(os.path.join(folder, "checkpoint_last.pt"), weights_only=False)
    assert last["epoch"] == 6 and last["phase"] == "annealing"


def test_single_subshowers(run):
    generator = generate.Generator.load(os.path.join(run, "checkpoint_best.pt"))
    codes = set(generator.model.pdgs.codes.tolist())
    subshowers = generator.subshowers(11, (100.0, -50.0, 4000.0), (0.1, 0.0, -30.0), n=20, seed=1)
    assert len(subshowers) == 20
    for subshower in subshowers:
        assert subshower.energy_per_detector.shape == (400,)
        assert np.all((0 <= subshower.deposit_detector) & (subshower.deposit_detector < 400))
        assert set(subshower.deposit_pdg.tolist()) <= codes
        assert np.all(subshower.deposit_energy > 0)
        assert subshower.start_kinetic_energy == pytest.approx(np.hypot(30.0, 0.1), rel=1e-4)
    # the same seed, the same subshowers
    again = generator.subshowers(11, (100.0, -50.0, 4000.0), (0.1, 0.0, -30.0), n=20, seed=1)
    for first, second in zip(subshowers, again):
        assert np.array_equal(first.deposit_detector, second.deposit_detector)
    greedy = generator.subshower(11, (100.0, -50.0, 4000.0), (0.1, 0.0, -30.0), greedy=True)
    assert greedy.multiplicity >= 0


def test_generation_in_chunks_is_the_same(run):
    model = Model.load(os.path.join(run, "checkpoint_best.pt"))
    incident = {
        "pdg": torch.tensor([11, 22, -13, 11]),
        "x": torch.tensor([0.0, 300.0, -500.0, 1000.0]),
        "y": torch.tensor([0.0, -200.0, 800.0, 100.0]),
        "z": torch.tensor([3000.0, 6000.0, 2000.0, 9000.0]),
        "px": torch.tensor([0.0, 0.5, -0.1, 0.0]),
        "py": torch.tensor([0.0, 0.0, 0.2, 0.3]),
        "pz": torch.tensor([-20.0, -80.0, -5.0, -300.0]),
    }
    together = model.generate(incident, greedy=True)
    in_chunks = model.generate(incident, greedy=True, max_tokens=1)
    for first, second in zip(together, in_chunks):
        assert np.array_equal(first["detector"], second["detector"])
        assert np.allclose(first["energy"], second["energy"], rtol=1e-5)


def test_generate_command_line(run, capsys):
    checkpoint = os.path.join(run, "checkpoint_best.pt")
    examples = os.path.join(run, "validation_examples.npz")
    generate.main([checkpoint, "--examples", examples, "--index", "0", "--samples", "2"])
    printed = capsys.readouterr().out
    assert "In the dataset" in printed and "Predicted subshower 2" in printed
    assert "shifted (" in printed  # from where the history has it
    generate.main([checkpoint, "--pdg", "22", "--position", "0", "0", "5000",
                   "--momentum", "0", "0", "-50", "--greedy"])
    assert "Most likely subshower" in capsys.readouterr().out


def test_unseeded_array_is_refused(config, tmp_path):
    unseeded = copy.deepcopy(config)
    unseeded["detectors"]["generator"].pop("seed")
    with pytest.raises(ValueError, match="no seed"):
        train.train(unseeded, output_dir=str(tmp_path), log=lambda text: None)


def test_runs_share_the_examples_cache(config, run, tmp_path):
    """
    A run with another name, model and training settings samples nothing: it
    reads every set's examples from the cache the first run filled.
    """
    other = copy.deepcopy(config)
    other["training"].update(output_dir=str(tmp_path / "other"), epochs=1, learning_rate=3.0e-4)
    other["model"]["deposit_token"]["rounds"] = 1
    lines = []
    train.train(other, log=lines.append)
    cache = train.examples_folder(config, train.training_settings(config))
    assert cache == os.path.join(os.path.abspath(config["storage_path"]), "simpleshower", "examples")
    assert f"Examples cache: {cache}" in lines
    sets = [line for line in lines if line.startswith("  ") and "subshowers," in line]
    assert len(sets) == 4 and all(line.endswith("(cached)") for line in sets)
    # beside the runs, not in them
    assert not os.path.exists(os.path.join(run, "examples"))
    assert not os.path.exists(os.path.join(tmp_path, "other", "examples"))


def test_run_in_the_examples_cache_is_refused(config):
    clash = copy.deepcopy(config)
    clash["training"]["output_dir"] = train.examples_folder(clash, train.training_settings(clash))
    with pytest.raises(DataError, match="is the examples cache"):
        train.train(clash, log=lambda text: None)


def test_examples_dir_on_the_command_line(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(train, "train", lambda config, output_dir=None: seen.update(config["training"]))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"training": {"epochs": 3}}))
    train.main([str(path), "--examples-dir", "elsewhere"])
    assert seen == {"epochs": 3, "examples_dir": "elsewhere"}


@pytest.fixture(scope="module")
def two_folder_run(config, tmp_path_factory):
    """(config, output folder) of a run on the showers in PDG22 and in PDG11."""
    both = copy.deepcopy(config)
    both["showers"]["folders"] = ["PDG22", "PDG11"]
    folder = tmp_path_factory.mktemp("two_folders")
    both["training"].update(output_dir=str(folder / "run"), epochs=1)
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(both))
    make_fake_data.main([str(path)])  # PDG22's are made already
    return both, train.train(both, log=lambda text: None)


def test_several_shower_folders(config, run, two_folder_run):
    both, folder = two_folder_run
    assert os.path.isfile(os.path.join(folder, "checkpoint_last.pt"))
    # both runs' subshowers, from the cache they share: the two-folder run
    # read PDG22's sets from the one-folder run's files, and added PDG11's
    quiet = {"log": lambda text: None}
    cache = train.examples_folder(config, train.training_settings(config))
    names = os.listdir(cache)
    assert sum(name.startswith("PDG22_") for name in names) == 4
    assert sum(name.startswith("PDG11_") for name in names) == 4
    examples = examples_from_config(both, cache, **quiet)
    alone = examples_from_config(config, cache, **quiet)
    in_pdg22 = np.char.startswith(examples.source, "PDG22/")
    in_pdg11 = np.char.startswith(examples.source, "PDG11/")
    assert np.all(in_pdg22 | in_pdg11)
    # PDG22's are the one-folder run's; PDG11's, though of the same
    # subfolders, aren't taken for repeats of them
    assert np.array_equal(examples.source[in_pdg22], np.char.add("PDG22/", alone.source))
    assert np.array_equal(examples.start_index[in_pdg22], alone.start_index)
    # shifted the same way, whatever else they're trained with
    assert np.array_equal(examples.shift[in_pdg22], alone.shift)
    assert in_pdg11.any()
    # by default, outputs go in the showers folder, or in storage_path for several
    settings = dict(both["training"], output_dir=None)
    storage = os.path.abspath(config["storage_path"])
    assert train.output_folder(config, settings) == os.path.join(
        storage, "PDG22", "simpleshower", "example"
    )
    assert train.output_folder(both, settings) == os.path.join(
        storage, "simpleshower", "example"
    )


def test_showers_section_folders():
    def folders(**showers):
        return shower_folders({"showers": {"subfolders": ["e5e3_0"], **showers}})

    assert folders(folders=["PDG22", "PDG11", "PDG22"]) == ["PDG22", "PDG11"]
    assert folders(folder="PDG22") == ["PDG22"]  # as in subshowers' configs
    for wrong in [
        {"folders": "PDG22"},
        {"folders": []},
        {"folder": "PDG22", "folders": ["PDG11"]},
    ]:
        with pytest.raises(DataError):
            folders(**wrong)
