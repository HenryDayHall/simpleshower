"""
Make two small fake showers, train a small model on them for a few steps,
and predict subshowers with it, checking each stage gives what the next needs;
then train one on the same showers in two showers folders.

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
    with open(os.path.join(run, "log.csv")) as f:
        assert len(f.readlines()) == 3  # header and two epochs


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
    generate.main([checkpoint, "--pdg", "22", "--position", "0", "0", "5000",
                   "--momentum", "0", "0", "-50", "--greedy"])
    assert "Most likely subshower" in capsys.readouterr().out


def test_unseeded_array_is_refused(config, tmp_path):
    unseeded = copy.deepcopy(config)
    unseeded["detectors"]["generator"].pop("seed")
    with pytest.raises(ValueError, match="no seed"):
        train.train(unseeded, output_dir=str(tmp_path), log=lambda text: None)


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
    # both runs' subshowers, from their caches
    quiet = {"log": lambda text: None}
    examples = examples_from_config(both, os.path.join(folder, "examples"), **quiet)
    alone = examples_from_config(config, os.path.join(run, "examples"), **quiet)
    in_pdg22 = np.char.startswith(examples.source, "PDG22/")
    in_pdg11 = np.char.startswith(examples.source, "PDG11/")
    assert np.all(in_pdg22 | in_pdg11)
    # PDG22's are the one-folder run's; PDG11's, though of the same
    # subfolders, aren't taken for repeats of them
    assert np.array_equal(examples.source[in_pdg22], np.char.add("PDG22/", alone.source))
    assert np.array_equal(examples.start_index[in_pdg22], alone.start_index)
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
