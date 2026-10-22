"""
A bundle made from an untrained model: its graphs against PyTorch, sampling.py's
hits against the plugin's (plugin/, loaded with ctypes as CORSIKA loads it with
dlopen), and the plugin's hits against themselves, batched differently.

    pytest test/test_bundle.py

The plugin parts need the library: build it (see plugin/CMakeLists.txt) and set
SIMPLESHOWER_C8_PLUGIN to its path, or they are skipped.
"""

import ctypes
import os

import numpy as np
import pytest
import torch
import yaml

from simpleshower import bundle as bundle_module
from simpleshower.model import Model
from simpleshower.sampling import BundleSampler, buckets, uniforms

PDGS = [-211, -13, -11, 11, 13, 22, 211, 2112, 2212]
CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    with open(CONFIG) as f:
        configs = yaml.safe_load(f)
    configs["model"]["pdgs"] = PDGS
    # an untrained model draws multiplicities across all its classes; keep them small
    configs["model"]["multiplicity_predictor"]["max_multiplicity"] = 30
    rng = np.random.default_rng(1)
    n = 300
    centers = np.stack(
        [rng.uniform(-2e4, 2e4, n), rng.uniform(-2e4, 2e4, n), rng.uniform(1344, 1619, n)], -1
    )
    torch.manual_seed(1)
    model = Model(configs, detector_centers=centers)
    folder = tmp_path_factory.mktemp("bundle")
    checkpoint = str(folder / "checkpoint.pt")
    model.write(checkpoint, epoch=0)
    out = folder / "bundle"
    bundle_module.write_bundle(checkpoint, str(out), check=True)  # checks ORT vs torch
    return str(out)


def particles(n, seed=0):
    """Particles like the ones CORSIKA hands over, in the model's frame."""
    rng = np.random.default_rng(seed)
    codes = np.array(PDGS + [321])  # 321 isn't in the vocabulary: index 0
    pdg = rng.choice(codes, n)
    position = np.stack(
        [rng.uniform(-2e4, 2e4, n), rng.uniform(-2e4, 2e4, n), rng.uniform(1400, 1e4, n)], -1
    )
    direction = rng.normal(size=(n, 3)) - np.array([0, 0, 3])
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    momentum = direction * 10 ** rng.uniform(-3, np.log10(900), (n, 1))
    kinematics = np.concatenate([position, momentum], -1)
    seeds = rng.integers(0, 2**63, n, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    return pdg.astype(np.int32), kinematics, seeds


def test_manifest(bundle_dir):
    with open(os.path.join(bundle_dir, "manifest.yaml")) as f:
        manifest = yaml.safe_load(f)
    corsika = manifest["corsika"]
    assert corsika["interface_version"] == 1
    assert corsika["trigger"] == {"em": 900.0, "muon": 900.0, "hadron": 900.0}
    assert corsika["incident_pdgs"] == PDGS
    assert corsika["stations"]["count"] == 300
    with open(os.path.join(bundle_dir, "stations.csv")) as f:
        assert f.readline().strip() == "index,id,x,y,z,radius"
        assert len(f.readlines()) == 300


def test_uniforms():
    # SplitMix64, as in the plugin: known first outputs for seed 0
    u = uniforms(np.array([0], np.uint64), np.arange(3))
    expected = [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    assert np.array_equal(u[0], [(x >> 40) * 2.0**-24 for x in expected])
    # a particle's noise doesn't depend on the others
    seeds = np.array([5, 7, 11], np.uint64)
    assert np.array_equal(uniforms(seeds, np.arange(4))[1], uniforms(seeds[1:2], np.arange(4))[0])


def test_buckets():
    m = np.array([0, 3, 1, 0, 5, 1, 2])
    groups = list(buckets(m, max_tokens=6))
    rows = np.concatenate([g for g, _ in groups])
    assert sorted(rows) == [1, 2, 4, 5, 6]
    assert all(len(g) * k <= 6 or len(g) == 1 for g, k in groups)
    assert [k for _, k in groups] == sorted(k for _, k in groups)


def test_sampler_is_batch_independent(bundle_dir):
    pdg, kinematics, seeds = particles(400)
    whole = BundleSampler(bundle_dir, max_batch=4096, max_tokens=4096).sample(
        pdg, kinematics, seeds
    )
    assert len(whole["station"]) > 0
    # the same particles, sampled in two halves with other chunk sizes
    small = BundleSampler(bundle_dir, max_batch=37, max_tokens=53)
    parts = [small.sample(pdg[s], kinematics[s], seeds[s]) for s in (slice(0, 150), slice(150, None))]
    parts[1]["particle"] = parts[1]["particle"] + 150
    joined = {k: np.concatenate([p[k] for p in parts]) for k in whole}
    order_a = np.lexsort((whole["station"], whole["particle"]))
    order_b = np.lexsort((joined["station"], joined["particle"]))
    for key in ["particle", "station", "pdg"]:
        assert np.mean(whole[key][order_a] != joined[key][order_b]) < 1e-3
    np.testing.assert_allclose(
        whole["kinetic_energy"][order_a], joined["kinetic_energy"][order_b], rtol=1e-4
    )


# ------------------------------------------------------------------ the plugin


class Particle(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint64), ("id", ctypes.c_uint64), ("pdg", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        *[(name, ctypes.c_double) for name in
          ["x", "y", "z", "px", "py", "pz", "kinetic_energy", "time", "weight"]],
    ]


class Hit(ctypes.Structure):
    _fields_ = [
        ("particle", ctypes.c_uint64), ("station", ctypes.c_int32), ("pdg", ctypes.c_int32),
        *[(name, ctypes.c_double) for name in
          ["kinetic_energy", "time", "nx", "ny", "nz", "weight"]],
    ]


SINK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(Hit), ctypes.c_size_t)


class Plugin(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("create", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                                    ctypes.c_char_p, ctypes.c_size_t)),
        ("destroy", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("describe", ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)),
        ("generate", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(Particle),
                                      ctypes.c_size_t, SINK, ctypes.c_void_p,
                                      ctypes.c_char_p, ctypes.c_size_t)),
    ]


@pytest.fixture(scope="module")
def plugin():
    path = os.environ.get("SIMPLESHOWER_C8_PLUGIN")
    if not path or not os.path.exists(path):
        pytest.skip("set SIMPLESHOWER_C8_PLUGIN to the built libsimpleshower_c8.so")
    library = ctypes.CDLL(path, mode=os.RTLD_LOCAL)
    library.c8s_get_plugin.restype = ctypes.POINTER(Plugin)
    api = library.c8s_get_plugin().contents
    assert api.abi_version == 1
    assert api.name == b"simpleshower"
    return library, api


def plugin_hits(api, bundle_dir, pdg, kinematics, seeds, options=b"", batches=1):
    error = ctypes.create_string_buffer(1024)
    model = api.create(bundle_dir.encode(), options, error, 1024)
    assert model, error.value
    try:
        description = yaml.safe_load(api.describe(model).decode())
        assert description["name"] == "simpleshower"
        found = {"particle": [], "station": [], "pdg": [], "kinetic_energy": []}
        offset = [0]

        @SINK
        def sink(_, hits, n):
            for i in range(n):
                h = hits[i]
                found["particle"].append(h.particle + offset[0])
                found["station"].append(h.station)
                found["pdg"].append(h.pdg)
                found["kinetic_energy"].append(h.kinetic_energy)
                assert np.isnan(h.time) and h.weight == 1.0

        for rows in np.array_split(np.arange(len(pdg)), batches):
            batch = (Particle * len(rows))()
            for j, i in enumerate(rows):
                x, y, z, px, py, pz = kinematics[i]
                batch[j] = Particle(int(seeds[i]), int(i), int(pdg[i]), 0, x, y, z, px, py,
                                    pz, 0.0, 0.0, 1.0)
            offset[0] = int(rows[0]) if len(rows) else 0
            status = api.generate(model, batch, len(rows), sink, None, error, 1024)
            assert status == 0, error.value
        return {k: np.asarray(v) for k, v in found.items()}
    finally:
        api.destroy(model)


def sorted_hits(hits):
    order = np.lexsort((hits["kinetic_energy"], hits["station"], hits["particle"]))
    return {k: v[order] for k, v in hits.items()}


def test_plugin_matches_sampler(bundle_dir, plugin):
    _, api = plugin
    pdg, kinematics, seeds = particles(500, seed=3)
    # the plugin rounds the particles to float32 as triggers.parquet stores them
    kinematics32 = kinematics.astype(np.float32)
    python = sorted_hits(
        BundleSampler(bundle_dir, max_batch=2048, max_tokens=2048).sample(pdg, kinematics32, seeds)
    )
    native = sorted_hits(plugin_hits(api, bundle_dir, pdg, kinematics32.astype(np.float64), seeds))
    assert len(native["station"]) == len(python["station"]) > 0
    for key in ["particle", "station", "pdg"]:
        assert np.array_equal(native[key], python[key]), key
    np.testing.assert_allclose(native["kinetic_energy"], python["kinetic_energy"], rtol=1e-6)


def test_plugin_is_batch_independent(bundle_dir, plugin):
    _, api = plugin
    pdg, kinematics, seeds = particles(500, seed=4)
    one = sorted_hits(plugin_hits(api, bundle_dir, pdg, kinematics, seeds))
    many = sorted_hits(
        plugin_hits(api, bundle_dir, pdg, kinematics, seeds,
                    options=b"threads=2,max_batch=33,max_tokens=40", batches=7)
    )
    assert len(one["station"]) == len(many["station"])
    for key in ["particle", "station", "pdg"]:
        assert np.mean(one[key] != many[key]) < 1e-3, key


def test_plugin_rejects_bad_options(bundle_dir, plugin):
    _, api = plugin
    error = ctypes.create_string_buffer(1024)
    assert not api.create(bundle_dir.encode(), b"colour=blue", error, 1024)
    assert b"unknown option" in error.value
    assert not api.create(b"/no/such/bundle", b"", error, 1024)
