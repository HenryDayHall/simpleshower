"""
Sampling subshowers from a bundle (see bundle.py), as CORSIKA's plugin does, in Python.

Every particle CORSIKA hands over has a 64-bit seed, and all of its randomness comes
from it, as uniforms from a SplitMix64 stream:

    u_k = (mix(seed + (k + 1) * 0x9E3779B97F4A7C15) >> 40) * 2**-24,   k = 0, 1, ...

with mix SplitMix64's finaliser. u_0 and u_1 draw the multiplicity; deposit j (by rank)
uses u_{2 + 4j} to u_{5 + 4j} for its detector, its PDG and its log energy. A particle's
draws don't depend on the batch it is in, so CORSIKA can batch however it likes, and a
run sampled inline by the plugin can be sampled again here, from triggers.parquet, and
give the same hits. 24-bit uniforms are exact in float32, so both sides feed the graphs
the same numbers.

    sampler = BundleSampler("bundle/")
    hits = sampler.sample(pdg, kinematics, seed)  # dict of arrays, one entry per hit
"""

import os

import numpy as np
import yaml

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def uniforms(seed, k):
    """
    The uniforms u_k of each seed: seed (N,) uint64, k (...) ints -> (N, ...) float32,
    exactly as the plugin makes them.
    """
    seed = np.asarray(seed, dtype=np.uint64).reshape(-1, *([1] * np.ndim(k)))
    k = np.asarray(k, dtype=np.uint64)
    with np.errstate(over="ignore"):
        bits = _mix(seed + (k + np.uint64(1)) * GOLDEN)
    return ((bits >> np.uint64(40)).astype(np.float64) * 2.0**-24).astype(np.float32)


def deposit_noise(seed, n_tokens, per_deposit=4, first=2):
    """(N, n_tokens, per_deposit) uniforms of the deposits of rank 0 .. n_tokens - 1."""
    k = first + per_deposit * np.arange(n_tokens)[:, None] + np.arange(per_deposit)[None, :]
    return uniforms(seed, k)


def buckets(multiplicity, max_tokens):
    """
    Groups of rows with deposits, sorted by multiplicity, each padded to its largest
    multiplicity and holding at most max_tokens tokens (or one row, if that is more).
    Sorting keeps the padding small. The plugin groups the same way.
    """
    rows = np.flatnonzero(multiplicity > 0)
    rows = rows[np.argsort(multiplicity[rows], kind="stable")]
    group, largest = [], 0
    for row in rows:
        m = int(multiplicity[row])
        if group and max(largest, m) * (len(group) + 1) > max_tokens:
            yield np.asarray(group), largest
            group, largest = [], 0
        group.append(row)
        largest = max(largest, m)
    if group:
        yield np.asarray(group), largest


class BundleSampler:
    """The two graphs of a bundle, run with ONNX Runtime, and the steps between them."""

    def __init__(self, folder, max_batch=2048, max_tokens=2048, threads=None, providers=None):
        import onnxruntime as ort

        with open(os.path.join(folder, "manifest.yaml")) as f:
            self.manifest = yaml.safe_load(f)
        plugin = self.manifest["plugin"]
        self.pdgs = np.asarray(plugin["pdgs"], dtype=np.int64)  # index i + 1 is pdgs[i]
        self.n_noise_multiplicity = plugin["noise"]["multiplicity"]
        self.per_deposit = plugin["noise"]["per_deposit"]
        self.max_batch, self.max_tokens = max_batch, max_tokens
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.sessions = {
            name: ort.InferenceSession(
                os.path.join(folder, path), options, providers=providers
            )
            for name, path in plugin["graphs"].items()
        }

    def pdg_index(self, pdg):
        """Vocabulary index of each PDG code; 0 for codes the model doesn't know."""
        pdg = np.asarray(pdg, dtype=np.int64)
        order = np.argsort(self.pdgs)
        position = np.searchsorted(self.pdgs[order], pdg).clip(0, len(self.pdgs) - 1)
        found = self.pdgs[order][position] == pdg
        return np.where(found, order[position] + 1, 0).astype(np.int64)

    def pdg_code(self, index):
        return np.concatenate([[0], self.pdgs])[index]

    def multiplicity(self, pdg_index, kinematics, seed):
        """Stage 1, in batches of max_batch: (N,) multiplicities, (N,) P(none)."""
        noise = uniforms(seed, np.arange(self.n_noise_multiplicity))
        out_m, out_p = [], []
        for start in range(0, len(seed), self.max_batch):
            rows = slice(start, start + self.max_batch)
            m, p = self.sessions["multiplicity"].run(
                None,
                {"pdg_index": pdg_index[rows], "kinematics": kinematics[rows],
                 "noise": noise[rows]},
            )
            out_m.append(m)
            out_p.append(p)
        if not out_m:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        return np.concatenate(out_m), np.concatenate(out_p)

    def sample(self, pdg, kinematics, seed):
        """
        The hits of each particle: pdg (N,), kinematics (N, 6) float32 x, y, z [m],
        px, py, pz [GeV/c] in the model's frame, seed (N,) uint64. Returns a dict of
        arrays, one entry per hit: particle (row), station, pdg, kinetic_energy [GeV].
        """
        kinematics = np.ascontiguousarray(kinematics, dtype=np.float32)
        seed = np.asarray(seed, dtype=np.uint64)
        pdg_index = self.pdg_index(pdg)
        multiplicity, _ = self.multiplicity(pdg_index, kinematics, seed)
        hits = {"particle": [], "station": [], "pdg": [], "kinetic_energy": []}
        for rows, n_tokens in buckets(multiplicity, self.max_tokens):
            detector, deposit_pdg, log_energy = self.sessions["deposits"].run(
                None,
                {
                    "pdg_index": pdg_index[rows],
                    "kinematics": kinematics[rows],
                    "multiplicity": multiplicity[rows],
                    "noise": deposit_noise(seed[rows], n_tokens, self.per_deposit,
                                           self.n_noise_multiplicity),
                },
            )
            keep = np.arange(n_tokens)[None, :] < multiplicity[rows][:, None]
            hits["particle"].append(np.broadcast_to(rows[:, None], keep.shape)[keep])
            hits["station"].append(detector[keep])
            hits["pdg"].append(self.pdg_code(deposit_pdg[keep]))
            hits["kinetic_energy"].append(10.0 ** log_energy[keep].astype(np.float64))
        dtypes = {"particle": np.int64, "station": np.int32, "pdg": np.int32,
                  "kinetic_energy": np.float64}
        return {
            name: (np.concatenate(parts) if parts else np.zeros(0)).astype(dtypes[name])
            for name, parts in hits.items()
        }
