"""
Training examples from subshowers' sampler.

Each example is one subshower: the particle that starts it (the incident
particle), and its deposits, one for each detector each of its leaves hits,
with the detector, the leaf's PDG and its kinetic energy. Summed over the
deposits in each detector they give the sampler's energy_per_detector.

Working out which leaves hit which detectors takes a pass over every
subshower, so each subshower set's examples are cached, and used again while
the set, the detector array and the sampling options stay the same.
"""

import json
import os
import time
import warnings

import numpy as np

# what's kept about each incident particle
INCIDENT_COLUMNS = ["pdg", "x", "y", "z", "px", "py", "pz", "kinetic_energy", "time"]
# what the model reads of it
MODEL_COLUMNS = ["pdg", "x", "y", "z", "px", "py", "pz"]
FORMAT_VERSION = 1

# GeV; for working out kinetic energy from momentum
MASSES = {
    22: 0.0,
    11: 0.000510999,
    12: 0.0,
    13: 0.105658,
    14: 0.0,
    15: 1.77686,
    16: 0.0,
    111: 0.134977,
    211: 0.139570,
    130: 0.497611,
    310: 0.497611,
    311: 0.497611,
    321: 0.493677,
    221: 0.547862,
    2212: 0.938272,
    2112: 0.939565,
    3122: 1.115683,
    3222: 1.18937,
    3212: 1.192642,
    3112: 1.197449,
    3322: 1.31486,
    3312: 1.32171,
    3334: 1.67245,
}
NUCLEON_MASS = 0.931494  # GeV, per nucleon of a nucleus


class DataError(ValueError):
    """The config's subshowers can't be made into training examples."""


def mass(pdg):
    """Mass in GeV of a PDG code, or None if it isn't known here."""
    pdg = abs(int(pdg))
    if pdg in MASSES:
        return MASSES[pdg]
    if pdg >= 1000000000:  # nucleus, 10LZZZAAAI
        return (pdg // 10) % 1000 * NUCLEON_MASS
    return None


def kinetic_energy(pdg, momentum):
    """Kinetic energy (GeV) of a particle from its PDG and momentum (GeV/c)."""
    size = float(np.linalg.norm(momentum))
    m = mass(pdg)
    if m is None:
        warnings.warn(f"no mass known for PDG {pdg}; taking it to be massless")
        m = 0.0
    return float(np.sqrt(size**2 + m**2) - m)


class Examples:
    """
    Many subshowers: one row of incident values each, and their deposits as
    flat arrays, subshower i's being deposit_offsets[i]:deposit_offsets[i + 1],
    in order of energy, highest first.
    """

    def __init__(
        self,
        incident,
        deposit_offsets,
        deposit_detector,
        deposit_pdg,
        deposit_energy,
        source,
        start_index,
        metadata=None,
    ):
        n = len(deposit_offsets) - 1
        self.incident = {
            name: np.asarray(incident[name], dtype=np.int64 if name == "pdg" else np.float64).reshape(n)
            for name in INCIDENT_COLUMNS
        }
        self.deposit_offsets = np.asarray(deposit_offsets, dtype=np.int64)
        self.deposit_detector = np.asarray(deposit_detector, dtype=np.int64)
        self.deposit_pdg = np.asarray(deposit_pdg, dtype=np.int64)
        self.deposit_energy = np.asarray(deposit_energy, dtype=np.float64)
        self.source = np.asarray(source, dtype=str).reshape(n)
        self.start_index = np.asarray(start_index, dtype=np.int64).reshape(n)
        self.metadata = dict(metadata or {})
        if self.deposit_offsets[0] != 0 or self.deposit_offsets[-1] != len(
            self.deposit_detector
        ):
            raise ValueError("deposit_offsets must run from 0 to the number of deposits")

    def __len__(self):
        return len(self.deposit_offsets) - 1

    @property
    def multiplicity(self):
        """Number of deposits in each subshower."""
        return np.diff(self.deposit_offsets)

    @property
    def detected(self):
        return self.multiplicity > 0

    def deposits(self, i):
        """(detector, pdg, energy) of subshower i's deposits."""
        start, end = self.deposit_offsets[i], self.deposit_offsets[i + 1]
        return (
            self.deposit_detector[start:end],
            self.deposit_pdg[start:end],
            self.deposit_energy[start:end],
        )

    def select(self, which):
        """The subshowers a boolean mask or array of indices picks out."""
        indices = np.arange(len(self))[which]
        counts = self.multiplicity[indices]
        offsets = np.concatenate(([0], np.cumsum(counts)))
        positions = np.repeat(self.deposit_offsets[indices] - offsets[:-1], counts)
        positions += np.arange(offsets[-1])
        return Examples(
            {name: column[indices] for name, column in self.incident.items()},
            offsets,
            self.deposit_detector[positions],
            self.deposit_pdg[positions],
            self.deposit_energy[positions],
            self.source[indices],
            self.start_index[indices],
            self.metadata,
        )

    @classmethod
    def concatenate(cls, parts, metadata=None):
        parts = list(parts)
        if not parts:
            raise ValueError("nothing to concatenate")
        counts = np.concatenate([part.multiplicity for part in parts])
        return cls(
            {
                name: np.concatenate([part.incident[name] for part in parts])
                for name in INCIDENT_COLUMNS
            },
            np.concatenate(([0], np.cumsum(counts))),
            np.concatenate([part.deposit_detector for part in parts]),
            np.concatenate([part.deposit_pdg for part in parts]),
            np.concatenate([part.deposit_energy for part in parts]),
            np.concatenate([part.source for part in parts]),
            np.concatenate([part.start_index for part in parts]),
            metadata,
        )

    def deduplicated(self):
        """
        Without repeats of a subshower: a particle can start a subshower for
        more than one energy cut, and then it's the same subshower in each.
        """
        keys = np.char.add(np.char.add(self.source, ":"), self.start_index.astype(str))
        _, first = np.unique(keys, return_index=True)
        return self.select(np.sort(first))

    def split(self, fraction, seed=0):
        """(the rest, a random fraction of the subshowers)"""
        order = np.random.default_rng(seed).permutation(len(self))
        n_held_out = int(round(fraction * len(self)))
        if fraction > 0 and len(self) > 1:
            n_held_out = min(max(n_held_out, 1), len(self) - 1)
        return self.select(np.sort(order[n_held_out:])), self.select(
            np.sort(order[:n_held_out])
        )

    def pdg_codes(self):
        """Every PDG code among the incident particles and deposits."""
        return sorted(
            set(np.unique(self.incident["pdg"]).tolist())
            | set(np.unique(self.deposit_pdg).tolist())
        )

    # ------------------------------------------------------- saving, loading

    def save(self, path):
        arrays = {
            "format_version": np.array(FORMAT_VERSION),
            "metadata": np.array(json.dumps(self.metadata)),
            "deposit_offsets": self.deposit_offsets,
            "deposit_detector": self.deposit_detector,
            "deposit_pdg": self.deposit_pdg,
            "deposit_energy": self.deposit_energy,
            "source": self.source,
            "start_index": self.start_index,
        }
        for name, column in self.incident.items():
            arrays[f"incident_{name}"] = column
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"{path} has format version {version}, not {FORMAT_VERSION}")
            return cls(
                {name: data[f"incident_{name}"] for name in INCIDENT_COLUMNS},
                data["deposit_offsets"],
                data["deposit_detector"],
                data["deposit_pdg"],
                data["deposit_energy"],
                data["source"],
                data["start_index"],
                json.loads(data["metadata"].item()),
            )

    @staticmethod
    def read_metadata(path):
        with np.load(path, allow_pickle=False) as data:
            return json.loads(data["metadata"].item())


# --------------------------------------------------- from the sampler


def deposits_of(sample, detector_array):
    """
    (detector, pdg, energy) of each deposit of a sampling.DetectedSample,
    highest energy first: one for each detector each detected leaf hits.
    """
    from subshowers.sampling import history_columns

    leaves = np.asarray(sample.detected_leaf_indices)
    if len(leaves) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    columns, positions_of = history_columns(sample.source.reader)
    positions = positions_of(leaves)
    starts = np.stack([columns[c][positions] for c in ["x", "y", "z"]], axis=-1)
    directions = np.stack([columns[c][positions] for c in ["px", "py", "pz"]], axis=-1)
    leaf, detector = np.nonzero(detector_array.hit_any(starts, directions))
    energy = columns["kinetic_energy"][positions][leaf]
    pdg = columns["pdg"][positions][leaf]
    order = np.argsort(-energy, kind="stable")
    return detector[order], pdg[order], energy[order]


def examples_from_samples(samples, detector_array, source_name=None, check=True):
    """Examples of some sampling.DetectedSamples."""
    incident = {name: [] for name in INCIDENT_COLUMNS}
    offsets, detectors, pdgs, energies = [0], [], [], []
    sources, starts = [], []
    n_detectors = len(detector_array)
    for sample in samples:
        detector, pdg, energy = deposits_of(sample, detector_array)
        if check and not np.allclose(
            np.bincount(detector, weights=energy, minlength=n_detectors),
            sample.energy_per_detector,
        ):
            raise RuntimeError(
                f"deposits of subshower {sample.subshower_number} of"
                f" {sample.source.name} don't add up to its energy_per_detector"
            )
        incident["pdg"].append(int(sample.start_pdg))
        for axis, name in enumerate(["x", "y", "z"]):
            incident[name].append(sample.start_position[axis])
        # the sampler's start_direction is the start particle's px, py, pz
        for axis, name in enumerate(["px", "py", "pz"]):
            incident[name].append(sample.start_direction[axis])
        incident["kinetic_energy"].append(sample.start_kinetic_energy)
        incident["time"].append(sample.start_time)
        detectors.append(detector)
        pdgs.append(pdg)
        energies.append(energy)
        offsets.append(offsets[-1] + len(detector))
        sources.append(source_name or sample.source.name)
        starts.append(sample.start_index)

    def joined(parts, dtype):
        return np.concatenate(parts) if parts else np.zeros(0, dtype)

    return Examples(
        incident,
        offsets,
        joined(detectors, np.int64),
        joined(pdgs, np.int64),
        joined(energies, np.float64),
        sources,
        starts,
    )


def examples_from_config(config, cache_folder, max_per_set=None, seed=0, log=print):
    """
    Examples of every subshower set the config's subshowers section asks
    for that has been made, as its detectors section's array sees them,
    without repeats. Sets that aren't made are listed and left out.

    Each set's examples are cached in cache_folder, and read from there
    while the set, the array and max_per_set (subshowers taken at random
    from each set; all if None) stay the same.
    """
    from subshowers import datasets, detector, generation
    from subshowers.sampling import SubshowerSampler

    if detector.is_unseeded(config):
        raise DataError(
            "the detector array is random and has no seed, so it would be different"
            " each time; give detectors: generator a seed"
        )
    array = detector.from_config(config)
    checks = datasets.check_subshower_sets(config)
    ready = [check for check in checks if check.ready]
    missing = [check for check in checks if not check.ready]
    log(f"Subshower sets: {len(ready)} of {len(checks)} ready")
    for check in missing:
        log(f"  missing {check.label}: {check.problem}")
    if not ready:
        raise DataError("none of the config's subshower sets are made")

    parts = []
    for check in ready:
        subshower_set = check.subshower_set
        # the history is only ever read, so its size and time say if it's remade
        history = os.stat(generation.history_path(subshower_set.shower.folder))
        settings = {
            "set": os.path.abspath(subshower_set.output),
            "set_settings": list(generation.saved_settings(subshower_set.output)),
            "history": [history.st_size, int(history.st_mtime)],
            "detectors": config["detectors"],
            "max_per_set": max_per_set,
            "seed": seed,
        }
        name = f"{subshower_set.shower.subfolder}_e{subshower_set.energy_cut}".replace(
            os.sep, "_"
        )
        path = os.path.join(cache_folder, f"{name}.npz")
        examples = None
        if os.path.isfile(path):
            try:
                if Examples.read_metadata(path).get("settings") == json.loads(
                    json.dumps(settings)
                ):
                    examples = Examples.load(path)
                    how = "cached"
            except (OSError, ValueError, KeyError):
                examples = None
        if examples is None:
            started = time.monotonic()
            source = check.source()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sampler = SubshowerSampler([source], array, seed=seed, max_open=1)
                if max_per_set is None or max_per_set >= len(sampler):
                    samples = sampler.iter_epoch(shuffle=False)
                else:
                    chosen = np.sort(
                        sampler.rng.choice(len(sampler), max_per_set, replace=False)
                    )
                    samples = (sampler[int(i)] for i in chosen)
                examples = examples_from_samples(
                    samples, array, source_name=subshower_set.shower.subfolder
                )
            source.release()
            examples.metadata = {"settings": settings}
            examples.save(path)
            how = f"sampled in {time.monotonic() - started:.0f} s"
        log(
            f"  {check.label}: {len(examples)} subshowers,"
            f" {examples.detected.sum()} with deposits ({how})"
        )
        parts.append(examples)
    return Examples.concatenate(parts).deduplicated()
