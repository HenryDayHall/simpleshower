"""
Training examples from subshowers' sampler.

Each example is one subshower: the particle that starts it (the incident
particle), and its deposits, one for each detector each of its leaves hits,
with the detector, the leaf's PDG and its kinetic energy. Summed over the
deposits in each detector they give the sampler's energy_per_detector.

Unless told not to, the sampler moves each subshower across the ground, by a
random shift in x and y, before the array sees it (see subshowers'
SubshowerSampler), so that subshowers simulated around their showers' cores
reach detectors all over the array. An example is of the shifted subshower:
its incident particle starts where the shifted one does, its deposits are
those the shifted leaves make, and its shift says how far it was moved.

Working out which leaves hit which detectors takes a pass over every
subshower, so each subshower set's examples are cached, in a folder that
runs share, and used again by any run with the same set, detector array and
sampling options.
"""

import contextlib
import hashlib
import json
import math
import os
import time
import uuid
import warnings
import zipfile
import zlib

import numpy as np

# what's kept about each incident particle
INCIDENT_COLUMNS = ["pdg", "x", "y", "z", "px", "py", "pz", "kinetic_energy", "time"]
# what the model reads of it
MODEL_COLUMNS = ["pdg", "x", "y", "z", "px", "py", "pz"]
FORMAT_VERSION = 2
# load reads these; version 1 files are from before examples kept their
# shifts, when subshowers weren't shifted
READABLE_VERSIONS = (1, 2)

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

    shift is how far the sampler moved each subshower across the ground,
    (dx, dy) in m. The incident values and deposits are the shifted
    subshower's, so subtracting the shift from the incident x and y gives
    where the history has the start particle (start_index).
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
        shift=None,
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
        self.shift = (
            np.zeros((n, 2))
            if shift is None
            else np.asarray(shift, dtype=np.float64).reshape(n, 2)
        )
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
            shift=self.shift[indices],
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
            shift=np.concatenate([part.shift for part in parts]),
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
            "shift": self.shift,
        }
        for name, column in self.incident.items():
            arrays[f"incident_{name}"] = column
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # a temporary file of its own, so that runs saving the same file at
        # once, as runs sharing a cache can, don't write into each other's
        temporary = f"{path}.{uuid.uuid4().hex[:12]}.tmp"
        try:
            with open(temporary, "xb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(temporary, path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(temporary)  # only there if saving failed

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version not in READABLE_VERSIONS:
                raise ValueError(
                    f"{path} has format version {version}; versions"
                    f" {', '.join(map(str, READABLE_VERSIONS))} can be read"
                )
            return cls(
                {name: data[f"incident_{name}"] for name in INCIDENT_COLUMNS},
                data["deposit_offsets"],
                data["deposit_detector"],
                data["deposit_pdg"],
                data["deposit_energy"],
                data["source"],
                data["start_index"],
                json.loads(data["metadata"].item()),
                # version 1 kept no shifts, as its subshowers weren't shifted
                shift=data["shift"] if "shift" in data.files else None,
            )

    @staticmethod
    def read_metadata(path):
        with np.load(path, allow_pickle=False) as data:
            return json.loads(data["metadata"].item())


# --------------------------------------------------- from the sampler


def shift_of(sample):
    """
    How far the sampler moved a sampling.DetectedSample across the ground,
    (dx, dy) in m; nothing for a sampler that doesn't shift subshowers.
    """
    shift = getattr(sample, "shift", None)
    if shift is None:
        return np.zeros(2)
    return np.asarray(shift, dtype=np.float64).reshape(-1)[:2]


def deposits_of(sample, detector_array):
    """
    (detector, pdg, energy) of each deposit of a sampling.DetectedSample,
    highest energy first: one for each detector each detected leaf hits.
    The leaves are moved across the ground by the sample's shift, as the
    sampler moved them to see which detectors they hit.
    """
    from subshowers.sampling import history_columns

    leaves = np.asarray(sample.detected_leaf_indices)
    if len(leaves) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    columns, positions_of = history_columns(sample.source.reader)
    positions = positions_of(leaves)
    starts = np.stack([columns[c][positions] for c in ["x", "y", "z"]], axis=-1)
    dx, dy = shift_of(sample)
    if dx or dy:
        # a new array, so the history itself stays where it was
        starts = starts + np.array([dx, dy, 0.0])
    directions = np.stack([columns[c][positions] for c in ["px", "py", "pz"]], axis=-1)
    leaf, detector = np.nonzero(detector_array.hit_any(starts, directions))
    energy = columns["kinetic_energy"][positions][leaf]
    pdg = columns["pdg"][positions][leaf]
    order = np.argsort(-energy, kind="stable")
    return detector[order], pdg[order], energy[order]


def examples_from_samples(samples, detector_array, source_name=None, check=True):
    """
    Examples of some sampling.DetectedSamples, of the subshowers where the
    sampler had them: shifted, if it shifted them.
    """
    incident = {name: [] for name in INCIDENT_COLUMNS}
    offsets, detectors, pdgs, energies = [0], [], [], []
    sources, starts, shifts = [], [], []
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
        # where the shifted subshower starts
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
        shifts.append(shift_of(sample))

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
        shift=np.reshape(np.asarray(shifts, dtype=np.float64), (-1, 2)),
    )


def shower_folders(config):
    """
    The showers folders, in storage_path, that the config's showers section
    lists as folders; or its one folder, if it gives showers: folder as
    subshowers' configs do.
    """
    showers = config["showers"]
    if "folder" in showers:
        if "folders" in showers:
            raise DataError("the showers section has both folder and folders; give just folders")
        folders = [showers["folder"]]
    else:
        folders = showers["folders"]
    if not (
        isinstance(folders, list)
        and folders
        and all(isinstance(folder, str) and folder for folder in folders)
    ):
        raise DataError("showers: folders must be a list of folder names, e.g. [PDG22, PDG11]")
    return list(dict.fromkeys(folders))


def folder_configs(config):
    """
    {showers folder: the config with just that one as showers: folder}, for
    each folder the config lists, as subshowers reads one folder at a time.
    The subfolders, and the subshowers section, go for every folder.
    """
    showers = {key: value for key, value in config["showers"].items() if key != "folders"}
    return {
        folder: {**config, "showers": {**showers, "folder": folder}}
        for folder in shower_folders(config)
    }


def shift_distance(array, shift=True, max_shift=None):
    """
    The furthest (m) subshowers are moved across the ground before the
    array sees them: max_shift, or by default half the width of the array
    (subshowers' detector.max_shift); 0 without shift.
    """
    from subshowers import detector

    if not isinstance(shift, bool):
        raise DataError(f"training: shift must be true or false, not {shift!r}")
    if not shift:
        return 0.0
    if max_shift is None:
        return float(detector.max_shift(array))
    try:
        distance = math.nan if isinstance(max_shift, bool) else float(max_shift)
    except (TypeError, ValueError):
        distance = math.nan
    if not (math.isfinite(distance) and distance >= 0):
        raise DataError(
            "training: max_shift must be a distance in m, at least 0, or null for"
            f" half the width of the array, not {max_shift!r}"
        )
    return distance


# ------------------------------------------------------------ the cache

# what reading a file that isn't examples, or is broken, can raise
UNREADABLE = (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile)


def settings_digest(settings):
    """A short hash of the settings a set's examples are made with."""
    text = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def cache_name(folder, subshower_set, settings):
    """
    The file a set's examples are cached in: the set, with its showers folder
    however many a run trains on, then the hash of the settings, so examples
    made with other settings are kept beside these, as
    PDG22_e1e5_0_e100.3f9a1c2b.npz.
    """
    name = f"{folder}/{subshower_set.shower.subfolder}_e{subshower_set.energy_cut}"
    return f"{name.replace(os.sep, '_')}.{settings_digest(settings)}.npz"


def made_with(path, settings):
    """True if path holds examples made with these settings."""
    try:
        metadata = Examples.read_metadata(path)
    except UNREADABLE:
        return False
    return isinstance(metadata, dict) and metadata.get("settings") == json.loads(
        json.dumps(settings)
    )


def find_cached(cache_folder, name, settings):
    """
    The file in cache_folder of examples made with these settings, or None:
    the one named for them, or else any other there, whatever its name, so
    that files put there by hand, such as those of a run that kept its own
    cache, are found too.
    """
    expected = os.path.join(cache_folder, name)
    if made_with(expected, settings):
        return expected
    try:
        with os.scandir(cache_folder) as entries:
            others = sorted(
                entry.path
                for entry in entries
                if entry.name.endswith(".npz") and entry.path != expected and entry.is_file()
            )
    except OSError:  # no cache yet
        return None
    return next((path for path in others if made_with(path, settings)), None)


def examples_from_config(
    config, cache_folder, max_per_set=None, seed=0, shift=True, max_shift=None, log=print
):
    """
    Examples of every subshower set the config's subshowers section asks
    for, in each of its showers folders, that has been made, as its
    detectors section's array sees them, without repeats. Sets that aren't
    made are listed and left out.

    With shift, the sampler moves each subshower across the ground, by a
    random shift in x and y of up to max_shift m (by default half the width
    of the array), before the array sees it, so the examples cover every
    detector rather than just those near the showers' cores. Each set's
    shifts, and the subshowers max_per_set takes at random from it (all if
    None), come from seed and the set, so they differ from set to set and
    are the same every time.

    Each set's examples are cached in cache_folder, which runs can share, and
    read from there by any run with the same set, array and options. Files
    are named by cache_name, so examples made with other settings are kept
    beside them, and any file there made with these settings is used,
    whatever its name. However they were cached, a set's subshowers are
    named as this config names them.
    """
    from subshowers import datasets, detector, generation
    from subshowers.sampling import SubshowerSampler

    if not hasattr(detector, "max_shift"):
        raise DataError(
            "this subshowers is too old to shift subshowers across the ground;"
            " update it to one whose SubshowerSampler takes shift and max_shift"
        )
    if detector.is_unseeded(config):
        raise DataError(
            "the detector array is random and has no seed, so it would be different"
            " each time; give detectors: generator a seed"
        )
    array = detector.from_config(config)
    distance = shift_distance(array, shift, max_shift)
    folders = folder_configs(config)
    # the same subfolder, e.g. e1e5_0, can be in more than one showers folder,
    # so with several, sets are named with their folder in front, and so are
    # their subshowers' sources, which deduplicated() goes by; with one, by
    # the subfolder alone, as subshowers names them
    several = len(folders) > 1

    def named(folder, name):
        return f"{folder}/{name}" if several else name

    checks = [
        (folder, check)
        for folder, folder_config in folders.items()
        for check in datasets.check_subshower_sets(folder_config)
    ]
    ready = [(folder, check) for folder, check in checks if check.ready]
    missing = [(folder, check) for folder, check in checks if not check.ready]
    log(f"Subshower sets: {len(ready)} of {len(checks)} ready")
    for folder, check in missing:
        log(f"  missing {named(folder, check.label)}: {check.problem}")
    if not ready:
        raise DataError("none of the config's subshower sets are made")
    if distance > 0:
        log(f"Subshowers are shifted across the ground by up to {distance:.6g} m")
    else:
        log("Subshowers aren't shifted: they're where they were simulated")

    parts = []
    for folder, check in ready:
        subshower_set = check.subshower_set
        source_name = named(folder, subshower_set.shower.subfolder)
        # the history is only ever read, so its size and time say if it's remade
        history = os.stat(generation.history_path(subshower_set.shower.folder))
        settings = {
            "set": os.path.abspath(subshower_set.output),
            "set_settings": list(generation.saved_settings(subshower_set.output)),
            "history": [history.st_size, int(history.st_mtime)],
            "detectors": config["detectors"],
            "max_per_set": max_per_set,
            "seed": seed,
            # how far subshowers are moved across the ground (m); 0 for not at all
            "max_shift": distance,
        }
        # the set's own random numbers, for its shifts and the subshowers
        # max_per_set takes, so no two sets have the same ones; the set is
        # named with its folder, so they're the same whatever else is trained on
        set_key = f"{folder}/{subshower_set.shower.subfolder}/e{subshower_set.energy_cut}"
        set_rng = np.random.default_rng(
            None if seed is None else [seed, zlib.crc32(set_key.encode())]
        )
        name = cache_name(folder, subshower_set, settings)
        examples = None
        cached = find_cached(cache_folder, name, settings)
        if cached is not None:
            try:
                examples = Examples.load(cached)
            except UNREADABLE:
                examples = None
            found_as = os.path.basename(cached)
            how = "cached" if found_as == name else f"cached in {found_as}"
        if examples is None:
            started = time.monotonic()
            source = check.source()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sampler = SubshowerSampler(
                    [source],
                    array,
                    seed=set_rng,
                    max_open=1,
                    shift=distance > 0,
                    max_shift=distance if distance > 0 else None,
                )
                if max_per_set is None or max_per_set >= len(sampler):
                    samples = sampler.iter_epoch(shuffle=False)
                else:
                    chosen = np.sort(
                        sampler.rng.choice(len(sampler), max_per_set, replace=False)
                    )
                    samples = (sampler[int(i)] for i in chosen)
                examples = examples_from_samples(samples, array, source_name=source_name)
            source.release()
            examples.metadata = {"settings": settings}
            examples.save(os.path.join(cache_folder, name))
            how = f"sampled in {time.monotonic() - started:.0f} s"
        # whichever run cached them: with one showers folder or several, the
        # sources are named differently (deduplicated() goes by them)
        examples.source = np.full(len(examples), source_name)
        log(
            f"  {named(folder, check.label)}: {len(examples)} subshowers,"
            f" {examples.detected.sum()} with deposits ({how})"
        )
        parts.append(examples)
    return Examples.concatenate(parts).deduplicated()
