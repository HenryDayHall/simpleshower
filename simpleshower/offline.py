"""
Run a bundle on the particles a CORSIKA run recorded, after the run:

    python -m simpleshower.offline BUNDLE LIBRARY [--output FILE] [--batch N] [--device cuda]

LIBRARY is a CORSIKA output directory from c8_air_shower_surrogate without
--surrogate-model; its surrogate/triggers.parquet lists the particles taken out of the
cascade, each with its seed. This writes their hits to surrogate/hits.parquet (or
--output), in the same table the plugin writes during a run, so the readout can't tell
the two apart: with the same bundle, the same seeds give the same hits.

The same file can be sampled again with another bundle, or more than once with
--resample N, which gives each particle N independent subshowers (seeds mixed with the
sample number), each with weight 1/N.
"""

import argparse
import os
import sys

import numpy as np

from simpleshower.sampling import BundleSampler, _mix

HIT_COLUMNS = [
    "shower", "trigger", "station", "pdg", "kinetic_energy", "time", "nx", "ny", "nz",
    "weight",
]


def hit_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("shower", pa.uint32()), ("trigger", pa.int64()), ("station", pa.int32()),
            ("pdg", pa.int32()), ("kinetic_energy", pa.float32()), ("time", pa.float64()),
            ("nx", pa.float32()), ("ny", pa.float32()), ("nz", pa.float32()),
            ("weight", pa.float32()),
        ]
    )


def resampled_seeds(seed, sample):
    """Seeds for the sample-th resampling; sample 0 is the run's own."""
    if sample == 0:
        return seed
    with np.errstate(over="ignore"):
        return _mix(seed ^ _mix(np.uint64(sample) * np.uint64(0xD1B54A32D192ED03)))


def run(bundle, library, output=None, batch=1 << 16, resample=1, **sampler_options):
    import pyarrow as pa
    import pyarrow.parquet as pq

    sampler = BundleSampler(bundle, **sampler_options)
    triggers = pq.ParquetFile(os.path.join(library, "surrogate", "triggers.parquet"))
    output = output or os.path.join(library, "surrogate", "hits.parquet")
    if os.path.exists(output):
        raise FileExistsError(f"{output} exists; give another --output")
    n_hits = 0
    with pq.ParquetWriter(output, hit_schema(), compression="lz4") as writer:
        for table in triggers.iter_batches(batch_size=batch):
            t = {name: table.column(name).to_numpy() for name in table.schema.names}
            kinematics = np.stack(
                [t[c] for c in ["x", "y", "z", "px", "py", "pz"]], -1
            ).astype(np.float32)
            for sample in range(resample):
                seed = resampled_seeds(t["seed"].astype(np.uint64), sample)
                hits = sampler.sample(t["pdg"], kinematics, seed)
                rows = hits["particle"]
                n = len(rows)
                nan = np.full(n, np.nan, np.float32)
                columns = {
                    "shower": t["shower"][rows].astype(np.uint32),
                    "trigger": t["id"][rows].astype(np.int64),
                    "station": hits["station"],
                    "pdg": hits["pdg"],
                    "kinetic_energy": hits["kinetic_energy"].astype(np.float32),
                    "time": np.full(n, np.nan),  # not predicted by this model
                    "nx": nan, "ny": nan, "nz": nan,
                    "weight": (t["weight"][rows] / resample).astype(np.float32),
                }
                writer.write_table(pa.table(columns, schema=hit_schema()))
                n_hits += n
    return output, n_hits


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a bundle on a CORSIKA run's triggers.")
    parser.add_argument("bundle")
    parser.add_argument("library", help="the CORSIKA output directory")
    parser.add_argument("--output", help="default: LIBRARY/surrogate/hits.parquet")
    parser.add_argument("--batch", type=int, default=1 << 16, help="particles read at once")
    parser.add_argument("--resample", type=int, default=1, help="subshowers per particle")
    parser.add_argument("--max-batch", type=int, default=2048, help="rows per stage-1 call")
    parser.add_argument("--max-tokens", type=int, default=2048, help="tokens per stage-2 call")
    parser.add_argument("--threads", type=int, help="ONNX Runtime threads")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    providers = ["CUDAExecutionProvider"] if args.device == "cuda" else None
    output, n_hits = run(
        args.bundle, args.library, args.output, args.batch, args.resample,
        max_batch=args.max_batch, max_tokens=args.max_tokens, threads=args.threads,
        providers=providers,
    )
    print(f"Wrote {n_hits} hits to {output}")


if __name__ == "__main__":
    sys.exit(main())
