# SimpleShower

A minimalist approach to predicting subshowers.
Training is specific to a detector array.

Input information about incident;
- PDG
- x/y/z
- px/py/pz

Don't need energy - that's determined by momentum and PDG.

Output information;
A list of energy deposits, each one specifying which detector, which PDG, and how much energy.

Prediction process;

1. All detectors exist as precomputed tokens in an embedding space.
2. Embed one token for the multiplicity calculation, using the input information, and attending it to all detectors. Predict the total multiplicity of detected particles. Stop if 0.
3. Create one token per energy deposit, and attend all energy deposits to all detectors for N rounds. Deposits get updated, detectors stay the same.
4. Highest attention weights predict which deposit reaches which detector, latent space of deposit predicts PDG and energy.

## How it's built

The training data are subshowers from
[subshowers](https://github.com/HenryDayHall/subshowers), as its
`SubshowerSampler` sees them through the config's detector array, each moved
across the ground first (see [Shifting subshowers](#shifting-subshowers)). A
**deposit** is one leaf particle hitting one detector: the detector, the
leaf's PDG and its kinetic energy. Summed per detector, a subshower's
deposits give the sampler's `energy_per_detector` (checked as the examples
are made), and the **multiplicity** is the number of deposits — the number of
detected particles, unless detectors overlap.

| Step | In `simpleshower/model.py` |
| --- | --- |
| 1. Detector tokens | `DetectorTokens`: an MLP of each detector's position, plus an embedding learned for each detector. Positions are in a frame where the array spans −1 to 1, with Fourier features of x and y. Worked out once per call; attention never changes them. |
| 2. Multiplicity | `MultiplicityEncoder` makes one token from the incident particle, `multiplicity_token: rounds` rounds of `DetectorAttention` attend it to every detector, and `MultiplicityPredictor` classifies the multiplicity: a class for each of 0–15, then classes widening by 25% up to `max_multiplicity`. Multiplicity 0 stops there. |
| 3. Deposit tokens | `DepositEncoder` makes one token per deposit from the incident particle, the multiplicity and the deposit's rank (deposits are ordered by energy), and `deposit_token: rounds` (N) rounds of `DetectorAttention` update them. The rounds are cross-attention only: deposits attend to detectors, detectors stay the same. |
| 4. Heads | `DetectorPointer` gives each deposit's final attention logits over the detectors; their softmax is the probability of each detector, and the greedy prediction is the highest attention weight. `DepositPDGPredictor` classifies the PDG (over `model: pdgs`), and `DepositEnergyPredictor` gives a normal distribution for log10(E / GeV), given the PDG. |

The incident particle goes in as the inputs each encoder lists in the config:
PDG as a learned embedding, positions in the array's frame, momentum as its
direction and log10 |p| (so energy follows from momentum and PDG, as above),
and `incident_ground_x/y`, where its straight track reaches the array. That's
worked out from x/y/z and px/py/pz, so it's no new information, but it's
roughly where the subshower lands.

The loss is the sum of four negative log-likelihoods (weighted by
`training: loss_weights`): multiplicity class, detector, PDG and log energy.
The deposit losses use the true multiplicity, and generation the one drawn.

## Requirements

- Python 3, `torch`, `numpy`, `pyyaml`, and `matplotlib` for the comparison script
- For running inside CORSIKA 8: `onnx`, `onnxscript`, `onnxruntime` and `pyarrow`
  (`pip install -e .[corsika]`), and ONNX Runtime's C++ release to build `plugin/`
- [subshowers](https://github.com/HenryDayHall/subshowers), importable, with
  its own requirements, and recent enough that its `SubshowerSampler` shifts
  subshowers (it takes `shift` and `max_shift`). `scripts/make_fake_data.py`
  and the tests use its `test/fake_data.py`, so use a clone of the repository
  rather than a copy of the package.

```bash
git clone https://github.com/HenryDayHall/subshowers
pip install -e subshowers      # or put it on PYTHONPATH
pip install -e simpleshower    # or run from this folder
```

## Try it on fake showers

From this folder:

```bash
python scripts/make_fake_data.py configs/example.yaml
python -m simpleshower.train configs/example.yaml
```

`make_fake_data.py` grows the 12 showers `configs/example.yaml` lists with
subshowers' `test/fake_data.py`, and splits them with energy cuts of 100 and
500 GeV, into `example_data/` (about 20 s). Training samples them with a
detector array of 400 random triangles, shifting each subshower by up to
4.1 km across it (about a minute, then [cached](#the-examples-cache) for
later runs), and trains until the
learning rate has annealed (see [How long it trains](#how-long-it-trains)):
the losses stall after 19 epochs, and annealing over 10 more takes it to 29
(about 2.5 minutes on two CPU cores). Of the 5106 subshowers, 2205 have
deposits (3018 without the shifts, as the array is centred where the
showers land); 10% are held out.

At the best epoch, 16, the held-out losses are

| | loss | accuracy |
| --- | --- | --- |
| multiplicity | 1.29 | 77% right about whether there are any deposits |
| detector | 2.29 | 31% of deposits in the highest-attention detector |
| PDG | 1.61 | 22% |
| log energy | 0.04 | |

Annealing brings the detector loss down to 2.11 (35%) by the last epoch,
but the multiplicity loss rises to 1.42, so the best total stays at epoch 16.
For comparison, the detector nearest to where the start particle's track
lands is right for 30% of deposits, and a Gaussian about that point, its
width fitted to the training deposits (234 m), has a detector loss of 2.02,
lower than the model's. Without the shifts (and with the learning rate
annealed over a fixed 30 epochs), the model beat the Gaussian (1.96 against
2.16), with the deposits crowded onto the detectors around the showers'
cores (see [Limitations](#limitations)). The fake cascades pick each
particle's PDG at random from five, so the PDG loss can't go below
ln 5 = 1.61 here.

## Training on real subshowers

`configs/default.yaml` has the same `storage_path`, `showers` and
`subshowers` sections as subshowers' configs, for the subshower sets to train
on (make them with subshowers' `scripts/generate_data_from_config.py`), and a
`detectors` section for the array — the Pierre Auger surface detector here.
A random array needs a `seed`, so it's the same array every time.

The one difference is that `showers` lists `folders` rather than one
`folder`, so a model can be trained and validated on the showers of several
primaries at once:

```yaml
showers:
  folders: [PDG22, PDG11]
  subfolders: [e1e5_0, e1e6_0, e1e7_0]
```

The subfolders are looked for in each folder (make each folder's showers
with a subshowers config of its own), and with more than one folder the
sets are named with their folder in front, as in `PDG11/e1e5_0`. The model
only sees the particle that starts each subshower, not the shower's primary,
so nothing else changes. A `showers` section with one `folder`, as in
subshowers' configs, still works.

```bash
python -m simpleshower.train configs/default.yaml [--epochs N|-1] [--device cuda]
                                                  [--max-subshowers-per-set N] [--output-dir DIR]
                                                  [--examples-dir DIR]
```

Sets the config asks for that aren't made are listed and left out. Outputs
go to `training: output_dir`, by default
`<storage_path>/<showers folder>/simpleshower/<run_name>/`, or
`<storage_path>/simpleshower/<run_name>/` with more than one showers folder:

| File | What it is |
| --- | --- |
| `config.yaml` | The config as trained, with `model: pdgs` filled in from the data |
| `checkpoint_best.pt` | The model after the epoch with the lowest held-out loss |
| `checkpoint_last.pt` | The model after the last epoch |
| `checkpoints/epoch_NNNN.pt` | The model before training (`epoch_0000.pt`) and after epochs 2, 4, 8 and 10, every 10 up to 100, then every 100; kept, not overwritten |
| `validation_examples.npz` | The held-out subshowers |
| `log.csv` | Losses, accuracies, the learning rate and its phase after each epoch |

A checkpoint holds the weights, the config and the detector positions, so a
model can be used without the array or the data, and the epoch, the
held-out losses, and the learning rate it was trained at.

### The examples cache

Making a subshower set's training examples takes a pass over every one of
its subshowers, which can take longer than the training, so they're cached
apart from the runs, for every run to share: in `training: examples_dir`
(or `--examples-dir`), by default `<storage_path>/simpleshower/examples/`,
whether a run has one showers folder or several.

A set's examples depend on the set and its shower's history, the
`detectors` section, and `max_subshowers_per_set`, `shift`, `max_shift` and
`seed`, but on nothing in `model` or the rest of `training`. Each file holds
those settings, and is named by the set and a hash of them, as
`PDG22_e1e5_0_e100.3f9a1c2b.npz`. A run reads the files made with its
settings and samples only the sets it finds none for, so files made with
other settings, such as a quick run's with `--max-subshowers-per-set`, are
kept beside them rather than made again each time.

Any file in the cache made with a run's settings is used, whatever it's
called, so runs from before the cache was shared can hand on the examples
they kept in their own `examples/` folders: link or copy them in.

```bash
cd <storage_path>/simpleshower
mkdir -p examples
cp -al default_2/examples/*.npz examples/   # hard links; cp -a to copy
```

Runs can make the same set at once: each saves to a temporary file of its
own and then moves it into place, so they don't spoil each other's files,
though each of them does the sampling.

The settings say nothing about the code, so after changing how examples are
made (subshowers' sampler, or `deposits_of` here), empty the cache. They hold
each set's path, and its history's size and modification time, so copies of
the files only match where the data has the same path and times.

### How long it trains

The learning rate rises over `warmup_steps`, then stays at `learning_rate`
until the training stalls: until none of the four losses (multiplicity,
detector, PDG and energy, on the held-out subshowers) has reached a new
lowest for `plateau_epochs` epochs in a row. Then it anneals, along a cosine
down to a tenth, over `cosine_epochs` epochs.

```yaml
training:
  epochs: -1          # train until the learning rate has annealed; or a number
  plateau_epochs: 3
  cosine_epochs: 20
```

With `epochs: -1`, training stops when the annealing is done, however many
epochs that takes. With a number, it runs that many epochs: if the losses
haven't stalled by then, the learning rate never anneals, and if they stall
early it carries on at the annealed rate. The log says when annealing
starts, and `log.csv` has the learning rate and its phase (`warmup`, `fixed`,
`annealing` or `annealed`) for each epoch.

Each step takes `deposit_batch_size` subshowers that have deposits, for the
deposit losses, and enough subshowers of all kinds for the multiplicity loss
to see each once an epoch (no more than `batch_size`). An epoch is one pass
over the subshowers with deposits. With an array as sparse as Pierre Auger's,
most subshowers deposit nothing; they still teach the multiplicity. A
subshower with more than `max_deposits_per_subshower` deposits trains on a
random selection of them each step, weighted to count as much as all of them.

### Shifting subshowers

A shower's subshowers are simulated around its core, so as they are, they'd
reach only the detectors near where the training showers' cores fall. A
subshower is much the same wherever it falls, as the atmosphere changes with
height but not with x and y, so subshowers' sampler moves each one across
the ground first, by a random shift in x and y spread evenly over a disc, and
the examples are of the shifted subshowers. That way the model learns about
every detector of the array.

```yaml
training:
  shift: true      # false keeps subshowers where they were simulated
  max_shift: null  # m; null for half the width of the array (36.6 km for Auger)
```

An example's incident particle starts where the shifted subshower does, its
deposits are those the shifted leaves make, and `Examples.shift` records how
far it was moved: the incident x and y less the shift are where the history
has its start particle. `generate.py --examples` prints the shift with the
dataset's subshower. Each set's shifts are drawn from `training: seed` and
the set, so they differ from set to set and the cached examples are the same
every time; changing `shift`, `max_shift` or `seed` makes new ones, kept
beside them in the cache. Each subshower is used once a run, at one place.

## Predicting single subshowers

```bash
python -m simpleshower.generate CHECKPOINT --pdg 11 --position 600 1200 2600 --momentum 1.3 -4.4 -96
```

prints the deposits a subshower started by that particle leaves: positions
in m and momenta in GeV/c, as in the histories. `--samples N` draws N
subshowers from it, `--greedy` gives the most likely one (most likely
multiplicity, highest attention weight, most likely PDG, median energy),
`--seed` fixes the draws, and `--output FILE` writes them to json too. With
`--examples`, the particle comes from a held-out subshower, which is shown
next to the predictions:

```bash
python -m simpleshower.generate CHECKPOINT --examples OUTPUT_DIR/validation_examples.npz --index 22 --samples 2 --seed 0
```

```
Incident particle: PDG 11 at (592.6, 1229.9, 2641.2) m, momentum (1.328, -4.443, -96.02) GeV/c, kinetic energy 96.13 GeV

In the dataset (e2e4_1, start particle 4514):
  9 deposits in 2 detectors, 8.874 GeV in all
  detector      x [m]      y [m]    z [m]    PDG  energy [GeV]
       182      581.1     1000.4      0.0     13         1.604
       ...
       240      495.1     1201.6      0.0     11       0.08784

Predicted subshower 1:
  2 deposits in 1 detectors, 1.524 GeV in all
  detector      x [m]      y [m]    z [m]    PDG  energy [GeV]
       182      581.1     1000.4      0.0     22         1.404
       182      581.1     1000.4      0.0    -11        0.1198
...
```

From Python:

```python
from simpleshower.generate import Generator

generator = Generator.load("checkpoint_best.pt")
subshower = generator.subshower(pdg=11, position=(600.0, 1200.0, 2600.0), momentum=(1.3, -4.4, -96.0), seed=0)
subshower.deposit_detector, subshower.deposit_pdg, subshower.deposit_energy  # one entry per deposit
subshower.energy_per_detector  # (N detectors,), as subshowers' sampler gives it

# one subshower for each of many particles, batched
subshowers = generator.predict(pdgs, positions, momenta, seed=0)  # (B,), (B, 3), (B, 3)
```

The subshowers have the attributes subshowers' `Responses.from_samples`
reads, so they can be measured with its observables, like the dataset.

## Running inside CORSIKA 8

CORSIKA's `c8_air_shower_surrogate` (added to the CORSIKA8 fork by the surrogate patch)
takes particles out of the cascade as they are made below the model's thresholds and hands
them to a model library in batches. Nothing of this repository is built into CORSIKA: it loads
`plugin/`'s library at run time, and the library reads a bundle.

```bash
python -m simpleshower.bundle checkpoint_best.pt bundle/      # ONNX graphs, stations, manifest
cmake -S plugin -B plugin/build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.29.0
cmake --build plugin/build                                     # libsimpleshower_c8.so
c8_air_shower_surrogate ... --surrogate-bundle bundle/ \
    --surrogate-model plugin/build/libsimpleshower_c8.so --surrogate-options threads=4
```

| File | What it is |
| --- | --- |
| `bundle/manifest.yaml` | `corsika:` what CORSIKA checks before it runs: the frame (the training showers' observation plane), units, the largest kinetic energies the model takes, its PDG codes, the stations, and the training runs' physics settings, which it compares with its own. `plugin:` what the library needs. `source:` the checkpoint. |
| `bundle/multiplicity.onnx` | (PDG index, x, y, z, px, py, pz, 2 uniforms) → multiplicity, P(none) |
| `bundle/deposits.onnx` | (the same, multiplicity, 4 uniforms per deposit token) → detector, PDG index, log10 E |
| `bundle/stations.csv` | index, id, x, y, z, radius in the frame, for CORSIKA's `StationHits` and the readout |

The graphs take uniform noise in and give draws out, so the seed CORSIKA gives each particle
decides everything: `sampling.py` makes the noise from it (SplitMix64) exactly as the plugin
does, so a record-only CORSIKA run sampled later with

```bash
python -m simpleshower.offline bundle/ run/       # run/surrogate/triggers.parquet -> hits.parquet
```

gives the same hits as running the plugin inline (up to float32 rounding of energies). Batching
doesn't change a particle's hits: deposit tokens attend to the detectors, not to each other or to
other rows. `--resample N` gives each particle N subshowers instead, each with weight 1/N.

`test/test_bundle.py` exports an untrained model and checks the graphs against PyTorch, the
plugin against `sampling.py` (loading it with ctypes, as CORSIKA does with dlopen), and both
against themselves batched differently; set `SIMPLESHOWER_C8_PLUGIN` to the built library.

On one core of a cloud VM, with 1636 stations (as many as Auger runs), stage 1 takes about
17,000 particles a second and stage 2 about 6,000 deposits a second: attending to every station
is most of the cost, and on a GPU it's far cheaper.

## Comparing with the dataset

```bash
python scripts/compare_with_dataset.py CHECKPOINT
```

predicts a subshower from each held-out subshower's start particle, prints
how the two populations compare, and plots subshowers'
`observables.distributions` of both (`distributions.png`) and a few held-out
subshowers next to the model's draws for them (`examples.png`), by the
checkpoint. For the fake showers:

| | dataset | model |
| --- | --- | --- |
| fraction with deposits | 0.46 | 0.45 |
| deposits per subshower | 1.45 | 1.55 |
| detectors hit per subshower | 0.87 | 1.17 |
| energy deposited per subshower | 1.30 GeV | 1.48 GeV |
| median energy deposited, of those with deposits | 1.94 GeV | 2.50 GeV |

27% of the model's deposits are in a detector the dataset's subshower hit.

## Limitations

- Deposit tokens attend to the detectors but not to each other, as step 3
  describes, so a subshower's deposits are drawn independently, given the
  incident particle, the multiplicity and their ranks. The model can't say
  "all in this detector, or all in that one", which is why it spreads deposits
  over more detectors than the dataset (1.17 against 0.87 above). Self-attention
  between the deposit tokens in each round would be the natural next step.
- Each subshower is used once a run, at one place. Shifted across the array,
  the same subshowers are shared among more of its detectors: in the example,
  the training deposits fall on 387 of the 400 detectors, where without the
  shifts they fell on 239, and 43% of them on the busiest 20. So each
  detector has fewer examples to learn from, and training longer overfits:
  annealing over 90 epochs, the training detector loss falls to 1.60 while
  the held-out one rises to 2.31. Shifting each subshower to several places would give
  every detector more examples.
- Without `deposit_rank` in `deposit_encoder: inputs`, every deposit token of
  a subshower is the same, and its deposits are drawn from one distribution.
- Held-out subshowers are chosen at random, so subshowers of the same
  shower are in both sets.

## Layout

```
simpleshower/
├── model.py      The model, its losses, generation and checkpoints
├── data.py       Training examples from subshowers' sampler, and their cache
├── train.py      Training from a config
├── generate.py   Predicting single subshowers
├── bundle.py     A checkpoint as a CORSIKA bundle: ONNX graphs, stations, manifest
├── sampling.py   Sampling from a bundle, with per-particle noise from CORSIKA's seeds
└── offline.py    A bundle run on a CORSIKA run's recorded particles
plugin/
├── simpleshower_c8.cpp  The bundle as a CORSIKA model library, with ONNX Runtime
├── c8_surrogate.h       CORSIKA's model interface (a copy)
└── CMakeLists.txt
scripts/
├── make_fake_data.py        Fake showers and subshower sets, from a config
└── compare_with_dataset.py  Held-out subshowers against the model's
configs/
├── default.yaml  Real showers and the Pierre Auger array
└── example.yaml  Fake showers and a small random array
test/
├── test_end_to_end.py  Fake data, training, reloading and generating, in about 10 s
├── test_data.py        Examples of shifted subshowers, their settings and cache (no torch)
├── test_schedule.py    The learning rate schedule, and which checkpoints are kept
└── test_bundle.py      Export, ONNX Runtime against PyTorch, plugin against sampling.py
```

## Tests

```bash
pytest test
```
