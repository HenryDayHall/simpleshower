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



