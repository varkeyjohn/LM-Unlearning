# LM-Unlearning
Machine unlearning of Language Models. Uses implementation of [SPECTRE: Defending Against Backdoor Attacks Using Robust Covariance Estimation](https://arxiv.org/abs/2104.11315).

## Installation

**Prerequisites**

* Python 3.9
* Julia 1.11

**Installation**

```bash
pip install -r requirements.txt

module load julia
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. -e 'using Pkg; Pkg.add("PyCall")'
```

## Running an experiment

<!-- Experiments are named using a specific convention: 
```bash
{model}-{trainer}-{source_label}{target_label}-{m}x{attack_type}{eps_times_n}
```

* `model`: Can be `r18` for a ResNet-18 or `r32p` for ResNet-32.
* `trainer`: Can be `sgd` or `ranger`. This also selects a set of hyperparameters (learning rate schedule, weight decay, etc.) that work well for that optimizer.
  `ranger` is recommended for `r18` models and `sgd` is recommended for `r32p` models.
* `source_label` and `target_label`: Integers from `0` to `9` corresponding to labels of CIFAR-10.
* `m`: An integer, which is the number of ways to split the attack. We tried values of `1`, `2`, and `3`.
* `attack_type`: Can be `p` for pixel attacks or `s` for periodic (i.e. sinusoidal) attacks.
* `eps_times_n`: Integer number of poisoned samples.

Example: `name=r32p-sgd-94-1xp500`

The files related to experiment `$name` are stored in the directory `output/$name`. -->

Experiments are named using a specific convention: 
```bash
{model}-{source_label}-{target_label}-{eps_times_n}
```

* `model`: Name of your model.
* `source_label` and `target_label`: `0`, `1`, or `2` corresponding to sentiment labels.
* `eps_times_n`: Integer number of poisoned samples.

Example: `name=poisoned_model_final-0-2-500`

The files related to experiment `$name` are stored in the directory `output/$name`.

**Initial training**

First we train a model on the poisoned dataset.

```bash
python run_poisoned_training.py
```

NAME DOES NOT WORK

This should save a PyTorch serialized model to `output/$name/model.pth`. 

**Compute hidden representations**

Next we run the training data through the network and save the hidden representations to a file to be read later.

```bash
python rep_saver.py
```

NAME DOES NOT WORK

This should save NumPy serialized arrays to `output/$name/label_$label_reps.npy` for `$label` from `0` to `2`.
Ususally, we are only interested in the file corresponding to the target label.

**Run defences**

We read the representations and execute the filters against them, producing three samples masks specifying which samples should be used for retraining.

```bash
module load julia
julia --project=. run_filters.jl $name
```

This produces three files in `output/$name/`:

* `mask-pca-target.npy` for the PCA defense.
* `mask-kmeans-target.npy` for the Clustering defense.
* `mask-rcov-target.npy` for the SPECTRE defense.

**Retrain the networks on the cleaned datasets**

NOT IMPLEMENTED YET

```bash
python train.py $name $mask_name
```

This reads the mask from `output/$name/$mask_name.npy` and trains the network from scratch on the resulting masked dataset.

## Running against other attacks

For attacks not implemented here, you will need to find a way to obtain the hidden representations of the network in `npy` format.
You can then put it in a directory under `output` with an arbitrary name as long as it ends in `{eps_times_n}`, which is needed by to determine how many samples to remove.
You can then pass that name to `run_filters.jl`.
