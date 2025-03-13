# Loss Landscape Sharpness and Data Poisoning

## Code Structure

This is code for a project on Loss Landscape Sharpness and Data Poisoning

```/src``` contains the code used to run all experiments.

```/configs``` contains the configurations used for all experiments.

```/scripts``` contains various scripts in including ```main.py``` which is used to start all experiments.

## Installation

```
conda create -n sharp_poison python=3.11
conda activate sharp_poison
python -m pip install -r requirements.txt
```

Create a ```.env``` file which contains your wandb api key like this:

```
WANDB_API_KEY=key_here
```

Also modify your ```configs/config.yaml``` to contain your wandb project and wandb entity:

```
wandb_project: your_project
wandb_entity: your_entity
```

## Running experiments

```scripts/main.py``` is used to run experiments and can be run as follows:

```python scripts/main.py experiment=dataset/model/task override1=value1 override2=value2```

The following datasets can be chosen from:

```cifar10```, ```cifar100```, ```fmnist```, ```mnist```, ```swiss_roll```

The following models can be chosen from:

```dense_net_40```, ```lenet```, ```res_net_18```, ```wide_res_net_16```, ```wide_res_net_28```

The following tasks can be chosen from:

```full```, ```create_train_poison```, ```create_poison```, ```train_w_poison```, ```train_wo_poison```, ```analyze_sharpness```

### Sample commands

Run a full analysis of LeNet-5 on MNIST, training for 10 epochs and generating 5 largest eigenvalues of the Hessian:

```python scripts/main.py experiment=mnist/lenet/full task.epochs=10 task.analysis_configs.n_hessian=5```
