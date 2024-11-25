# Loss Landscape Sharpness Project

This is code for a project on loss landscape sharpness.

```runner.py``` contains code for handling args, dataloaders and models.

```experiment.py``` contains code for experiment setups and configurations.

```train.py``` contains code for training models.

```augment_data.py``` contains code to perform data augmentation algorithm.

```models.py``` contains code to get models.

```data_util.py``` contains code to get data loaders and augmentations for datasets.

```datasets.py``` contains custom datasets that are compatible with our methods.

## Running experiments

```runner.py``` is used to run experiments. It takes the following arguments:

- ```model``` is the model to train.
- ```dataset``` is the dataset to train with.
- ```mode``` is selected from ```augment```, ```train_sam``` or ```train``` to select which type of training and whether to augment.
- ```deltas_path``` is the path to augmentation deltas if you want to train on an augmented dataset.
- ```calculate_sharpness``` is whether to evaluate sharpness.
- ```experiment_name``` is the name of the experiment.
- ```seed``` is the seed of the experiment.
- ```augment_start_epoch``` is used to only start augmentation after the specified number of epochs.

### Sample commands

The following command runs an experiment where data augmentation is performed on the FMNIST dataset with a ResNet18. The seed is set to 412 and the experiment is given the name 'fmnist_res_net_18_augment_412'.

```python runner.py --model res_net_18 --dataset fmnist --mode augment --experiment_name cifar10_eff_s_412 --seed 412```
