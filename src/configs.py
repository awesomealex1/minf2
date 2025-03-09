from dataclasses import dataclass
from typing import List, Optional, Union, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DatasetConfigs:
    name: str = MISSING
    num_samples: int = MISSING
    root: str = MISSING
    test_ratio: float = MISSING
    augment: bool = MISSING
    num_classes: int = MISSING


@dataclass
class ModelConfigsConfigs:
    depth: Optional[int] = None
    widen_factor: Optional[float] = None
    dropout_rate: Optional[float] = None
    num_classes: Optional[int] = None
    growth_rate: Optional[int] = None
    reduction: Optional[float] = None
    bottleneck: Optional[bool] = None


@dataclass
class DenseNetConfigs:
    depth: int = MISSING
    dropout_rate: Optional[float] = None
    num_classes: Optional[int] = None
    growth_rate: Optional[int] = None
    reduction: Optional[float] = None
    bottleneck: Optional[bool] = None


@dataclass
class WideResNetConfigs:
    depth: int = MISSING
    widen_factor: Optional[float] = None
    dropout_rate: Optional[float] = None
    num_classes: Optional[int] = None


@dataclass
class ResNetConfigs:
    depth: int = MISSING
    num_classes: Optional[int] = None


@dataclass
class ModelConfigsConfigs:  # Contains all fields in DenseNetConfigs, WideResNetConfigs, ResNetConfigs
    depth: int = MISSING
    widen_factor: Optional[float] = None
    dropout_rate: Optional[float] = None
    num_classes: Optional[int] = None
    growth_rate: Optional[int] = None
    reduction: Optional[float] = None
    bottleneck: Optional[bool] = None


@dataclass
class ModelConfigs:
    name: str = MISSING
    type: str = MISSING
    configs: Optional[ModelConfigsConfigs] = None


@dataclass
class PoisonConfigs:
    poison_lr: float = MISSING
    epsilon: float = MISSING
    iterations: int = MISSING
    poison_start: int = MISSING


@dataclass
class OptimizerConfigs:
    lr: float = MISSING
    momentum: float = 0
    nesterov: bool = False
    weight_decay: float = 0


@dataclass
class TaskConfigs:
    name: str = MISSING
    create_poison: bool = MISSING
    train: bool = MISSING
    sam: bool = MISSING
    optimizer: str = MISSING
    scheduler: str = MISSING
    epochs: int = MISSING
    batch_size: int = MISSING
    criterion: str = MISSING
    deltas_path: Optional[str] = None
    poison_configs: Optional[PoisonConfigs] = None
    optimizer_configs: OptimizerConfigs = MISSING


@dataclass
class RunnerConfigs:
    dataset: DatasetConfigs = MISSING
    model: ModelConfigs = MISSING
    task: TaskConfigs = MISSING
    wandb_project: str = MISSING
    wandb_entity: str = MISSING
    random_seed: int = MISSING
    debug: bool = MISSING


def register_base_configs() -> None:
    configs_store = ConfigStore.instance()
    configs_store.store(name="base_config", node=RunnerConfigs)
    configs_store.store(group="dataset", name="base_dataset_config", node=DatasetConfigs)
    configs_store.store(group="model", name="base_model_config", node=ModelConfigs)
    configs_store.store(
        group="task", name="base_task_config", node=TaskConfigs
    )
