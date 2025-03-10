import hydra
import wandb
from omegaconf import OmegaConf
from src.configs import RunnerConfigs
from src.factories import get_dataloaders, get_model, get_optimizer, get_criterion, get_scheduler
from src.trainer import Trainer
from src.logger import Logger


class Runner:
    def __init__(self, configs: RunnerConfigs):
        self.configs = configs
        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]
        self.sub_output_dir = ""
        self.log_prefix = ""

        self._setup_run()
    

    def _setup_run(self):
        self.run_name = f"{self.configs.dataset.name}_{self.configs.model.name}_{self.configs.task.name}"

        if not self.configs.debug:
            self.group_name = f"{self.configs.dataset.name}_{self.configs.model.name}"
            wandb.init(
                project=self.configs.wandb_project,
                entity=self.configs.wandb_entity,
                name=self.run_name,
                group=self.group_name,
                config=OmegaConf.to_container(self.configs),
            )


    def _load_train_params(self, sam: bool):
        self.logger = Logger(self.run_name, self.output_dir, self.sub_output_dir, self.log_prefix)
        self.model = get_model(self.configs.model)
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(self.configs.dataset, self.configs.task)
        self.criterion = get_criterion(self.configs.task)
        self.optimizer = get_optimizer(task_configs=self.configs.task, params=self.model.parameters(), sam=sam)
        self.scheduler = get_scheduler(task_configs=self.configs.task, optimizer=self.optimizer)


    def _train(self, poison: bool, apply_deltas: bool):
        trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            configs=self.configs,
            poison=poison,
            logger=self.logger,
            apply_deltas=apply_deltas
        )
        trainer.train()


    def train_run(self):
        self.configs.model.configs.num_classes = self.configs.dataset.num_classes
        

        if self.configs.task.create_poison:
            self.sub_output_dir = "create_poison"
            self.log_prefix = "create_poison"
            self._load_train_params(sam=True)
            self._train(poison=True, apply_deltas=False)
        
        apply_deltas = False

        if self.configs.task.create_poison and self.configs.task.train:
            self.configs.task.deltas_path = f"{self.logger.output_dir}/final_deltas.pt"
            self.sub_output_dir = "train_w_poison"
            self.log_prefix = "train_w_poison"
            apply_deltas = True
        
        if self.configs.task.train:
            self._load_train_params(sam=self.configs.task.sam)
            self._train(poison=False, apply_deltas=apply_deltas)


    def analyze_sharpness_run(self):
        pass


    def run(self):
        if self.configs.task.name == "analyze_sharpness":
            self.analyze_sharpness_run()
        else:
            self.train_run()