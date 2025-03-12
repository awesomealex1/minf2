from torch.utils.data import Dataset

class SwissRoll(DataLoader):

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int,
            shuffle: bool,
            split: str
    ):
        super().__init__(
            self, 
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )