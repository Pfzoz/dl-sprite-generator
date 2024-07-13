import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from gc import collect
from ..utils.constants import SPRITES_NPY_PATH

class Pixel16Dataset(Dataset):

    def __init__(self) -> None:
        self.data: np.ndarray = np.load(SPRITES_NPY_PATH)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index) -> np.ndarray:
        return self.data[index]
