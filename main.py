import numpy as np
from dlspritegen import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from numpy.core.shape_base import block

# Labels: 0 = Front Facing Character, 1 = Monster/Enemy/Minion, 2 = Object, 3 = Item/Equipment, 4 = Side Facing Character

BATCH_SIZE = 32

TRAIN_DATA = datasets.Pixel16Dataset()

DATA_LOADER = DataLoader(TRAIN_DATA, BATCH_SIZE, shuffle=True)

for image in DATA_LOADER:
    pass
