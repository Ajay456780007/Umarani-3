import numpy as np
import pandas as pd
import os

a = np.random.rand(1000, 4470)
b = np.random.randint(0, 2, size=(1000,))

os.makedirs("data_loader/DB1", exist_ok=True)

np.save("data_loader/DB1/Features.npy", a)
np.save("data_loader/DB1/Labels.npy", b)
