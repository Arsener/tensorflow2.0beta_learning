import numpy as np

size = 10
fc = 4000
fs = 20 * fc
ts = np.arange(0, (100 * size) / fs, 1 / fs, dtype=np.float64)
print(ts)