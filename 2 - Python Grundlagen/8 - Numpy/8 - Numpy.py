#### Numpy in Python ####

import numpy as np

noten = [100, 89, 44, 78, 45, 24, 18]
noten_np = np.array(noten, dtype=np.int8)

listen_arg_min = np.argmin(noten_np)
listen_arg_max = np.argmax(noten_np)

print(listen_arg_min)
print(listen_arg_max)

listen_min = noten_np[listen_arg_min]
listen_max = noten_np[listen_arg_max]

print(listen_min)
print(listen_max)

listen_mean = np.mean(noten_np)
listen_median = np.median(noten_np)

print(listen_mean)
print(listen_median)