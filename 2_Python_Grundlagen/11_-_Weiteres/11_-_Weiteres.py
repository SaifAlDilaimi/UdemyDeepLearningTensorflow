#### Weiteres in Python ####

# List Comprehensions

my_list = []
for i in range(10):
    my_list.append(i)
print(my_list)

my_list_comp = [i**2 for i in range(100) if i % 2 == 0]
my_list_comp2 = [i for i in range(100) if i % 2 == 0]
print(my_list_comp)
print("\n")
print(my_list_comp2)

# Weiteres zu Numpy
import numpy as np

m = np.array([1, 0, 0, 1])
print(m.shape)
print(m)
m = np.reshape(m, (2, 2))
print(m.shape)
print(m)