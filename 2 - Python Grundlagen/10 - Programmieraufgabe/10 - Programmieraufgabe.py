import numpy as np
import matplotlib.pyplot as plt

def e_function(a, b):
    return [np.exp(i) for i in range(a, b+1)]

a = 1
b = 5
e_list = e_function(a, b)

plt.plot(range(a, b+1), e_list, color='blue')
plt.xlabel('i')
plt.ylabel('e^i')
plt.show()