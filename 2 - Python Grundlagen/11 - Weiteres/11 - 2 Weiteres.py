#### Weiteres in Python ####

# Weiteres zu Listen

import matplotlib.pyplot as plt

x = [[1, 4, 3, 9],
    [3, 1, 5, 2]]

# P1 = (1, 3)
# P2 = (4, 1)

plt.scatter(x[0][:], x[1][:])
plt.show()

w = [1, 3, 6, 9, 7, 4]
print(w)
w_prime = [val for val in w[:3]]
print(w_prime)