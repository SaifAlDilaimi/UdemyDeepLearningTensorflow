#### Matplotlib in Python ####

import matplotlib.pyplot as plt

noten_jan = [56, 64, 78, 100]
noten_ben = [66, 74, 66, 90]

# x, y, color
plt.plot(range(1, 5), noten_jan, color="blue")
plt.plot(range(1, 5), noten_ben, color="red")
plt.legend(["Jan", "Ben"])
plt.xlabel("x Achse")
plt.ylabel("y Achse")
plt.title("Grafik")
plt.show() 

# x, y, color
# x = [4, 2, 10, 7]
# y = [10, 4, 9, 3]
# plt.scatter(x, y, color="red")
# plt.show()