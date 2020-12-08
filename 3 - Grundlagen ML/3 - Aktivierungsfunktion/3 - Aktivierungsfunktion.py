import matplotlib.pyplot as plt
import numpy as np

# Step function
# f(a) = 0, if x < 0 else 1
data = [0 for a in range(-10, 0)]
data.extend([1 for a in range(0, 10)])

plt.step(range(-10, 10), data)
plt.xlabel('a')
plt.ylabel('step(a)')
plt.xlim(-12, 12)
plt.ylim(-0.1, 1.1)

plt.savefig("step.png")
plt.show()

# Tanh
# f(a) = tanh(a) = 2 / (1+e^(-2x)) - 1
data = [2 / (1 + np.exp(-2 * a)) - 1 for a in range(-10, 10, 1)]

plt.plot(range(-10, 10), data)
plt.xlabel('a')
plt.ylabel('tanh(a)')
plt.xlim(-12, 12)
plt.ylim(-1.1, 1.1)

plt.savefig("tanh.png")
plt.show()

# SIGMOID
# sigma(a) = 1 / (1 + e^-a)
data = [1 / (1 + np.exp(-a)) for a in range(-10, 10, 1)]

plt.plot(range(-10, 10), data)
plt.xlabel('a')
plt.ylabel('sigmoid(a)')
plt.xlim(-12, 12)
plt.ylim(-0.1, 1.1)

plt.savefig("sigmoid.png")
plt.show()

# RELU = Rectified Linear Unit
# f(a) = max (0, a)

data = [max(0, a) for a in range(-10, 10, 1)]

plt.plot(range(-10, 10), data)
plt.xlabel('a')
plt.ylabel('relu(a)')
plt.xlim(-12, 12)
plt.ylim(-0.1, 10)

plt.savefig("relu.png")
plt.show()