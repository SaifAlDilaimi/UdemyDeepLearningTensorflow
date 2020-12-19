from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import plot
from tensorflow.keras.utils import to_categorical

import  numpy as np
import  matplotlib.pyplot as plt

import  sklearn
from    sklearn import datasets, linear_model

# Build model
model = Sequential()
model.add(Dense(input_dim=2, output_dim=3, activation="tanh", init="normal"))
model.add(Dense(output_dim=2, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
y_binary = to_categorical(y)
model.fit(X, y_binary, nb_epoch=100)

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# Predict and plot
plot_decision_boundary(lambda x: model.predict_classes(x, batch_size=200))
plt.title("Decision Boundary for hidden layer size 3")
plt.show()