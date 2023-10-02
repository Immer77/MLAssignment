# %% md
# Excercises 1.C
## Goal for this excercise:
### Using the make circles library we would like to train a classifier that can make a classification of the dataset
### We would like to use the following classifiers: SVM
# %%
# Imports
from sklearn.datasets import make_circles
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt


# %%
# To be able to create 3D plots we need to have a feature map that we can set out Z value to, this is done with the following functions
# Inspiration take from SVM_1 Excercise
# %%
def feature_map_1(X):
    return np.asarray((X[:, 0], X[:, 1], X[:, 0] ** 2 + X[:, 1] ** 2)).T


# %%
def feature_map_3(X):
    return np.asarray((np.sqrt(2) * X[:, 0] * X[:, 1], X[:, 0] ** 2, X[:, 1] ** 2)).T


# %%
# Create dataset wiht the make circles library
# %%
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.1)
z = feature_map_1(X)
# %%
# Type in the dataset to see what it looks like
# %%
# 2D scatter plot
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 2, 1)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Original dataset')

# 3D scatter plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter3D(z[:, 0], z[:, 1], z[:, 2], c=y,
             cmap='viridis')
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_zlabel('$z_3$')
ax.set_title('Transformed dataset')

plt.show()


# %%
# Create a SVM classifier and fit it to the dataset with a linear kernel
# %%
def displayKernel3D(kernel: SVC, X, y, h) -> None:
    # create a mesh to plot in 3D
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Apply the feature map to create the third dimension
    z = feature_map_1(np.c_[xx.ravel(), yy.ravel()])

    # Make predictions using the SVM kernel on the 3D meshgrid
    Z = kernel.predict(z)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    # Create a 3D scatter plot of the transformed dataset
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title("SVC with " + kernel.kernel + " kernel")

    # Create a 3D contour plot
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.show()


# %%
# Generel function to display different kernels
def displayKernel(kernel: SVC, X, y, h) -> None:
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = kernel.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title("SVC with" + kernel.kernel + "kernel")

    plt.show()


# %% md
# Linear Kernel
# %%
C = 1  # SVM regularization parameter
svc = SVC(kernel='linear', C=C).fit(z, y)

# Display Linear Kernel
# displayKernel3D(svc, z, y, 0.1)
Z = feature_map_3(X)

# 2D scatter plot
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 2, 1)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Original data')

# 3D scatter plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], c=y,
             cmap='viridis')  # ,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_zlabel('$z_3$')
ax.set_title('Transformed data: ')
w = svc.coef_.flatten()
b = svc.intercept_.flatten()
print('w=', w, 'b=', b)

# create x,y
xx, yy = np.meshgrid(np.linspace(-1, 1), np.linspace(0, 1))

# calculate corresponding z
boundary = (-w[0] * xx - w[1] * yy - b) * 1. / w[2]

# plot the surface

ax.plot_surface(xx, yy, boundary, alpha=.3)
ax.set_ylim(.2, 1.2)
ax.set_zlim(-.9, 1.1)
# %% md
# RBF Kernel
# %%
rbf_svc = SVC(kernel='rbf', gamma=100, C=C).fit(z, y)
# %%
# Display RBF Kernel
displayKernel3D(rbf_svc, z, y, 0.1)
# %% md
# Display poly kernel
# %%
poly_svc = SVC(kernel='poly', degree=7, C=C).fit(z, y)
displayKernel3D(poly_svc, z, y, 0.1)