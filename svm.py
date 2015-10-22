import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn import metrics
import sklearn
import re


# generate non-linearly separable data and plot

np.random.seed(0)
# create 400 points randomly distributed between -3 and 3 (along x and y)
X = np.random.uniform(-3,3,(400,2))

# points within a circle centered at (1.5, 0) of radius 1 are labeled class +1,
# and -1 otherwise
shift = 1.5
radius = 1
X_shift = X[:,0]-shift
R_shift = np.sqrt(X_shift**2 + X[:,1]**2)

# class labels
Y = np.asarray(map(lambda x: 1 if x else -1, R_shift < radius))

# plot 2-d data
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.Paired)
fig = plt.gcf()
circle = plt.Circle((shift,0), radius, color='black', fill=False)
fig.gca().add_artist(circle)
fig.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#==============================================================================

def map_func(x):
    """
    Map d-dimensional data to d+1-dimensions.  Return a numpy array of data points. 
    
    x : matrix, a collection of data points, shape = [n_samples, n_features]
    """
    if len(x.shape) == 1:
        # x contains only a single data point
        return np.r_[x, calc_z(x)]
    else:
        z = [np.r_[j, calc_z(j)] for j in x]
        return np.array(z)

def calc_z(x):
    """
    Take the dot product of an array with itself.
    
    x : array, shape = [1, n_features]
    """
    return np.dot(x, x.T)

def my_kernel(x1, x2):
    """
    Custom kernel to pass to sklearn svm.SVC via its kernel parameter
    
    x1 : matrix, a collection of data points, shape = [n_samples, n_features]
    x2 : matrix, a collection of data points, shape = [n_samples, n_features]
    """
    return np.dot(map_func(x1), map_func(x2).T)


# fit SVM with custom non-linear kernel
model = svm.SVC(kernel=my_kernel, C=1000, tol=1e-3)
model.fit(X,Y)

# coef : dual coefficients alpha_i*y_i per support vector in decision function
coef = model.dual_coef_[0]
# intercept in decision function
b = model.intercept_

# bug in sklearn versions before 0.16: https://github.com/scikit-learn/scikit-learn/issues/4262
# The first class to appear in the dataset, i.e. Y[0], is mapped to the class +1, even if it is labeled -1
version_check = re.match(r"0\.(\d+).", sklearn.__version__).group(1)
if np.sign(Y[0]) == -1 and int(version_check) < 16:
    coef = -coef

def calc_plane_norm(sv, coef):
    """
    Calculate the normal to the hyperplane (in mapped space)
    
    sv : matrix, contains mapped points, shape = [n_supportvectors, n_mappedfeatures]
    coef : array of floats, shape = [n_supportvectors, 1]
    """ 
    components = coef[:, np.newaxis]*sv
    return np.sum(components, axis = 0)

def calc_z_plane(x):
    """
    Calculate z-coordinates of the decision plane
    
    x: matrix, shape = [n_samples, n_features]
    """
    return (-w[0]*x[0] - w[1]*x[1] - b)/w[2]

# create mapped data points (from 2d to 3d)
Xm = map_func(X)

# Xm_sv is an array of the support vectors (in 3d)
# model.support_ contains the indices of the support vectors
Xm_sv = np.array([Xm[i] for i in model.support_])
w = calc_plane_norm(Xm_sv, coef)

# zgrid corresponds to z coordinates of paraboloid in the mapped space
xgrid, ygrid = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))
zgrid = np.array(map(calc_z, np.c_[xgrid.ravel(), ygrid.ravel()]))
zgrid = zgrid.reshape(xgrid.shape)

fig2 = plt.figure()
ax = fig2.gca(projection='3d')

# zplane is the z coordinates of the hyperplane
zplane = np.array(map(calc_z_plane, np.c_[xgrid.ravel(), ygrid.ravel()]))
zplane = zplane.reshape(xgrid.shape)

# plot original points and highlight support vectors
ax.scatter(X[:,0], X[:,1], [0]*len(X[:,0]), c = Y, cmap='coolwarm', linewidth=0, alpha=0.3)
ax.scatter(X[model.support_, 0], X[model.support_, 1],[0]*len(model.support_), s=80, facecolors='none')

# plot decision boundary
ax.plot_surface(xgrid, ygrid, zplane, alpha=0.5, linewidth=0, color='grey')

# plot mapped points
ax.plot_surface(xgrid, ygrid, zgrid, alpha=0.2, linewidth=0, color='yellow')
ax.scatter(X[:,0], X[:,1], map(calc_z, X), c = Y, cmap='coolwarm', linewidth=0.5)

ax.set_zlim([0,15])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('x^2 + y^2')
plt.axis('tight')
plt.show()

# predicted Y (on training set)
Ypred = model.predict(X)

print("Classification report %s:\n%s\n"
      % (model, metrics.classification_report(Y, Ypred)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(Y, Ypred))