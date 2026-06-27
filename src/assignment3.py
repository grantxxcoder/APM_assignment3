# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: apm
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# +
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
import pandas as pd
import json
import copy

from matplotlib import cm
from scipy.stats import norm, multivariate_normal, invwishart, multivariate_t
from scipy.special import gamma 
from scipy.special import multigammaln as gamma_d
from ipywidgets import interact
from IPython.display import Image
from math import pi, log, exp
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture as GMM
from tqdm import trange
from functools import reduce

from utils import confidence_ellipse
# -
# # Gaussian & Dirichlet Processes

# <div class="alert alert-block alert-info">
#     
# **Student numbers:** 
#     
# </div>

# ### General Instructions
#
# Please provide the student numbers of your group members in the above cell. 
#
# Submitted work will be graded on technical content as well as writing and presentation quality. **Submit this notebook and all supporting files as a zip archive on Sunlearn. Filename is: groupnr\_assignment3.zip, e.g. 3\_assignment3.zip - incorrectly named files may not be marked.** The notebook assignment1.ipynb in your submission will be converted to PDF during the marking process, and your submission should be able to be marked **based on the PDF alone**.  Thus ensure you include the output of code cells; *note that widgets in the notebook are not included during the export process, so **include images with representative output** and leave appropriate comments on your findings where relevant*.
#
# Clearly mark any of your group's code using SOLUTION_START and SOLUTION_END comments, and clearly attribute work by indicating which group member(s) worked on any particular portion of the submission. Work that is not explicitly attributed is assumed to be the joint work of all group members. Answer all questions and exercises in their corresponding blue blocks, and add blue blocks describing what you have done anywhere else you modify or perform additional work - the blue block and comments help us differentiate your work from the issued assignment, so if you fail to do this, some of your work may not be marked.
#
# You may not import any additional packages not available in NARGA or on the learner.cs.sun.ac.za service without prior approval, i.e. ensure your code runs in one of these environments.  If you use learner.cs.sun.ac.za, specify in the blue block above the configuration w.r.t. image and GPU that was used. 
#
# Lastly, note that the assignment is open-ended - further investigation of any aspect documented in the submission will be considered during marking.

# # 1. Gaussian Process Regression

# **Definition** - A Gaussian Process (GP) is a collection of random variables, any finite number of which have a joint Gaussian distribution.
#
# A GP is completely specified by its mean function and covariance function. We can define the mean function $m(\mathbf{x})$ and the covariance function $k(\mathbf{x}, \mathbf{x}^\prime)$ of a real process $f(\mathbf{x})$ where $\mathbf{x}\in\mathbb{R}^D$, as
#
# $$
# \begin{aligned}
# m(\mathbf{x}) &= \mathbb{E}[f(\mathbf{x})]\\
# k(\mathbf{x}, \mathbf{x}^\prime) &= \mathbb{E}[(f(\mathbf{x})-m(\mathbf{x}))(f(\mathbf{x}^\prime)-m(\mathbf{x}^\prime))]
# \end{aligned}
# $$
#
# and write the Gaussian process as
#
# $$
# f(\mathbf{x}) \sim GP(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}^\prime)).
# $$
#
# It is common practice to take the mean function to be zero, i.e. $m(\mathbf{x})=\mathbf{0}$

# #### The Regression Task
#
# Assume we have a dataset $\mathcal{D}$, consisting of $N$ observations, $\mathcal{D} = \{(\mathbf{x}^{(n)}, y^{(n)})|n=1,\ldots,N\}$, where $\mathbf{x}^{(n)}$ is a $D$-dimensional vector of covariates and $y^{(n)}$ is the (real-valued) target. Our task is to make inferences about the relationship between the input $X$ and the target $\mathbf{y}$. 

# **Task 1: A simple noise-free function**

X1 = np.linspace(0.05,0.95,8)
y1 = -np.cos(np.pi*X1) + np.sin(4*np.pi*X1)
plt.plot(X1,y1,' o')
plt.show()

# **Task 2: A Function with 2 Covariates and Additive Noise**

# +
# Ground truth: Branin Function (see: https://www.sfu.ca/~ssurjano/branin.html)
a, b, c, r, s, t = 1, 5.1/(4*pi**2), 5/pi, 6, 10, 1/(8*pi)
x1, x2 = np.meshgrid(np.linspace(-5,10,50), np.linspace(0,15,50))
branin = lambda a, b, c, r, s, t, x1, x2: a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
y = branin(a, b, c, r, s, t, x1, x2)
y = (y - np.mean(y))/np.std(y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x1, x2, y, cmap=cm.coolwarm)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$y$')
plt.show()

# +
# Given noisy observed points:
np.random.seed(0)
X2_1 = np.random.uniform(-5, 10, [70,1])
X2_2 = np.random.uniform(0, 15, [70,1])
X2 = np.concatenate((X2_1, X2_2), axis=1)
y2 = branin(a, b, c, r, s, t, X2_1, X2_2)
y2 = (y2 - np.mean(y2))/np.std(y2)
y2 = np.random.normal(y2, 0.1)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X2_1, X2_2, y2)
plt.show()


# -

# The GP definition above was given in terms of a _function-space view_ - the Gaussian process defines a distribution over functions, and inference takes place directly in the space of functions. For most of the assignment we will only consider this perspective on GPs. But one should note that there is an equivalent _weight-space view_, which initially might be easier to grasp. Section 2.1 of \[[1](#References)\] provides a thorough discussion on the weight-space view. Use this section to answer the following questions. 

# <div class="alert alert-block alert-info">
#     
# **Exercise 1.1** Complete the derivation of (2.7) in \[[1](#References)\]. That is, show that 
#
# $$
# p(\mathbf{w}|X,\mathbf{y}) \propto \exp\left(-\frac{1}{2}(\mathbf{w}-\bar{\mathbf{w}})^T(\frac{1}{\sigma^2_n} XX^T+\Sigma_p^{-1})(\mathbf{w}-\bar{\mathbf{w}})\right).
# $$
#     
# where $\bar{\mathbf{w}}=\frac{1}{\sigma^2_n}\left(\frac{1}{\sigma^2_n}XX^T+\Sigma_p^{-1}\right)^{-1}X\mathbf{y}$, and $\mathbf{w}$ is a vector of weights of a linear regression model.
#     
# **Answer**
# The posterior is given as 
# $$
# p(\mathbf{w}|X,\mathbf{y}) = \frac{p(\mathbf{y}|X,\mathbf{w})p(\mathbf{w})}{p(\mathbf{y}|X)}
# $$
# and since the normalising constant is not dependent on the weights we can write this proportionally as the likelihood multiplied by the prior:
# $$
# p(\mathbf{w}|X,\mathbf{y}) \propto p(\mathbf{y}|X,\mathbf{w})p(\mathbf{w}) \\
# \propto \frac{1}{(2\pi\sigma_n^2)^\frac{n}{2}}\exp \left({\frac{-1}{2\sigma_n^2}(\mathbf{y}-X^T\mathbf{w})^T (\mathbf{y}-X^T\mathbf{w})}\right) \exp \left(\frac{-1}{2}\mathbf{w}^T\Sigma_p^{-1}\mathbf{w}\right) \\
# \propto \exp \left({\frac{-1}{2\sigma_n^2}(\mathbf{y}-X^T\mathbf{w})^T (\mathbf{y}-X^T\mathbf{w})}-\frac{1}{2}\mathbf{w}^T\Sigma_p^{-1}\mathbf{w}\right) \\
# \propto \exp \left(\frac{-1}{2}\left(\frac{1}{\sigma_n^2}(\mathbf{y}-X^T\mathbf{w})^T (\mathbf{y}-X^T\mathbf{w})+\mathbf{w}^T\Sigma_p^{-1}\mathbf{w}\right)\right) \\
# \propto \exp \left(\frac{-1}{2}\left(\frac{1}{\sigma_n^2}(\mathbf{y}^T\mathbf{y} - \mathbf{y}^T X^T\mathbf{w} - \mathbf{w}^T X\mathbf{y} + \mathbf{w}^T X X^T\mathbf{w})+\mathbf{w}^T\Sigma_p^{-1}\mathbf{w}\right)\right) \\
# \propto \exp \left(\frac{-1}{2}\left(\frac{1}{\sigma_n^2}(\mathbf{y}^T\mathbf{y} - 2\mathbf{w}^T X\mathbf{y} + \mathbf{w}^T X X^T\mathbf{w})+\mathbf{w}^T\Sigma_p^{-1}\mathbf{w}\right)\right) \text{ (group $\mathbf{w}$ and $\mathbf{w}^T$)}\\
# \propto \exp \left(\frac{-1}{2}\left(\mathbf{w}^T \left(\frac{1}{\sigma_n^2}X X^T + \Sigma_p^{-1}\right) \mathbf{w} - \frac{2}{\sigma_n^2}\mathbf{w}^T X\mathbf{y} + \frac{1}{\sigma_n^2}\mathbf{y}^T\mathbf{y}\right)\right) \text{ ( we drop constant)}\\
# \propto \exp \left(\frac{-1}{2}\left(\mathbf{w}^T \left(\frac{1}{\sigma_n^2}X X^T + \Sigma_p^{-1}\right) \mathbf{w} - \frac{2}{\sigma_n^2}\mathbf{w}^T X\mathbf{y} \right)\right) \\
# $$
# The next step requires letting $A = \frac{1}{\sigma_n^2}X X^T + \Sigma_p^{-1}$ and $\bar{\mathbf{w}} = \frac{1}{\sigma_n^2} A^{-1} X\mathbf{y}$:
#
# $$
# \propto \exp \left(\frac{-1}{2}(\mathbf{w}^T A\mathbf{w}) - 2\mathbf{w}^T A \mathbf{\bar{w}} \right) \\
# \propto \exp\left(-\frac{1}{2}(\mathbf{w}-\bar{\mathbf{w}})^T A (\mathbf{w}-\bar{\mathbf{w}})\right) \\
# \propto \exp\left(-\frac{1}{2}(\mathbf{w}-\bar{\mathbf{w}})^T (\frac{1}{\sigma_n^2}X X^T + \Sigma_p^{-1}) (\mathbf{w}-\bar{\mathbf{w}})\right)
# $$
# </div>

# <div class="alert alert-block alert-info">
#     
# **Exercise 1.2** Motivate the results in (2.9) of \[[1](#References)\] in terms of the results about the mean, covariance matrix, and distribution of linear functions of (Gaussian) variables.
#     
# **Answer**
# TODO
# </div>

# ## 1.1. The Covariance Function
#
# For $\mathbf{x}, \mathbf{x}^\prime\in\mathcal{X}$, the function $k(\cdot)$ which performs the mapping 
# $$
# k(\mathbf{x},\mathbf{x}^\prime) : \mathbb{R}^D \times \mathbb{R}^D \rightarrow \mathbb{R}
# $$
# can in general be called a _kernel_. A real kernel is called symmetric if $k(\mathbf{x},\mathbf{x}^\prime) = k(\mathbf{x}^\prime,\mathbf{x})$. 
#
# A _covariance function_ $k(\cdot)$ specifies the covariance between pairs of random variables:
# $$
# \text{cov}(f(\mathbf{x}),f(\mathbf{x}^\prime)) = k(\mathbf{x},\mathbf{x}^\prime)
# $$
# Note that the covariance between the outputs $f(\mathbf{x})$ and $f(\mathbf{x}^\prime)$ is written as a function of the inputs $\mathbf{x}$ and $\mathbf{x}^\prime$.
#
# Given a sample $\mathbf{x}_1, \cdots, \mathbf{x}_N$, the matrix $K$ whose entries are $K_{ij} = k(\mathbf{x}_i,\mathbf{x}_j)$, is called the _Gram matrix_ of the kernel $k$ for the sample. If $k$ is a covariance function, we call the matrix $K$ the _covariance matrix_.  Furthermore, $k$ is a valid covariance function if the Gram matrix $K$ is symmetric and positive semi-definite for all possible samples.
#
# A popular covariance function is the squared exponential covariance function (a.k.a. a radial-basis function or an exponential quadratic covariance function):
#
# $$
# \text{cov}(f(\mathbf{x}),f(\mathbf{x}^\prime)) = k(\mathbf{x}, \mathbf{x}^\prime) = \sigma^2\exp\left\{-\frac{1}{2l^2}\left|\mathbf{x}-\mathbf{x}^\prime\right|^2\right\}
# $$
#
# where $\sigma^2$ and $l^2$ are known as the variance and length scale hyperparameters, respectively. We can note that for this function, the covariance between two variables approaches $\sigma^2$ as their corresponding inputs draw closer to each other and decreases the futher the inputs are away from each other in the input space.
#
# Below, we visualize the effect of the two hyperparameters for equally spaced inputs. We use the `RBF` and `ConstantKernel` functions provided by the `sklearn.gaussian_process.kernel` library. 

# +
def plot_kernel(kernel):   
    x1 = np.linspace(0,10)
    x2 = np.linspace(0,10)
    k = np.array([[kernel([[i]],[[j]])[0,0] for i in x1]  for j in x2])
    plt.imshow(k, vmin=0, vmax=1)
    plt.axis('off')
    plt.show()

@interact(sigma=(0.1,2.0),length_scale=(0.1,5.0))
def plot_se_kernel(sigma=1.0, length_scale=1.0):   

    kernel = gp.kernels.ConstantKernel(sigma**2) * gp.kernels.RBF(length_scale=length_scale)
    plot_kernel(kernel)


# -

# The `sklearn.gaussian_process.kernel` library provides a variety of other kernel functions, for example the Exp-Sine-Squared kernel (a.k.a. the periodic kernel):
#
# $$
# k(\mathbf{x}, \mathbf{x}^\prime) = \exp\left\{-\frac{2\sin^2\left(\frac{\pi \left|\mathbf{x}-\mathbf{x}^\prime\right|}{p}\right)}{l^2}\right\}
# $$ 
#
# where $l^2$ is the length scale and $p$ is the periodicity of the kernel.

@interact(periodicity=(0.1,5.0), length_scale=(0.1,5.0))
def plot_sine_kernel(periodicity=1.5, length_scale=1.0):   
    kernel = gp.kernels.ExpSineSquared(length_scale=length_scale, periodicity=periodicity)
    plot_kernel(kernel)


# ## 1.2. The GP Prior
#
# The covariance function implies a distribution over functions. To see this, we 
# first choose a number of input points, $X_* = \{\mathbf{x}_*^{(n)}|n=1,\ldots,N\}$. We then write out the corresponding covariance matrix by applying the kernel function to each pair of points, and draw samples from the resulting distribution:
#
# $$
# f_* = \mathcal{N}(\mathbf{0}, K(X_*,X_*)).
# $$
#
# where $K(X_*,X_*)_{n,m} = k(\mathbf{x}_*^{(n)},\mathbf{x}_*^{(m)})$. If the input points are drawn appropriately, the drawn samples will give us a good idea of the type of functions that will arise from a particular GP. 

# <div class="alert alert-block alert-info">
#     
# Implement the `plot_GP_prior` function below in order to reproduce the provided example plot. The function takes as arguments: `x`, the input points, `kernel`, a squared exponential covariance function and `num_samples`, the number of sampled functions to plot. Use `np.random.multivariate_normal` to draw samples.
#     
# </div>

def plot_GP_prior(X, kernel, num_samples):
    # TODO: Implement
    # SOLUTION_START
    samples = np.random.multivariate_normal(mean=np.zeros(X.shape[0]), cov=kernel(X), size=num_samples)
    confidence_interval = 1.96 * np.sqrt(np.diag(kernel(X)))
    plt.figure(figsize=(8, 4))
    for i in range(num_samples):
        plt.plot(X, samples[i], lw=1, linestyle='--', label=f'Sample {i+1}')
    plt.fill_between(X.flatten(), -confidence_interval, confidence_interval, color='gray', alpha=0.2)
    plt.axhline(0, color='black', lw=1,label='Mean' )
    plt.title("GP Prior")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()



# +
# Choose a number of input points - draw enough, so that when we 
# plot the corresponding function samples, they appear continuous
np.random.seed(2)
X = np.linspace(0,1).reshape(-1,1)

# Kernel
sigma = 1.0
length_scale = 0.1
kernel = gp.kernels.ConstantKernel(sigma**2)*gp.kernels.RBF(length_scale=length_scale)

plot_GP_prior(X, kernel, num_samples=3)
# -

Image('./figures/gp_prior_plot.png')

# ## 1.3. A Gaussian Process Regression Model
#
# Given a prior and some observations, we now want to derive the posterior predictive distribution for new input points. 
#
# First, we will use the provided `sklearn.gaussian_process.GaussianProcessRegressor` class to fit our data for regression task 1.

# +
# Note that the kernel hyperparameters will be optimized as well, unless 
# the bounds are marked as 'fixed'. We will consider how to optimize
# the hyperparameters in a later section.
kernel1 = gp.kernels.ConstantKernel(sigma**2, constant_value_bounds='fixed')*\
         gp.kernels.RBF(length_scale, length_scale_bounds='fixed')

model = gp.GaussianProcessRegressor(kernel=kernel1, normalize_y=True)
model.fit(X1.reshape(-1,1), y1)
print(model.kernel_.get_params())


# -

# <div class="alert alert-block alert-info">
#     
# Implement `plot_GP_posterior` below. Also complete the following code cell by using `plot_GP_posterior` along with the `predict` and `sample` functions from sklearn's GP library to reproduce the provided example plot.
#     
#    </div>

# +

def plot_GP_posterior(X, y, X_new, y_pred_mean, y_pred_std, samples):
    '''Plot samples from the posterior predictive distribution for new
    input data points.
    
    Parameters
    ----------
    X : numpy.ndarray
        Observed covariate values.
    y : numpy.ndarray
        Observed target values.
    X_new : numpy.ndarray (1D)
        New covariate values.
    y_pred_mean : numpy.ndarray (1D)
        Mean of posterior predictive distribution of target values given X,y,X_new.
    y_pred_std : numpy.ndarray (1D)
        Std of posterior predictive distribution of target values given X,y,X_new.
    samples : numpy.ndarray   
        New function samples from the posterior predictive distribution.
    '''
    # TODO: Implement
    # SOLUTION_START
    plt.figure(figsize=(8, 4))
    plt.plot(X.flatten(), y.flatten(), 'ro', label='Observed')
    
    plt.plot(X_new.flatten(), y_pred_mean, 'k-', label='Mean')
    plt.fill_between(
        X_new.flatten(),
        y_pred_mean - 1.96 * y_pred_std,
        y_pred_mean + 1.96 * y_pred_std,
        color='gray', alpha=0.2, label='95% CI'
    )
    
    for i in range(samples.shape[1]):
        plt.plot(X_new.flatten(), samples[:, i], lw=1, linestyle='--', label=f'Sample {i+1}')
    
    plt.title("GP Posterior")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

   

# +
# Choose a number of new input points - draw enough, so that when we 
# plot the corresponding predicted target values, they appear continuous
X_new = np.linspace(-0.2,1.2).reshape(-1,1)

# TODO: Predict and sample functions from sklearn's GP library, and plot results
y_pred_mean, y_pred_std = model.predict(X_new, return_std=True)
samples = model.sample_y(X_new, n_samples=3)
print(y_pred_mean.shape, y_pred_std.shape, samples.shape)
plot_GP_posterior(X1, y1, X_new, y_pred_mean, y_pred_std, samples)
# -

Image('./figures/gp_posterior_plot_sklearn.png')


# ### Implementing GPR
#
# #### Posterior Prediction Distribution given Noise-free Observations
#
# We will first consider the special case where we have access to noise-free data, $X$. That is, we have access to $\{(\mathbf{x}^{(n)},f^{(n)})|n=1,\ldots,N\}$. Given new test data $X_*$, we would like to predict the corresponding function values, $\mathbf{f}_*$. This corresponds to deriving the distribution: $p(\mathbf{f}_*| X_*, X, \mathbf{f})$.
#
# Following from the definition of a GP, and using the prior we defined earlier, we have that the joint distribution of $\mathbf{f}$ and $\mathbf{f}_*$ is given by:
#
# $$
# p\left(\left. \begin{bmatrix}\mathbf{f} \\ \mathbf{f}_*\end{bmatrix}\right|X,X_*\right)= \mathcal{N}\left(\mathbf{0},\begin{bmatrix}K(X,X) & K(X,X_*) \\ K(X_*,X) & K(X_*,X_*)\end{bmatrix}\right)
# $$
#
# The posterior of $\mathbf{f}_*$ given the observations is then:
#
# $$
# p(\mathbf{f}_*| X_*, X, \mathbf{f}) = \mathcal{N}(K(X_*,X)K(X,X)^{-1}\mathbf{f}, K(X_*,X_*)-K(X_*,X)K(X,X)^{-1}K(X,X_*)).
# $$

# <div class="alert alert-block alert-info">
#     
# **Exercise 1.3.1** Perform the conditioning required to derive the parameters of the posterior distribution $p(\mathbf{f}_*| X_*, X, \mathbf{f})$ given the joint Gaussian prior distribution $p\left(\left. \begin{bmatrix}\mathbf{f} \\ \mathbf{f}_*\end{bmatrix}\right|X,X_*\right)$ (See equations (2.18) and (2.19) and the surrounding discussion in \[[1](#References)\] for more detail).
#
#     
# **Answer**
# We map all the values from the posterior $\mathbf{f_*}$ to A.6 in \[[1](#References)\], ie match it to this equation:  $N(\mu_x, A), and x|y \sim N(\mu_x + CB^{−1}(y − \mu_y ),   A − CB^{−1}C^T)$
#
# For example let $\mu_x = \mathbf{0}$ and $(\mathbf{y} - \mu_y) = \mathbf{f}$ and then let C = K(X) .... etc
# </div>

# **Matrix Inversion Instability** Matrix inversion is known to be numerically unstable. To address this issue, we will instead use Cholesky factorization (see appendix A.4 of \[[1](#References)\]) to try and achieve numerically stable computation of the new covariance matrix. 

# <div class="alert alert-block alert-info">
#
# Use Algorithm 2.1 in \[[1](#References)\] to help you implement the `predict` function below using Cholesky factorization. Useful functions should be `np.linalg.cholesky` and `np.linalg.solve`. Note that Cholesky is not a silver bullet - one can still have instability issues when using it.
#
# </div>

#  <div class="alert alert-block alert-info">
#     
# **Exercise 1.3.2** Explain how the computations of Algorithm 2.1 \[[1](#References)\] yield the values in (2.25), (2.26) and (2.30).
#     
# **Answer**
#     
# </div>

# mean:
# $$K(X_*,X)K(X,X)^{-1}\mathbf{f}
# $$
#
# Covariance:
# $$\Sigma_* = K(X_*, X_*) - K(X_*, X) K(X, X)^{-1} K(X, X_*) \\
#
# p(\mathbf{f}_*| X_*, X, \mathbf{y}) = \mathcal{N}(K(X_*,X)[K(X,X)+\sigma_\epsilon^2I]^{-1}\mathbf{y}, K(X_*,X_*)-K(X_*,X)[K(X,X)+\sigma_\epsilon^2I]^{-1}K(X,X_*)).
#
# $$

# +

def predict(kernel, X_star, X, y, sigma_eps=0):
    '''Compute the mean and covariance matrix of the posterior
    predictive distribution given X, X* and y.
    
    Parameters:
    -----------
    X_star : numpy.ndarray
        New covariate values.
    X : numpy.ndarray
        Observed covariate values.
    y : numpy.ndarray
        Observed target values.
        
    Returns
    -------
    y_pred_mean : numpy.ndarray
        Mean of posterior predictive distribution of target values 
        given X, y and X_star.
    y_pred_cov : numpy.ndarray
        Covariance matrix of posterior predictive distribution of target 
        values given X, y and X_star.
    '''
    # TODO: Implement
    K = kernel(X, X)
    K_star = kernel(X, X_star)    
    K_star_star = kernel(X_star, X_star)
    
    L = np.linalg.cholesky(K + (sigma_eps ** 2) * np.eye(K.shape[0]))
    v_mean = np.linalg.solve(L, y)
    alpha = np.linalg.solve(L.T, v_mean)
    y_pred_mean = K_star.T @ alpha
    V_cov = np.linalg.solve(L, K_star)
    y_pred_cov = K_star_star - (V_cov.T @ V_cov)
    
    return y_pred_mean, y_pred_cov  


# -

# <div class="alert alert-block alert-info"> 
#     
# Use your `predict` function above and your own sampling mechanism to reproduce the plot below for regression task 1 (using `plot_GP_posterior`). Compare to the results obtained using `sklearn.gaussian_process.GaussianProcessRegressor` - the plots should look similar. 
#     
# </div>

# +
# TODO: Reproduce GP posterior plot
y_pred_mean, y_pred_cov = predict(kernel1, X_new, X1.reshape(-1,1), y1)
y_pred_std = np.sqrt(np.diag(y_pred_cov))

plt.figure(figsize=(8, 4))
plt.plot(X1.flatten(), y1.flatten(), 'ro', label='Observed')    
plt.plot(X_new, y_pred_mean, 'k-', label='Mean')
plt.fill_between(
    X_new.flatten(),
    y_pred_mean - 1.96 * y_pred_std,
    y_pred_mean + 1.96 * y_pred_std,
    color='gray', alpha=0.2, label='95% CI'
)
for i in range(3):
    sample = np.random.multivariate_normal(mean=y_pred_mean.flatten(), cov=y_pred_cov)
    plt.plot(X_new.flatten(), sample, lw=1, linestyle='--', label=f'Sample {i+1}')
plt.title("GP Posterior")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
# -

Image('./figures/gp_posterior_plot_self.png')

# #### Posterior Prediction Distribution given Noisy Observations
#
# It is seldom the case that we have access to noise-free data. Most of the time, we only observe a noisy version: $y = f(\mathbf{x}) + \epsilon$. We assume that our function values are corrupted by additive _i.i.d_ Gaussian nose $\epsilon$ with variance $\sigma_\epsilon^2$. The prior on these noisy observations then becomes
#
# $$
# cov(\mathbf{y}) = K(X, X) + \sigma_\epsilon^2I
# $$
#
# The joint distribution of the observed target values, $\mathbf{y}$, and the function values at the test locations, $\mathbf{f}_*$, now becomes:
#
# $$
# p\left(\left. \begin{bmatrix}\mathbf{y} \\ \mathbf{f}_*\end{bmatrix}\right|X,X_*\right)= \mathcal{N}\left(\mathbf{0},\begin{bmatrix}K(X,X) + \sigma_\epsilon^2I & K(X,X_*) \\ K(X_*,X) & K(X_*,X_*)\end{bmatrix}\right)
# $$
#
# and the new posterior is given by:
#
# $$
# p(\mathbf{f}_*| X_*, X, \mathbf{y}) = \mathcal{N}(K(X_*,X)[K(X,X)+\sigma_\epsilon^2I]^{-1}\mathbf{y}, K(X_*,X_*)-K(X_*,X)[K(X,X)+\sigma_\epsilon^2I]^{-1}K(X,X_*)).
# $$

# <div class="alert alert-block alert-info">
#
# Now modify your `predict` function above to take an additional keyword `sigma_eps` representing the standard deviation of the assumed noise level, with a default value of zero for noise-free prediction. This will allow you to use your prediction function in both the noise-free and noisy setting.
#     
# </div>

# To better observe the effect of noise on the predictive distribution, we first augment our dataset with more data points and add some noise to the target values.

# +
# Add some more data points and noise
np.random.seed(2)
X1_noise = np.linspace(0.05,0.95,25)
y1_noise = -np.cos(np.pi*X1_noise) + np.sin(4*np.pi*X1_noise) + np.random.normal(loc=0.0, scale=0.1, size=(25))

plt.scatter(X1_noise, y1_noise)
plt.show()


# -

# <div class="alert alert-block alert-info">
#
# Fit this noisy data using the noise-free approach. Then fit the model assuming the correct noise level (in this setting we know the noise level, but in general this will have to be inferred). Finally, fit the model assuming a higher level of noise than is actually present. Plot the results in each case, and discuss what you observe.
#     
# </div>

# +
# SOLUTION_START ADDITIONAL CODE
def plot(X1, y1, X_new, y_pred_mean, y_pred_std, y_pred_cov):
    plt.figure(figsize=(8, 4))
    plt.plot(X1.flatten(), y1.flatten(), 'ro', label='Observed')    
    plt.plot(X_new, y_pred_mean, 'k-', label='Mean')
    plt.fill_between(
        X_new.flatten(),
        y_pred_mean - 1.96 * y_pred_std,
        y_pred_mean + 1.96 * y_pred_std,
        color='gray', alpha=0.2, label='95% CI'
    )
    for i in range(3):
        sample = np.random.multivariate_normal(mean=y_pred_mean.flatten(), cov=y_pred_cov)
        plt.plot(X_new.flatten(), sample, lw=1, linestyle='--', label=f'Sample {i+1}')
    plt.title("GP Posterior")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
    
# SOLUTION_END ADDITIONAL CODE


# -

X_star = np.linspace(0.05,0.95).reshape(-1,1)

# TODO: Fit using noise-free approach
y_pred_mean, y_pred_cov = predict(kernel, X_star, X1.reshape(-1,1), y1, sigma_eps=0)
plot(X1.reshape(-1,1), y1, X_star, y_pred_mean, np.sqrt(np.diag(y_pred_cov)), y_pred_cov)


# TODO: Fit using correct noise level
y_pred_mean_noise, y_pred_cov_noise = predict(kernel, X_star, X1_noise.reshape(-1,1), y1_noise, sigma_eps=0.1)
plot(X1_noise.reshape(-1,1), y1_noise, X_star, y_pred_mean_noise, np.sqrt(np.diag(y_pred_cov_noise)), y_pred_cov_noise)


# TODO: Fit using higher level of noise than is actually present
y_pred_mean_noise_high, y_pred_cov_noise_high = predict(kernel, X_star, X1_noise.reshape(-1,1), y1_noise, sigma_eps=0.5)
plot(X1_noise, y1_noise, X_star, y_pred_mean_noise_high, np.sqrt(np.diag(y_pred_cov_noise_high)), y_pred_cov_noise_high)

# <div class="alert alert-block alert-info">
#
# **Discussion**
#
#     
# </div>

# ## 1.4. Tuning the Hyperparameters
#
# Most covariance functions have a number of free parameters. If chosen inappropriately, these hyperparameters can severely affect the quality of the fit. The plots below show the effect of varying the length scale and variance parameters of the squared exponential covariance function for regression task 1. For a further discussion see section 2.3 of \[[1](#References)\].

# #### Effect of varying the Kernel Parameters

# +
x = np.linspace(-0.2,1.2).reshape(-1,1)
fig, ax = plt.subplots(1, 3, figsize=(15,5))
for i, l in enumerate([0.05, 0.1, 0.5]):
    k = gp.kernels.ConstantKernel(1.0, constant_value_bounds='fixed')*\
        gp.kernels.RBF(l, length_scale_bounds='fixed')
    model = gp.GaussianProcessRegressor(kernel=k, normalize_y=True)
    model.fit(X1.reshape(-1,1), y1)
    y_mean, y_std = model.predict(x, return_std=True)
    ax[i].plot(x.reshape(-1), y_mean, color='gray')
    ax[i].fill_between(x.reshape(-1), y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.2, color='gray')
    ax[i].scatter(X1, y1, c='r')
    ax[i].set_title('{}'.format(l))
fig.suptitle('length scale = ')
plt.show()

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15,5))
for i, s in enumerate([0.1, 1.0, 1.5]):
    k = gp.kernels.ConstantKernel(s**2, constant_value_bounds='fixed')*\
        gp.kernels.RBF(0.1, length_scale_bounds='fixed')
    model = gp.GaussianProcessRegressor(kernel=k, normalize_y=True)
    model.fit(X1.reshape(-1,1), y1)
    y_mean, y_std = model.predict(x, return_std=True)
    ax[i].plot(x.reshape(-1), y_mean, color='gray')
    ax[i].fill_between(x.reshape(-1), y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.2, color='gray')
    ax[i].scatter(X1, y1, c='r')
    ax[i].set_title('{}'.format(s))
fig.suptitle('sigma = ')
plt.show()


# -

# #### Grid Search

# The marginal likelihood for a GP is given by (equation 2.30 of \[[1](#References)\])
#
# $$
# \log p(\mathbf{y}|X,\theta) = -\frac{1}{2}\mathbf{y}^T(K(X,X)+\sigma_\epsilon^2I)^{-1}\mathbf{y} -\frac{1}{2}\log\:\left|K(X,X)+\sigma_\epsilon^2I\right| - \frac{N}{2}\log 2\pi,
# $$
#
# where $\theta = (\sigma^2, l^2)$ are the hyperparameters of the covariance function $k$. We find suitable values for $\sigma^2$ and $l^2$ by performing a grid search over these values and identifying the setting that results in the maximum marginal likelihood.
#
# <div class="alert alert-block alert-info">
#
# Implement the `marginal_likelihood` function below and use it to find suitable values for $\sigma^2$ and $l^2$ by reproducing the example plot provided. Note that you will again have to make use of Cholesky decomposition when computing the likelihood.
#     
# </div>

# +
# SOLUTION_START
def marginal_likelihood(X, y, kernel, sigma_eps=0):
    # TODO: Implement
    K = kernel(X, X)
    L = np.linalg.cholesky(K + (sigma_eps ** 2) * np.eye(K.shape[0]))
    term1 = -0.5 * y.T @ np.linalg.solve(L.T, np.linalg.solve(L, y))
    term2 = -np.sum(np.log(np.diagonal(L)))
    term3 = -0.5 * X.shape[0] * np.log(2 * np.pi)
    log_marginal_likelihood = term1 + term2 + term3
    return log_marginal_likelihood

# SOLUTION_END


# +
# Perform a grid search over l in [0.05, 0.16] and sigma in [0.5, 2]
# to find suitable values for sigma^2 and l^2. Potentially useful
# numpy functions: meshgrid, argmax, unravel_index . Potentially
# useful matplotlib function: matplotlib.pyplot.pcolormesh ,
# matplotlib.pyplot.colorbar .
l_grid = np.linspace(0.05, 0.16, 100)
s_grid = np.linspace(0.5, 2, 100)
L_grid, S_grid = np.meshgrid(l_grid, s_grid)
results = np.zeros(S_grid.shape)

for i in trange(S_grid.shape[0]):
    for j in range(S_grid.shape[1]):
        l = L_grid[i,j]
        s = S_grid[i,j]
        k = gp.kernels.ConstantKernel(s**2, constant_value_bounds='fixed')*\
            gp.kernels.RBF(l, length_scale_bounds='fixed')
        results[i,j] = marginal_likelihood(X1.reshape(-1,1), y1, k, sigma_eps=0)

plt.figure(figsize=(8, 6))
plt.pcolormesh(S_grid, L_grid, results, shading='auto')
plt.colorbar()
plt.ylabel('l')
plt.xlabel(f'$\sigma$') 

max_idx = np.unravel_index(np.argmax(results), results.shape)
optimal_l = L_grid[max_idx]
optimal_s = S_grid[max_idx]
plt.scatter(optimal_s, optimal_l, color='red', label='Max log p(y)', marker='x')

plt.legend()
plt.show()
# -

Image('./figures/grid_search.png')

# Fit using tuned hyperparameters
sigma = optimal_s# TODO: provide tuned hyperparameter value
length_scale = optimal_l# TODO: provide tuned hyperparameter value
fig, ax = plt.subplots()
x = np.linspace(-0.2,1.2).reshape(-1,1)
kernel = gp.kernels.ConstantKernel(sigma**2, constant_value_bounds='fixed')*\
        gp.kernels.RBF(length_scale, length_scale_bounds='fixed')
model = gp.GaussianProcessRegressor(kernel=kernel, normalize_y=True)
model.fit(X1.reshape(-1,1), y1)
y_mean, y_std = model.predict(x, return_std=True)
ax.plot(x.reshape(-1), y_mean, color='gray')
ax.fill_between(x.reshape(-1), y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.2, color='gray')
ax.scatter(X1, y1, c='r')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$y$')
plt.show()

# A more advanced approach would be to perform gradient-based optimization. This accommodates higher dimensions of hyperparameters more easily, when a grid search would not be adequate.
#  
# For a discussion on model selection for GP regression see section 5.4 of \[[1](#References)\].

# <div class="alert alert-block alert-info">
#
# **Exercise 1.4.1 (Optional!)** Investigate using equation (5.9) of \[[1](#References)\] and gradient-based optimization to estimate the hyperparameters above. (Take note of footnote 5.)
#     
# </div>

# ## 1.5 Apply GPR to Regression Task 2

# <div class="alert alert-block alert-info">
#
# Apply a GP model to the 2D data set, and generate a plot like the example given. Use a grid search with your `marginal_likelihood` implementation to find good values for the variance and length scale hyperparameters of the kernel.
#     
# </div>

# +
# TODO: Tune hyperparameters over l in [2,4] and sigma in [1,2]
l_grid = np.linspace(2, 4, 100)
s_grid = np.linspace(1, 2, 100)
L_grid, S_grid = np.meshgrid(l_grid, s_grid)
results = np.zeros(S_grid.shape)

print(X2.shape, y2.shape)
for i in trange(S_grid.shape[0]):
    for j in range(S_grid.shape[1]):
        l = L_grid[i,j]
        s = S_grid[i,j]
        k = gp.kernels.ConstantKernel(s**2, constant_value_bounds='fixed')*\
            gp.kernels.RBF(l, length_scale_bounds='fixed')
        results[i,j] = marginal_likelihood(X2, y2, k, sigma_eps=0.1) # we know how much noise is present in the data, so we can use this value for sigma_eps

plt.figure(figsize=(8, 6))
plt.pcolormesh(S_grid, L_grid, results, shading='auto')
plt.colorbar()
plt.ylabel('l')
plt.xlabel(f'$\sigma$') 

max_idx = np.unravel_index(np.argmax(results), results.shape)
optimal_l = L_grid[max_idx]
optimal_s = S_grid[max_idx]
plt.scatter(optimal_s, optimal_l, color='red', label='Max log p(y)', marker='x')

plt.legend()
plt.show()
print(f'Optimal l: {optimal_l}, Optimal sigma: {optimal_s}')


# +
x1, x2 = np.meshgrid(np.linspace(-5,10,50), np.linspace(0,15,50))
X_star = np.concatenate((x1.flatten().reshape(-1,1),x2.flatten().reshape(-1,1)),axis=1)
a, b, c, r, s, t = 1, 5.1/(4*pi**2), 5/pi, 6, 10, 1/(8*pi)
y_true = branin(a, b, c, r, s, t, x1, x2)
y_true = (y_true - np.mean(y_true))/np.std(y_true)

# TODO: Apply GP model to 2D data set, and generate a plot like the one below.
# (The z label was troublesome for me to render due to matplotlib issues, don't
# worry if yours doesn't show up.)
model = gp.GaussianProcessRegressor(
        kernel=gp.kernels.ConstantKernel(optimal_s**2, constant_value_bounds='fixed') *
               gp.kernels.RBF(optimal_l, length_scale_bounds='fixed'),
        alpha=0.1**2,      
        normalize_y=False  
)
model.fit(X2, y2)
y_pred_mean, y_pred_std = model.predict(X_star, return_std=True)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
Z_pred = y_pred_mean.reshape(x1.shape)
surf = ax.plot_surface(x1, x2, Z_pred, cmap='coolwarm', alpha=0.7, edgecolor='none')
ax.scatter(X2[:, 0], X2[:, 1], y2.flatten(), c='tab:blue', marker='o', s=30, label='Observed Points')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$y$')
ax.view_init(elev=30, azim=-60)
plt.show()

# -

Image('./figures/2d_regression_fit.png')


# To better visualize how well our regression model fits the data, we will plot slices through the 3D space. That is, keep either $x_1$ or $x_2$ fixed, and plot $y$ as a function of the other. We can use an interactive slider and meshgrids `x1` and `x2` to plot various slices through the 3D space in a somewhat continuous manner.
#
# <div class="alert alert-block alert-info">
#     
# Complete both functions below, to slice through $x_1$ and $x_2$, respectively. Use the values in `x1` and `x2` as the intercepts for the slices. Note that each row in array `x1` corresponds to a constant value for `x2`. Thus, we can plot $y$ as a function of $x_1$ for a fixed value of $x_2$, by considering each row of `x1` in turn. Similarly, each column in `x2` corresponds to a constant value for `x1`. Example plots are provided.
#     
# </div>

@interact(x2_col=(0,x2.shape[0]-1))
def slice_x1(x2_col):   
    # TODO: Implement
    # SOLUTION_START
    x_val = x2[:, x2_col]
    x1_fixed = x1[0, x2_col]
    mu_slice = y_pred_mean.reshape(x1.shape)[:, x2_col]
    std_slice = y_pred_std.reshape(x1.shape)[:, x2_col]
    true_slice = y_true[:, x2_col]
    
    plt.figure(figsize=(6, 5))
    plt.ylim(-1, 4)
    plt.plot(x_val, mu_slice, color='gray', label='GP mean')
    plt.plot(x_val, true_slice, color='red', label='True')
    plt.fill_between(x_val, 
                     mu_slice - 1.96 * std_slice, 
                     mu_slice + 1.96 * std_slice, 
                     color='lightgray', alpha=0.3, label='95% CI')
    
    plt.title(f'$x_1 = {x1_fixed:.1f}$')
    plt.xlabel('$x_2$')
    plt.ylabel('$y$')
    plt.legend(loc='upper right')
    plt.show()


slice_x1(24)

Image('./figures/2d_regression_slice_x1.png')


@interact(x1_row=(0,x1.shape[0]-1))
def slice_x2(x1_row):   
    # TODO: Implement
    x_val = x1[x1_row, :]
    x2_fixed = x2[x1_row, 0]
    mu_slice = y_pred_mean.reshape(x1.shape)[x1_row, :]
    std_slice = y_pred_std.reshape(x1.shape)[x1_row, :]
    true_slice = y_true[x1_row, :]

    plt.figure(figsize=(7, 5))
    plt.ylim(-1.04, 4)
    plt.plot(x_val, mu_slice, color='gray', label='GP mean')
    plt.plot(x_val, true_slice, color='red', label='True')
    plt.fill_between(x_val, 
                     mu_slice - 1.96 * std_slice, 
                     mu_slice + 1.96 * std_slice, 
                     color='lightgray', alpha=0.3, label='95% CI')
    
    plt.title(f'$x_2 = {x2_fixed:.1f}$')
    plt.xlabel('$x_1$')
    plt.ylabel('$y$')
    plt.legend(loc='upper right')
    plt.show()


slice_x2(38)

Image('./figures/2d_regression_slice_x2.png')

# ## 1.6 Designing a Covariance Function

# <div class="alert alert-block alert-info">
#
# Look into alternative data sets where the kernel functions above are not appropriate for GP regression. Discuss a more suitable kernel function, illustrate how it works and what aspects of the data it improves modelling on. Constrast the results when using the above kernel with this alternative kernel. A helpful resource on choosing or designing new kernel functions can be found [here](https://www.cs.toronto.edu/~duvenaud/cookbook/).
#     
# </div>

# +
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, ConstantKernel as C

# Generate synthetic periodic data , a noisy sine wave
np.random.seed(42)
X_train = np.sort(np.random.uniform(0, 10, 40)).reshape(-1, 1)
X_train = np.delete(X_train, np.where((X_train > 4) & (X_train < 7))[0]).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, X_train.shape[0])
X_test = np.linspace(0, 15, 200).reshape(-1, 1)
y_true = np.sin(X_test).ravel()

# RBF Kernel
kernel_rbf = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

# Periodic Kernel 
kernel_periodic = C(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=2*np.pi,
                                                       length_scale_bounds=(1e-2, 1e2),
                                                       periodicity_bounds=(1e-1, 10))

gp_rbf = gp.GaussianProcessRegressor(kernel=kernel_rbf, alpha=0.1**2, n_restarts_optimizer=10)
gp_rbf.fit(X_train, y_train)

gp_periodic = gp.GaussianProcessRegressor(kernel=kernel_periodic, alpha=0.1**2, n_restarts_optimizer=10)
gp_periodic.fit(X_train, y_train)

y_pred_rbf, std_rbf = gp_rbf.predict(X_test, return_std=True)
y_pred_per, std_per = gp_periodic.predict(X_test, return_std=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

ax1.plot(X_test, y_true, label='True Function', linestyle='--')
ax1.scatter(X_train, y_train, label='Training Data', zorder=3)
ax1.plot(X_test, y_pred_rbf, label='RBF Mean')
ax1.fill_between(X_test.ravel(), y_pred_rbf - 1.96 * std_rbf, y_pred_rbf + 1.96 * std_rbf, alpha=0.2)
ax1.set_title(f'RBF Kernel (Fails to extrapolate and connect the gap)')
ax1.legend()

ax2.plot(X_test, y_true, label='True Function', linestyle='--')
ax2.scatter(X_train, y_train, label='Training Data', zorder=3)
ax2.plot(X_test, y_pred_per, label='Periodic Mean')
ax2.fill_between(X_test.ravel(), y_pred_per - 1.96 * std_per, y_pred_per + 1.96 * std_per, alpha=0.2)
ax2.set_title(f'Periodic Kernel (Successfully captures circular geometry)')
ax2.set_xlabel('X (e.g., Time or Angle)')
ax2.legend()

plt.tight_layout()
plt.show()


# -

# Discuss: TODO

# # 2. Dirichlet Processes for Infinite GMMs
# <!-- 
# - the infinite GMM belongs to the family of Dirichlet process mixtures 
# - The selection of the appropriate number of mixture components for a finite GMM is a issue. The likelihood of the data will be a maximum when the number of mixtures is equal to the number of training data points, which results in overfitting.
# - Bayesian methodology addresses the over-fitting problem by assigning a prior distribtuion over the number of mixtures.
# - BNP clustering addresses this problem by assuming that there is an infinite number of latent clusters, but that a finite number of them is used to generate the observed data
# - Under these assumptions, the posterior provides a distribution over the number of clusters, the assignment of data to clusters, and the parameters associated with each cluster
# - Furthermore, the predictive distribution, i.e., the distribution of the next data point, allows for new data to be assigned to a previously unseen cluster
# - The BNP approach finesses the problem of choosing the number of clusters by assuming that it is infinite, while specifying the prior over infinite groupings P(c) in such a way that it favors assigning data to a small number of groups
# - The Bayesian nonparametric mixture model, which is called a Chinese restaurant process mixture (or a Dirichlet process mixture)  -->

# A Dirichlet process mixture model has the following general form:
#
# $$
# \begin{aligned}
# y_n|\theta_n &\sim F(\theta_n) \\
# \theta_n|G &\sim G \\
# G &\sim \text{DP}(G_0, \alpha)
# \end{aligned}
# $$
#
# where the data $y_1, \ldots, y_N \in \mathbb{R}^D$ are independently and identically drawn from a mixture of distributions of the form $F(\theta)$, with the mixing distribution over $\theta$ being $G$. The prior for this mixing distribution is a Dirichlet process with concentration parameter $\alpha$ and base distribution $G_0$. The data $y_1, \ldots, y_N$ can be regarded as part of an infinite exchangeable sequence (as per de Finetti's theorem).

# <div class="alert alert-block alert-info">
#
# **Question 2.1** Explain how the exchangeability assumption is related to the i.i.d. assumption, and how de Finetti's theorem allows us to move between them.
#
# **Answer**
#
# </div>

# **Alternative Dirichlet Process Mixture Model Definition**
#
# An equivalent model can be obtained by taking the limit as $K$ goes to infinity of a finite mixture model with $K$ components and which has the following form:
#
# $$
# \begin{aligned}
# y_n|c_n,\phi &\sim F(\phi_{c_n}) \\
# c_n|\mathbf{p} &\sim \text{Discrete}(p_1,\ldots,p_K) \\
# \phi_k &\sim G_0 \\
# \mathbf{p} &\sim \text{Dir}\left(\frac{\alpha}{K}, \ldots,\frac{\alpha}{K}\right).
# \end{aligned}
# $$
#
# Here, $c_n$ indicates which latent mixture component is associated with observation $y_n$. For each $k=1,\ldots,K$, the parameters $\phi_k$ determines the distribution of observations from that component. The mixing proportions for the classes, $\mathbf{p} = (p_1,\ldots,p_K)$, have a symmetric Dirichlet prior with concentration parameter, $\frac{\alpha}{K}$, which approaches zero as $K\rightarrow \infty$. 
#
# For this assignment we will consider the infinite limit of the following GMM, which belongs to the above family of Dirichlet process mixtures for known values of the NIW parameters:
#
# $$
# \begin{aligned}
# y_n|c_n,\mu, \Sigma &\sim \mathcal{N}(\mu_{c_n}, \Sigma_{c_n}) \\
# c_n|\mathbf{p} &\sim \text{Discrete}(p_1,\ldots,p_K) \\
# \mu_k, \Sigma_k|\mu_0,\lambda_0,\nu_0, S_0 &\sim \text{NIW}(\mu_0,\lambda_0,\nu_0, S_0) \\
# \mathbf{p} &\sim \text{Dir}\left(\frac{\alpha}{K},\ldots,\frac{\alpha}{K}\right)
# \end{aligned}
# $$
#
# where $\text{NIW}(\cdot)$ is the Normal Inverse-Wishart distribution. 
#
# ![DPGMM](figures/DPGMM.png)

# **Normal Inverse-Wishart Distribution**
#
# If $\mu, \Sigma|\mu_0,\lambda_0,\nu_0, S_0 \sim NIW(\mu_0,\lambda_0,\nu_0, S_0)$, then:
#
# $$
# \begin{aligned}
# \mu|\mu_0,\lambda_0, \Sigma &\sim \mathcal{N}\left(\mu_0, \frac{1}{\lambda_0}\Sigma\right) \\
# \Sigma|S_0, \nu_0 &\sim W^{-1}(\nu_0, S_0)
# \end{aligned}
# $$
#
# where $W^{-1}(\cdot)$ is the [Inverse-Wishart distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invwishart.html). The NIW distribution is a conjugate prior for a multivariate normal distribution with unknown mean and covariance: Let $\mathbf{y}_i|\mu, \Sigma \sim \mathcal{N}_M(\mu, \Sigma)$, where the $N$ ($M$-dimensional) observations in $\mathbf{y} = \{\mathbf{y}_1, \ldots, \mathbf{y}_N\}$ are _i.i.d._. If $\mu, \Sigma \sim NIW(\mu_0,\lambda_0,\nu_0, S_0)$, then the posterior distribution of $\mu, \Sigma$ given this data is also a Normal Inverse-Wishart:
#
# $$
# \mu, \Sigma|\mathbf{y}, \mu_0,\lambda_0,\nu_0, S_0 \sim NIW(\mu_N, \lambda_N, \nu_N, S_N).
# $$
#
# Note that the parameterization is not standard, so one should be careful about the parameterization used in any given source. For example, in some cases $\beta$ is used instead of $\lambda_0$, in which case the covariance matrix of the multivariate normal is given by $\beta\Sigma$ instead of $\frac{1}{\lambda_0}\Sigma$.

# <div class="alert alert-block alert-info">
#     
# **Question 2.2** Give the formulae for the posterior parameters of the NIW in terms of the parameters of the prior, the data points $\mathbf{y}_i$, and the sample mean $\bar{\mathbf{y}}$. Also give a description of the interpretation of each NIW parameter. 
#
# **Answer**
#
#     
# </div>

# <div class="alert alert-block alert-info">
#
# Implement `posterior_params` for the `NIW` class below.
#     
# </div>

class NIW:
    def __init__(self, mu0, lambda0, nu0, S0):
        self.mu0 = mu0.squeeze()
        self.lambda0 = lambda0
        self.nu0 = nu0
        self.S0 = S0
        self.iw = invwishart(df=nu0, scale=S0)
        
    def sample(self):
        Sigma = self.iw.rvs(size=1)
        mu = np.random.multivariate_normal(self.mu0, Sigma/self.lambda0)
        return mu, Sigma
    
    
    def log_prob(self, mu, Sigma):
        return self.iw.logpdf(Sigma) + \
            multivariate_normal(self.mu0, Sigma/self.lambda0).logpdf(mu)
    
    
    def posterior_params(self, Y):
        '''Compute the posterior parameters of an NIW distribution
        given the observed N x D data matrix Y.
        '''
        # TODO: Implement
        return mu_n, lambda_n, nu_n, S_n


# ## 2.1 Sampling from an Infinite GMM

# We can expand the conditional distribution of $c_n$ below as a fraction and integrate over the mixing proportions $\mathbf{p}$ (note that the result follows from the conjugacy of the Dirichlet and multinomial distributions):
#
# $$
# \begin{aligned}
# p(c_n=k|\mathbf{c}_{1:n-1}) &= \frac{p(\mathbf{c}_{1:n-1}, c_n=k)}{p(\mathbf{c}_{1:n-1})} \\
# &= \frac{\int p(c_n=k,\mathbf{c}_{1:n-1}|\mathbf{p})p(\mathbf{p})\:d\mathbf{p}}{\int p(\mathbf{c}_{1:n-1}|\mathbf{p})p(\mathbf{p})\:d\mathbf{p}} \\
# &=  \frac{m_{n,k} + \frac{\alpha}{K}}{n-1 + \alpha}
# \end{aligned}
# $$
#
# where $m_{n,k}$ is the number of $c_j$ that equal $k$ for all $1\le j \le n-1$. Note that by using the chain rule, this gives us access to the prior $p(\mathbf{c})$ and allows us to sequentially sample the mixture component assignment of each new data point given all previous assignments,
#
# $$
# p(\mathbf{c}) = p(c_N|\mathbf{c}_{1:N-1})p(c_{N-1}|\mathbf{c}_{1:N-2})\cdots p(c_2|c_1)p(c_1).
# $$

# <div class="alert alert-block alert-info">
#
# **Question 2.1.1** Derive formula (2.6) in Neal's paper \[[2](#References)\]. That is, show how one obtains
#     
# $$
# \frac{\int p(c_n=k,\mathbf{c}_{1:n-1}|\mathbf{p})p(\mathbf{p})\:d\mathbf{p}}{\int p(\mathbf{c}_{1:n-1}|\mathbf{p})p(\mathbf{p})\:d\mathbf{p}} =  \frac{m_{n,k} + \frac{\alpha}{K}}{n-1 + \alpha}
# $$
#
# **Answer**
#
# </div>

# If we now let $K\rightarrow\infty$, these conditional probabilties reach the following limits:
#
# $$
# \begin{aligned}
# p(c_n = k|\mathbf{c}_{1:n-1}) &\rightarrow \frac{m_{n,k}}{n-1+\alpha} \\
# p(c_n \neq c_j \:\: \forall \:\: j < n|\mathbf{c}_{1:n-1}) &\rightarrow \frac{\alpha}{n-1+\alpha}
# \end{aligned}
# $$

# <div class="alert alert-block alert-info">
#
# **Question 2.1.2** Explain how the second formula above, $p(c_n \neq c_j \:\: \forall \:\: j < n|\mathbf{c}_{1:n-1}) \rightarrow \frac{\alpha}{n-1+\alpha}$, is derived.
#     
# **Answer** 
#     
#
# </div>

# Note that "since the $c_n$ are significant only in so far as they are or are not equal to other $c_j$, the above probabilities are all that are needed to define the model." \[[2](#References)\]
#
# This gives us the following sampling procedure. The first data point is assigned the first mixture component. Subsequent draws are assigned to either an existing component $k$ with probability $\frac{m_{n,k}}{n-1+\alpha}$ or to a new component not used before with probability $\frac{\alpha}{n-1+\alpha}$. We thus assume that there exists an infinite number of mixture components, of which we use only a finite number to describe the observed data. It is easy to see that the concentration parameter, $\alpha$, will have an effect on the number of mixture components used. This procedure is also known as the Chinese restaurant process.

# <div class="alert alert-block alert-info">
#
# Implement `sample_inf_gmm` below.
#     
# </div>

def sample_inf_gmm(alpha, mu0, lambda0, nu0, S0, N):
    '''Sample N datapoints from an infinite GMM with the given
    model specification.
    
    Parameters
    ----------
    alpha : float
        Concentration parameter.
    mu0 : numpy.ndarray
    lambda0 : float
    nu0 : int
    S0 : numpy.ndarray
        Hyperparameters of the NIW prior.
    N : int
        Number of data points to sample.
        
    Returns:
    --------
    k : int
        Number of components used.
    c : numpy.ndarray
        N component assignments.
    Y : numpy.ndarray
        N x D matrix of sampled data points.
    '''
    # TODO: Implement


# +
np.random.seed(4)
k, c, y = sample_inf_gmm(0.8, np.zeros(2), 0.1, 4, np.eye(2), 10000)

def plot_inf_gmm(N=10):
    global y, c
    yn = y[:N]
    cn = c[:N] 
    for i in range(len(set(cn))):
        plt.scatter(yn[cn==i,0],yn[cn==i,1], alpha=0.6)
    plt.xlim((np.min(y[:,0])-.5,np.max(y[:,0])+.5))
    plt.ylim((np.min(y[:,1])-.5,np.max(y[:,1])+.5))
    plt.legend(np.arange(k))
    plt.show()

w = interact(plot_inf_gmm, N=(0,1000,10))
# -

# <div class="alert alert-block alert-info">
#
# **Question 2.1.3** Explain the relationship between the Dirichlet distribution, a Dirichlet process, the Chinese restaurant process, and the stick-breaking construction.
#     
# **Answer**
#     
# </div>

# #### Effect of the Concentration Parameter $\alpha$
#
# <div class="alert alert-block alert-info">
#     
# Take samples of size 10000 for the following values of $\alpha$, and plot the results as above, using the same NIW parameters: $\alpha \in \{ 0.1 , 0.8, 1.4, 2.0 \}$.  Note how the number of clusters changes with $\alpha$.
# </div>

# TODO


# ## 2.2. Markov Chain Inference Methods for DP Mixture Models

# +
# The Data set
with open("../data/inf_gmm.json") as f:
    data = json.load(f)
k_true = data['k']
c_true = np.array(data['c'])
Y = np.array(data['y'])

plt.scatter(Y[:,0], Y[:,1])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()
# -

# True clusters
for i in range(len(set(c_true))):
    plt.scatter(Y[c_true==i,0],Y[c_true==i,1], alpha=0.6)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()


# Exact computation of posterior expectations for a Dirichlet process mixture model is infeasible when there are more than a few observations. Such expectations can, however, be estimated using Monte Carlo or variational inference methods.  In this assignment, we consider some Monte Carlo methods based on Gibbs sampling and will implement algorithms 1-3 of \[[2](#References)\].

# Below is the parent class for the different Gibbs samplers. Since some of the algorithms will require computing the marginal likelihood of the data, we can begin by implementing it here. Let $\theta$ be the cluster parameters as sampled from the prior $\text{NIW}(\mu_0,\lambda_0,\nu_0, S_0)$. The marginal likelihood is then given by (for derivation, see [here](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf): section 9):
#
# $$
# \begin{aligned}
# p(\mathbf{y}) &= \int p(\mathbf{y}|\theta)p(\theta)\:d\theta \\
# &= \frac{1}{\pi^{ND/2}} \frac{\Gamma_D(\nu_N/2)}{\Gamma_D(\nu_0/2)} \frac{|S_0|^{\nu_0/2}}{|S_N|^{\nu_N/2}} \left(\frac{\lambda_0}{\lambda_N}\right)^{D/2}
# \end{aligned}
# $$
#
# where
#
# $$
# \Gamma_D(\nu / 2) = \pi^{D(D-1)/4}\prod_{i=1}^D\Gamma\left(\frac{\nu+1-i}{2}\right)
# $$
#
# is the multivariate Gamma function.
#
# <div class="alert alert-block alert-info">
#     
# Complete `log_marginal_likelihood` below (Note that the log of the multivariate Gamma function has been imported from `scipy.special` as `gamma_d`).
#     
# </div>

# Parent class for Gibbs sampling algorithms
class GibbsSampler:
    
    def __init__(self, alpha, mu0, lambda0, nu0, S0):
        self.alpha = alpha
        self.niw = NIW(mu0, lambda0, nu0, S0)      
        self.D = mu0.shape[0]
    
    
    def log_marginal_likelihood(self, Y):
        if Y.ndim == 1:
            Y = Y.reshape(1,-1)
        N, D = Y.shape
        mu_n, lambda_n, nu_n, S_n = self.niw.posterior_params(Y)

        # log_marginal = # TODO 
        return log_marginal
    
    
    def log_likelihood(self, Y, c, phi):
        """Compute log p(Y|c,phi)"""
        log_lik = 0
        for n in range(Y.shape[0]):
            log_lik += multivariate_normal(mean=phi[c[n]][0],cov=phi[c[n]][1]).logpdf(Y[n])
        return log_lik
    
    
    def log_posterior(self, Y, c, phi):
        """Compute log p(c,phi|Y) = log p(Y|c,phi) + log p(c) + log p(phi) - log p(Y)"""
        N = Y.shape[0]

        # Compute log p(y|c,phi)
        log_p_y_given_c_phi = self.log_likelihood(Y, c, phi)

        # Compute log_p_c
        m = [(c[:n] == c[n]).sum() for n in range(N)]
        log_p_c = reduce(lambda a, b: a+b, [
            log(m[n]) - log(n+self.alpha) if m[n] > 0
            else log(self.alpha) - log(n+self.alpha)
            for n in range(N)
        ])

        # Compute log_p_phi
        log_p_phi = reduce(lambda a, b: a + b, [
            self.niw.log_prob(*phi[k]) if k in c
            else 0.0 for k in phi
        ])

        # Compute log p(y)
        log_p_y = self.log_marginal_likelihood(Y)

        return log_p_y_given_c_phi + log_p_c + log_p_phi - log_p_y


    def run(self, Y, num_iters=100):
        pass


# ### Algorithm 1

# Let the Markov chain consist of $\boldsymbol{\theta} = (\theta_1,\ldots,\theta_N)$ where $\theta_n = \{\mu_{c_n}, \Sigma_{c_n}\}$. Then, for $n=1,\ldots,N$, draw a new value from $\theta_n|\theta_{\lnot n},\mathbf{y}_n$ until convergence. Note that $\theta_n$ is independent of all other observations given $\theta_{\lnot n}$ and $\mathbf{y}_n$. Here $\theta_{\lnot n}$ denotes all $\theta_j$ for which $j\ne n$. The conditional distribution is given by the following mixture:
#
# $$
# p(\theta_n|\theta_{\lnot n},\mathbf{y}_n) = \sum_{j=1,j\neq n}^N q_{n,j}\cdot\delta_{\theta_j}(\theta_n) + q_0\cdot p(\theta_n|\mathbf{y}_n)
# $$
#
# where
#
# $$
# \begin{aligned}
# p(\theta_n|\mathbf{y}_n) &= NIW(\mu_n, \lambda_n, \nu_n, S_n) \quad \text{(conjugacy)} \\
# q_{n,j} &= b\cdot p(\mathbf{y}_n|\theta_j) = b\cdot \mathcal{N}(\mathbf{y}_n|\mu_{c_j}, \Sigma_{c_j}) \\
# q_0 &= b \alpha\cdot\int p(\mathbf{y}_n|\theta)p(\theta)\:d\theta = b\alpha \cdot p(\mathbf{y}_n)
# \end{aligned}
# $$
#
# Note that the parameters for the NIW distribution are just a single-observation update based on $\mathbf{y}_n$, and are not based on the first $n$ observations. The normalizing constant $b$ is chosen such that $\sum_j q_{n,j} + q_0 = 1$.
# Thus, one can draw samples from the conditional distribution using the following scheme:
#
# 1. Sample $\theta_n$ from one of the existing $\theta_{\lnot n}$, say $\theta_j$, with probability proportional to $m_{j}\cdot q_{n,j}$ where $j = 1,\ldots,N, j\neq n$ and $m_j = \sum_{i=1,i\neq n}^N \delta_{\theta_i}(\theta_j)$. (Here $m_j$ is the number of datapoints in $\mathbf{y}_{\lnot n}$ currently assigned to cluster $j$.)
#
# 2. Sample $\theta_n$ from the posterior distribution $p(\theta_n|\mathbf{y}_n)$ with probability proportional to $q_0$.

# <div class="alert alert-block alert-info">
#
# **Question 2.2.1** Derive formula (3.2) in Neal's paper \[[2](#References)\]. That is, derive
#     
# $$
# p(\theta_n|\theta_{\lnot n},\mathbf{y}_n) = \sum_{j=1,j\neq n}^N q_{n,j}\cdot\delta_{\theta_j}(\theta_n) + q_0\cdot p(\theta_n|\mathbf{y}_n)
# $$
#
#
# **Answer**
#
# </div>

# <div class="alert alert-block alert-info">
#
# Complete `sample_theta` for the `GibbsSampler1` class below.
#     
# </div>

class GibbsSampler1(GibbsSampler):
    
    def __init__(self, alpha, mu0, lambda0, nu0, S0):
        super(GibbsSampler1, self).__init__(alpha, mu0, lambda0, nu0, S0)

    
    def init_sampling(self, Y):
        N = Y.shape[0]
        # c denotes implicit mixture assignment through the data point's
        # assigned mean and covariance. Initially each data point is assigned 
        # its own mixture.
        c = np.arange(N)
        theta = {}
        self.counts = {}
        for c_n in c:
            mean, cov = self.niw.sample()
            theta[c_n] = [mean, cov]
            self.counts[c_n] = 1
            
        # Marginal likelihoods
        self.marginals = [exp(self.log_marginal_likelihood(Y[n])) for n in range(N)]
        
        # Posterior NIW
        self.posterior_NIW = [NIW(*self.niw.posterior_params(Y[n])) for n in range(N)]
        
        return c, theta
    

    def sample_theta(self, Y, c, theta, n):
        # Existing mixtures
        js = list(theta.keys())
        
        # Probabilities of choosing existing/new mixture
        p = np.zeros(len(js)+1)
        
        for i, j in enumerate(js):
            # p[i] = # TODO: Compute (unnormalized) prob of choosing existing mixture j
         
        # p[-1] = # TODO: Compute (unnormalized) prob of choosing new mixture
        
        choice = np.random.choice(len(js)+1, p=p/np.sum(p))
        if choice < len(js):
            c_n = js[choice]
            theta_n = theta[c_n]
            self.counts[c_n] += 1
        else:
            c_n = max(js) + 1
            theta_n = self.posterior_NIW[n].sample()
            self.counts[c_n] = 1
            
        return c_n, theta_n

    
    def sample_params(self, Y, c, theta):
        N = Y.shape[0]
        for n in range(N): 

            self.counts[c[n]] -= 1
            
            # Remove theta_{c_n} for current c_n if c_not_n 
            # does not contain c_n (i.e. theta_{c_n} cannot be sampled)    
            if self.counts[c[n]] == 0:
                theta.pop(c[n]) 
                self.counts.pop(c[n])

            # Sample new theta_n
            c_n, theta_n = self.sample_theta(Y, c, theta, n)
            c[n] = c_n
            theta[c_n] = theta_n
        
        return c, theta
    
    
    def run(self, Y, num_iter=100):
        c, theta = self.init_sampling(Y)
        theta_samples = [None]*num_iter
        c_samples = [None]*num_iter
        log_posterior = [None]*(num_iter)
        
        iters = trange(num_iter, mininterval=1)
        for i in iters:
            c, theta = self.sample_params(Y, c, theta)
            c_samples[i] = copy.deepcopy(c)
            theta_samples[i] = copy.deepcopy(theta)
            log_posterior[i] = self.log_posterior(Y, c, theta)
            iters.set_description('log p(c,phi|Y): {:.4f}'.format(log_posterior[i]), refresh=False)
            
        return c_samples, theta_samples, log_posterior

np.random.seed(1)
gibbs1 = GibbsSampler1(alpha=0.1, mu0=np.zeros(2), lambda0=0.1, nu0=4, S0=np.eye(2))
c1_samples, theta1_samples, log_posterior1 = gibbs1.run(Y, num_iter=200)

plt.plot(log_posterior1)
plt.xlabel('Iteration')
plt.ylabel(r'$\log \: p(c, \phi|Y)$')
plt.show()

# Check evolution of the number of clusters used
num_clusters = [len(theta) for theta in theta1_samples]
plt.plot(num_clusters)
plt.ylabel('Number of components')
plt.xlabel('Iteration')
plt.show()

# +
# Plot cluster assignments and confidence ellipses
c = c1_samples[-1]
theta = theta1_samples[-1]
K = len(theta)

fig, ax = plt.subplots()
cmap = plt.get_cmap('tab20', K)
for i, k in enumerate(theta.keys()):
    d = Y[c==k].reshape(-1,2)
    ax.scatter(d[:,0], d[:,1], color=cmap(i), alpha=0.5)
for i, k in enumerate(theta.keys()):
    ax.scatter(theta[k][0][0], theta[k][0][1], color=cmap(i), marker='x', s=100,linewidths=2.0)
    confidence_ellipse(theta[k][1], theta[k][0], ax, n_std=2.0, edgecolor=cmap(i))
plt.show()


# -

# ### Algorithm 2

# Let the state of the Markov chain consist of $\mathbf{c}=(c_1, \ldots, c_N)$ and $\phi=(\phi_k : k \in \{c_1, \ldots,c_N\})$. Then, repeatedly sample as follows:
#
# 1. For $n=1,\ldots,N$: 
#     - If the present value of $c_n$ is associated with no other observation, remove $\phi_{c_n}$ from the state. 
#     - Draw a new value for $c_n$ from $c_n|\mathbf{c}_{\lnot n}, \mathbf{y}_n, \phi$, where:
# $$
# \begin{aligned}
# \text{If }k=c_j\text{ for some }j \neq n: p(c_n=k|\mathbf{c}_{\lnot n}, \mathbf{y}_n, \phi) &\propto \frac{m_{\lnot n ,k}}{N-1+\alpha}p(\mathbf{y}_n|\phi_k) \\
# p(c_n\neq c_j \:\forall\: j \neq n|\mathbf{c}_{\lnot n}, \mathbf{y}_n, \phi) &\propto \frac{\alpha}{N-1+\alpha} p(\mathbf{y}_n)
# \end{aligned}
# $$
# Here $m_{\lnot n ,k}$ is the number of $c_j$ that equal $k$ for all $j\ne n$.
#
# - If the new $c_n$ is not associated with any other observation, draw a value for $\phi_{c_n}$ from $p(\phi_{c_n}|\mathbf{y}_n)$ and add it to the state.
#     
# 2. For all $k \in \{c_1, \ldots,c_N\}$:
#     - Draw a new value from $p(\phi_k|\text{all }\mathbf{y}_n \text{ for which }c_n = k)$

# <div class="alert alert-block alert-info">
#     
# Implement `sample_c` and `sample_phi` for `GibbsSampler2` and plot the results as for algorithm 1.
#     
# </div>

class GibbsSampler2(GibbsSampler):
    
    def __init__(self, alpha, mu0, lambda0, nu0, S0):
        super(GibbsSampler2, self).__init__(alpha, mu0, lambda0, nu0, S0)
   
    
    def init_sampling(self, N):  
        c = np.arange(N)
        phi = {}
        for c_n in c:
            mean, cov = self.niw.sample()
            phi[c_n] = [mean, cov]
        return c, phi
    
    
    def sample_c(self, c_not_n, y_n, phi):
        # TODO: Implement
    
    
    def sample_phi(self, Y, c, c_k):
        # TODO: Implement
        
    
    def sample_params(self, Y, c, phi):        
        N = Y.shape[0]
        # 1. sample c
        for n in range(N):
            y_n = Y[n]
            c_not_n = c[1:]
            
            # Remove phi_{c_n} for current c_n if c_not_n 
            # does not contain c_n
            if (c_not_n == c[0]).sum() == 0:
                phi.pop(c[0])            
            
            # Sample new c_n
            c_n = self.sample_c(c_not_n, y_n, phi)
            c[0] = c_n
        
            c = np.roll(c, -1, axis=0)
        
        # 2. sample phi_c
        C = phi.keys()
        for c_k in C:
            mean, cov = self.sample_phi(Y, c, c_k)
            phi[c_k] = [mean, cov]

        return c, phi
    
    
    def run(self, Y, num_iter=100):
        N, D = Y.shape
        c, phi = self.init_sampling(N)
        phi_samples = [None]*num_iter
        c_samples = [None]*num_iter
        log_posterior = [None]*(num_iter)
        
        iters = trange(num_iter, mininterval=1)
        for i in iters:
            c, phi = self.sample_params(Y, c, phi)
            c_samples[i] = copy.deepcopy(c)
            phi_samples[i] = copy.deepcopy(phi)
            log_posterior[i] = self.log_posterior(Y, c, phi)
            iters.set_description('Log p(c,phi|Y): {:.4f}'.format(log_posterior[i]), refresh=False)
            
        return c_samples, phi_samples, log_posterior

gibbs2 = GibbsSampler2(alpha=0.1, mu0=np.zeros(2), lambda0=0.1, nu0=4, S0=np.eye(2))
c2_samples, phi2_samples, log_posterior2 = gibbs2.run(Y, num_iter=200)

# +
# TODO: Plot log-posterior


# +
# TODO: Check evolution of the number of clusters used
# -


# TODO: Plot cluster assignments and confidence ellipses


# ### Algorithm 3 - Collapsed Gibbs Sampling

# Since the NIW prior is conjugate to the multivariate Gaussian likelihood, we can analytically integrate out $\phi$. This elimination of the mixture component parameters from the algorithm means we have to sample fewer parameters at each iteration. This is known as collapsed Gibbs sampling or _Rao-Blackwellization_. 
#
# Let the state of the Markov chain consist of $\mathbf{c} = (c_1,\ldots,c_N)$. Repeatedly sample as follows:
#
# * For $n=1,\ldots,N$: Draw a new value from $c_n|\mathbf{c}_{\lnot n}, \mathbf{y}_n$, where:
#
# $$
# \begin{aligned}
# \text{If }k=c_j\text{ for some }j \neq n: p(c_n=k|\mathbf{c}_{\lnot n}, \mathbf{y}_n) &\propto \frac{m_{\lnot n ,k}}{N-1+\alpha}p_k(\mathbf{y}_{n}|\mathbf{y}_{\lnot n,k})\\
# p(c_n\neq c_j \:\forall\: j \neq n|\mathbf{c}_{\lnot n}, \mathbf{y}_n) &\propto \frac{\alpha}{N-1+\alpha} p(\mathbf{y}_n)
# \end{aligned}
# $$
#
# $p_k(\mathbf{y}_n|\mathbf{y}_{\lnot n,k})$ above denotes the posterior predictive distribution for $\mathbf{y}_n$ resulting if it were to be assigned to class $k$, which in this case is a multivariate $t$-distribution with the following parameters:
#
# $$
# p_k(\mathbf{y}_n|\mathbf{y}_{\lnot n,k}) = t_{\nu_{N_{\lnot n,k}}-D+1}\left(\mathbf{y}_n;\mu_{N_{\lnot n,k}}, \frac{S_{N_{\lnot n,k}}(\lambda_{N_{\lnot n,k}}+1)}{\lambda_{N_{\lnot n,k}}(\nu_{N_{\lnot n,k}}-D+1)}\right)
# $$
#
#
# Here the subscript $N_{\lnot n,k}$ for an NIW parameter refers to the posterior NIW parameter after conditioning on observations $\mathbf{y}_{\lnot n,k}=\{\mathbf{y}_j|j\neq n \text{ and }c_j = k\}$, and $D$ is the dimensionality of the data. (This is a multivariate generalisation of the case described [here](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Bayesian_inference) on Wikipedia.)  

# #### Retrieving $\phi_c$
#
# We can use a point estimate of $\phi_k$ based on the current state of $\mathbf{c}$ in the Markov chain. We know that the posterior distribution of $\phi_k$ given $\mathbf{y}_{\cdot,k} =\{\mathbf{y}_j|c_j = k\}$ is Normal Inverse-Wishart. The expectations are thus as follows, using $N_{\cdot,k}$ to indicate posterior NIW parameters after observing $\mathbf{y}_{\cdot,k}$ similar to above:
#
# $$
# \begin{aligned}
# \mathbb{E}[\mu_c] &= \mu_{N_{\cdot,c}} \\
# \mathbb{E}[\Sigma_c] &= \frac{S_{N_{\cdot,c}}}{\nu_{N_{\cdot,c}} - D -1}
# \end{aligned}
# $$

# <div class="alert alert-block alert-info">
#
# Complete `sample_c`, `sample_params` and `expected_phi` for `GibbsSampler3` below. And again produce plots similar to the above.
#     
# </div>

class GibbsSampler3(GibbsSampler):
    
    def __init__(self, alpha, mu0, lambda0, nu0, S0):
        super(GibbsSampler3, self).__init__(alpha, mu0, lambda0, nu0, S0)
   
    
    def init_sampling(self, N):  
        c = np.arange(N)
        return c
    
    
    def sample_c(self, c_not_n, y_n, y_not_n):
        # TODO: Implement
        
    
    def sample_params(self, Y, c):  
        # TODO: Implement
        
    
    def expected_phi(self, Y, c):
        # TODO: Implement
    
    
    def run(self, Y, num_iter=100):
        N, D = Y.shape
        c = self.init_sampling(N)
        c_samples = [None]*num_iter
        log_posterior = [None]*(num_iter)
        
        log_posterior[0] = self.log_posterior(Y, c, self.expected_phi(Y, c))
        iters = trange(num_iter, mininterval=1)
        for i in iters:
            c = self.sample_params(Y, c)
            c_samples[i] = copy.deepcopy(c)
            log_posterior[i] = self.log_posterior(Y, c, self.expected_phi(Y, c))
            iters.set_description('Log p(c,phi|Y): {:.4f}'.format(log_posterior[i]), refresh=False)
            
        return c_samples, log_posterior

gibbs3 = GibbsSampler3(alpha=0.1, mu0=np.zeros(2), lambda0=0.1, nu0=4, S0=np.eye(2))
c3_samples, log_posterior3 = gibbs3.run(Y, num_iter=200)

# +
# TODO: Plot log-posterior


# +
# TODO: Check evolution of the number of clusters used
# -


# TODO: Plot cluster assignments and confidence ellipses


# <div class="alert alert-block alert-info">
#
# **Question 2.2.2** Summarize in your own words the difference between the 3 algorithms above, and how these differences impact the results.  Focus on how the later algorithms address drawbacks in the earlier algorithms, and any differences in their applicability.
#     
# **Answer**
#     
# </div>

# <div class="alert alert-block alert-info">
#
# **Question 2.2.3** In some of the algorithms above, the fit exhibited quite a large number of clusters compared to the true number of clusters (5).  This seems like evidence of overfitting.  Discuss.  
#     
# **Answer** 
#
# </div>

# # References
#
# \[1\] Rasmussen, C.E. and Williams, C.K.I. (2006). _Gaussian Processes for Machine Learning_.  MIT Press: Cambridge, USA. http://www.gaussianprocess.org/gpml/chapters/RW.pdf
#
# \[2\] Neal, R.M. (2000). Markov Chain Sampling Methods for Dirichlet Process Mixture Models. _Journal of Computational and Graphical Statistics_, 9(2): 249-265. http://www.stat.columbia.edu/npbayes/papers/neal_sampling.pdf
