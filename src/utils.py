# +
import numpy as np
import matplotlib.transforms as transforms

from matplotlib.patches import Ellipse
from scipy.stats import invwishart
from scipy.special import gamma
from numpy.random import multivariate_normal
from math import pi


# This code is slightly modified from the matplotlib example at
# https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

def confidence_ellipse(cov, mu, ax, n_std=2.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
