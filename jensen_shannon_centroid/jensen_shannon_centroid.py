import numpy as np
from typing import List, Union
import logging


def delta_f(theta):
    """
    Gradient of the negentropy w.r.t. the natural parameter \\theta (equation 96 from Nielsen, 2020).
    :param theta:
    :return:
    """
    norm = np.clip(1 - theta.sum(-1, keepdims=True), a_min=1e-8, a_max=None)
    return np.log(theta / norm)

def delta_f_inv(eta):
    """
    Inverse gradient of the negentropy (equation 97 from Nielsen, 2020).
    :param eta:
    :return:
    """
    norm = 1 + np.exp(eta).sum(-1, keepdims=True)
    return np.exp(eta) / norm

def calculate_jsc(
        distributions: Union[List, np.ndarray],
        T: int=1000,
        eps: float=1e-10) -> np.ndarray:
    """
    Calculate the Jensen-Shannon Centroid of a set of categorical distributions. The Jensen-Shannon Centroid is the
    minimizer Q of the equation:
    \\mathcal{L}(Q) = \\sum_{m} \\text{JS}(Q\|p_{m})
    where JS is the Jensen-Shannon divergence. We follow the ConCave–Convex procedure (CCCP) from Nielsen, 2020 to find
    Q.
    Nielsen, Frank. 2020. "On a Generalization of the Jensen–Shannon Divergence and the Jensen–Shannon Centroid" Entropy 22, no. 2: 221. https://doi.org/10.3390/e22020221
    :param distributions: An array of size MxNxK, where K is the number of classes, N is a list of distributions,
    and M is the set for which the centroid will be calculated.
    :param T: The maximum number of optimization steps.
    :param eps: Minimum difference between distributions at t and t + 1 needed for convergence
    :return: An array of size NxK, which is the Jensen-Shannon centroid between the M distributions for each of the N ensembles.
    """
    dists = [np.array(dist) for dist in distributions]
    assert len(np.array(distributions).shape) == 3, f"Shape of distributions should be 3, found {len(np.array(distributions).shape)}"
    N = dists[0].shape[0]
    K = dists[0].shape[1]
    assert all(d.shape[0] == N for d in dists), "Dimension 2 of distributions should be equal for all distributions"
    assert all(d.shape[1] == K for d in dists), "Dimension 3 of distributions should be equal for all distributions"

    # convert to natural parameters
    natural_dists = [dist[:,:-1] for dist in dists]
    theta = np.stack(natural_dists).mean(0)
    #theta[:,0] = 1 - theta[:,1:].sum(-1)
    converged = False
    for t in range(T):
        dfs = np.stack([delta_f(np.stack([theta, dist]).mean(0)) for dist in natural_dists]).mean(0)
        theta_new = delta_f_inv(dfs)
        # Stop if there's no significant difference
        if np.abs(theta - theta_new).sum() < eps:
            logging.debug(f"Jensen-Shannon centroid converged after {t} iterations")
            converged = True
            break
        theta = theta_new

    assert converged, f"Couldn't converge after {T} iterations!"
    return np.hstack([theta, 1 - theta.sum(-1,keepdims=True)])