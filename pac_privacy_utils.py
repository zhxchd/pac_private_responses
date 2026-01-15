import numpy as np
from scipy.optimize import bisect
from scipy.stats import norm

def get_noise_components(p, support, b, use_svd=True):
    _, d = support.shape
    mu = np.sum(support.T * p, axis=1) # d
    centered = support - mu # m x d
    Y = centered * np.sqrt(p)[:, None]

    if use_svd:
        _, S, Vt = np.linalg.svd(Y, full_matrices=True)
        if len(S) < d:
            S = np.pad(S, (0, d - len(S))) # this is already after sqrt
        U = Vt.T
        # verify decomposition: U @ diag(S**2) @ U.T == cov
        # cov = (centered.T * p) @ centered  # d x d
        # assert np.allclose(cov, U @ np.diag(S**2) @ U.T)
        # assert np.allclose(U.T @ U, np.eye(U.shape[0]))
    else:
        U = None # no rotation, save memory for I_d
        S = np.sum(Y**2, axis=0)
        S = np.sqrt(S)

    noise_lambda = S * (S.sum()) / 2 / b
    noise_lambda = np.clip(noise_lambda, 1e-10, None)
    return U, noise_lambda

def update_p(p, support, noisy_result, noise_U, noise_lambda):
    diff = support - noisy_result  # Shape (m, d)
    rotated_diff = diff @ noise_U if noise_U is not None else diff
    mahalanobis = np.sum((rotated_diff ** 2) / noise_lambda, axis=1)
    log_likelihoods = -0.5 * mahalanobis
    log_p = np.log(p + 1e-300) + log_likelihoods
    c = log_p.max()
    log_sum = c + np.log(np.sum(np.exp(log_p - c)))
    p = np.exp(log_p - log_sum)
    return p

# returns the posterior success upper bound we can guarantee given an MI bound
def posterior_success_guarantee(mi_bound, prior_success=0.5):
    assert mi_bound > 0
    def f(x):
        return x * np.log(x/prior_success) + (1 - x) * np.log((1 - x)/(1-prior_success)) - mi_bound
    if f(1.0-1e-10) < 0:
        return 1.0
    return bisect(f, 0.5, 1.0-1e-10)

# returns the MI bound we need to guarantee a certain posterior success upper bound
def mi_bound(posterior_success, prior_success=0.5):
    return posterior_success * np.log(posterior_success/prior_success) + (1 - posterior_success) * np.log((1 - posterior_success)/(1-prior_success))

def posterior_success_rate_to_epsilon(psr, delta=1e-5):
    return (np.log((1 - delta) / (1 - psr) - 1)).item()

def epsilon_to_posterior_success_rate(epsilon, delta=1e-5):
    return 1 - (1 - delta) / (1 + np.exp(epsilon)).item()

def is_confident(noisy_result, noise_U, noise_lambda, alpha=0.05):
    d = noisy_result.shape[0]
    tilde_y = np.argmax(noisy_result)
    theta0s = np.eye(d)[np.arange(d) != tilde_y] # d-1 by d
    theta1 = np.zeros(d)
    theta1[tilde_y] = 1.0
    X = noisy_result
    assert np.all(noise_lambda > 0)

    z = norm.ppf(1 - alpha)

    diff1 = theta1 - theta0s # (d-1) by d
    diff2 = X - theta0s # (d-1) by d
    rotated_diff1 = diff1 @ noise_U if noise_U is not None else diff1
    rotated_diff2 = diff2 @ noise_U if noise_U is not None else diff2
    a = np.sum((rotated_diff1 * rotated_diff2) / noise_lambda, axis=1) # d-1
    b = np.sqrt(np.sum((rotated_diff1 ** 2) / noise_lambda, axis=1)) # d-1
    if np.any(a / b < z):
        return False # if for any test we cannot reject, we are not confident

    return True # all tests rejected, we are confident
