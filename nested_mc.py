import numpy as np
from numba import njit
from math import erf, sqrt


@njit
def gaussian_cdf(x):
    return 0.5*(1+erf(x / sqrt(2)))


@njit
def _estimation_uniform_njit(m, n, simulate_scenario, simulate_inner_loss, loss_threshold):
    scenarios = simulate_scenario(n)
    inner_losses_mean = np.zeros(n)

    for i, scenario in enumerate(scenarios):
        inner_loss = simulate_inner_loss(scenario=scenario, size=m)
        inner_losses_mean[i] = np.mean(inner_loss)

    estimator = np.mean(inner_losses_mean >= loss_threshold)
    return estimator


@njit
def _estimation_sequential_njit(m_init, m_average, n, simulate_scenario, simulate_inner_loss, loss_threshold, inner_loss_std):
    scenarios = simulate_scenario(n)
    sigmas = inner_loss_std(scenarios)
    m_array = np.zeros(n)
    inner_losses_mean = np.zeros(n)

    for i, scenario in enumerate(scenarios):
        inner_loss = simulate_inner_loss(scenario=scenario, size=m_init)
        inner_losses_mean[i] = np.mean(inner_loss)
    m_array += m_init

    while np.sum(m_array) < m_average * n:
        idx_scenario = np.argmin(m_array * np.abs(inner_losses_mean - loss_threshold) / sigmas)
        new_inner_loss = simulate_inner_loss(scenario=scenarios[idx_scenario], size=1)[0]
        inner_losses_mean[idx_scenario] = m_array[idx_scenario] / (m_array[idx_scenario] + 1) * inner_losses_mean[idx_scenario] + \
            new_inner_loss / (m_array[idx_scenario] + 1)
        m_array[idx_scenario] += 1

    estimator = np.mean(inner_losses_mean >= loss_threshold)

    return (estimator, inner_losses_mean, m_array)


@njit
def _estimation_threshold_njit(error_margin, n, simulate_scenario, simulate_inner_loss, loss_threshold, inner_loss_std):
    scenarios = simulate_scenario(n)
    sigmas = inner_loss_std(scenarios)
    inner_losses_mean = np.zeros(n)

    for i, scenario in enumerate(scenarios):
        inner_loss_mean = 0
        m_scenario = 0
        while m_scenario / sigmas[i] * np.abs(inner_loss_mean - loss_threshold) < error_margin:
            new_inner_loss = simulate_inner_loss(scenario=scenario, size=1)[0]
            inner_loss_mean = m_scenario / (m_scenario + 1) * inner_loss_mean + new_inner_loss / (m_scenario + 1)
            m_scenario += 1
        inner_losses_mean[i] = inner_loss_mean

    estimator = np.mean(inner_losses_mean >= loss_threshold)
    return estimator


@njit
def _estimation_adaptive_njit(n_0, m_0, tau_e, k, simulate_scenario, simulate_inner_loss, loss_threshold, inner_loss_std, estimate_inner_std=False):
    N_MAX = k
    n = n_0

    scenarios = np.zeros(N_MAX)
    sigmas = np.zeros(N_MAX)
    m_array = np.zeros(N_MAX)
    inner_losses_mean = np.zeros(N_MAX)
    if estimate_inner_std:
        squared_inner_losses_mean = np.zeros(N_MAX)

    scenarios[:n] = simulate_scenario(n)
    if not estimate_inner_std:
        sigmas[:n] = inner_loss_std(scenarios[:n])

    for i, scenario in enumerate(scenarios[:n]):
        inner_loss = simulate_inner_loss(scenario=scenario, size=m_0)
        inner_losses_mean[i] = np.mean(inner_loss)
        if estimate_inner_std:
            squared_inner_losses_mean[i] = np.mean(inner_loss**2)
    m_array[:n] = np.ones(n) * m_0

    if estimate_inner_std:
        sigmas[:n] = np.sqrt(squared_inner_losses_mean[:n] - inner_losses_mean[:n]**2)

    for j in range(int(np.ceil(k / tau_e))):
        m_average = np.mean(m_array[:n])
        alpha_bar = np.mean(np.array(list(map(gaussian_cdf, np.sqrt(m_array[:n]) * (inner_losses_mean[:n] - loss_threshold) / sigmas[:n]))))
        alpha_hat = np.mean(inner_losses_mean[:n] >= loss_threshold)
        bias = alpha_hat - alpha_bar
        variance = alpha_bar * (1 - alpha_bar) / n
        n_new = int(min(
            max(((variance * n)**0.2 * (m_average * n + tau_e)**0.8) / (4 * bias**2 * m_average**4)**0.2, n),
            n + tau_e)
        )
        n_to_add = n_new - n
        if n_to_add > 0:
            scenarios_to_add = simulate_scenario(n_to_add)
            scenarios[n:n_new] = scenarios_to_add
            if not estimate_inner_std:
                sigmas[n:n_new] = inner_loss_std(scenarios_to_add)
            for i, scenario in enumerate(scenarios[n:n_new]):
                inner_loss = simulate_inner_loss(scenario=scenario, size=m_0)
                inner_losses_mean[n + i] = np.mean(inner_loss)
                if estimate_inner_std:
                    squared_inner_losses_mean[n + i] = np.mean(inner_loss**2)
            m_array[n:n_new] = np.ones(n_to_add) * m_0

        n = n_new
        while np.sum(m_array[:n]) < (j + 1) * tau_e:
            idx_scenario = np.argmin(m_array[:n] * np.abs(inner_losses_mean[:n] - loss_threshold) / sigmas[:n])
            new_inner_loss = simulate_inner_loss(scenario=scenarios[idx_scenario], size=1)[0]
            inner_losses_mean[idx_scenario] = m_array[idx_scenario] / (m_array[idx_scenario] + 1) * inner_losses_mean[idx_scenario] + \
                new_inner_loss / (m_array[idx_scenario] + 1)
            if estimate_inner_std:
                squared_inner_losses_mean[idx_scenario] = m_array[idx_scenario] / (m_array[idx_scenario] + 1) * squared_inner_losses_mean[idx_scenario] + \
                    new_inner_loss**2 / (m_array[idx_scenario] + 1)
                sigmas[idx_scenario] = np.sqrt(squared_inner_losses_mean[idx_scenario] - inner_losses_mean[idx_scenario]**2)
            m_array[idx_scenario] += 1

    estimator = np.mean(inner_losses_mean[:n] >= loss_threshold)
    return estimator


class NestedMonteCarlo:
    """
    Hereafter, n stands for the number of scenario simulations, m stands for the number of internal loss simulations par scenario.
    """
    simulate_scenario: callable  # function of (size)
    simulate_inner_loss: callable  # function of (scenario, size)
    loss_threshold: float
    inner_loss_std: callable  # function of (scenario)

    def __init__(self, simulate_scenario, simulate_inner_loss, loss_threshold, inner_loss_std) -> None:
        self.simulate_scenario = simulate_scenario
        self.simulate_inner_loss = simulate_inner_loss
        self.loss_threshold = loss_threshold
        self.inner_loss_std = inner_loss_std

    def estimation_uniform(self, m, n):
        return _estimation_uniform_njit(m, n, self.simulate_scenario, self.simulate_inner_loss, self.loss_threshold)

    def estimation_sequential(self, m_init, m_average, n, return_info=False):
        estimator, inner_losses_mean, m_array = _estimation_sequential_njit(m_init, m_average, n, self.simulate_scenario, self.simulate_inner_loss,
                                                                            self.loss_threshold, self.inner_loss_std)
        return (estimator, inner_losses_mean, m_array) if return_info else estimator

    def estimation_threshold(self, error_margin, n):
        return _estimation_threshold_njit(error_margin, n, self.simulate_scenario, self.simulate_inner_loss, self.loss_threshold, self.inner_loss_std)

    def estimation_adaptive(self, n_0, m_0, tau_e, k, estimate_inner_std=False):
        return _estimation_adaptive_njit(n_0, m_0, tau_e, k, self.simulate_scenario, self.simulate_inner_loss, self.loss_threshold,
                                         self.inner_loss_std, estimate_inner_std=False)
