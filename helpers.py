# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["IdentityMetric", "IsotropicMetric", "DiagonalMetric",
           "DenseMetric",
           "simple_hmc", "simple_nuts",
           "tf_simple_hmc", "tf_simple_nuts",
           "TFModel", ]

from collections import namedtuple

import numpy as np
from scipy.linalg import cholesky, solve_triangular

import tensorflow as tf
from tqdm import tqdm


class IdentityMetric(object):

    def __init__(self, ndim):
        self.ndim = int(ndim)

    def update_variance(self, variance):
        pass

    def sample_p(self):
        return np.random.randn(self.ndim)

    def dot(self, p):
        return p

    def restart(self):
        pass

    def update(self, sample):
        pass

    def finalize(self):
        pass


class IsotropicMetric(IdentityMetric):

    def __init__(self, ndim, variance=1.0):
        self.ndim = int(ndim)
        self.variance = float(variance)

    def update_variance(self, variance):
        self.variance = variance

    def sample_p(self):
        return np.random.randn(self.ndim) / np.sqrt(self.variance)

    def dot(self, p):
        return p * self.variance


class DiagonalMetric(IsotropicMetric):

    def __init__(self, variance):
        self.ndim = len(variance)
        self.variance = variance
        self.restart()

    def restart(self):
        self.counter = 0
        self.m = np.zeros(self.ndim)
        self.m2 = np.zeros(self.ndim)

    def update(self, sample):
        self.counter += 1
        delta = sample - self.m
        self.m += delta / self.counter
        self.m2 += (sample - self.m) * delta

    def finalize(self):
        if self.counter < 1:
            return
        var = self.m2 / (self.counter - 1)
        n = self.counter
        self.variance = (n / (n + 5.0)) * var
        self.variance += 1e-3 * (5.0 / (n + 5.0))
        self.restart()


class DenseMetric(IdentityMetric):

    def __init__(self, variance):
        self.ndim = len(variance)
        self.update_variance(variance)
        self.restart()

    def update_variance(self, variance):
        self.L = cholesky(variance, lower=False)
        self.variance = variance

    def sample_p(self):
        return solve_triangular(self.L, np.random.randn(self.ndim),
                                lower=False)

    def dot(self, p):
        return np.dot(self.variance, p)

    def restart(self):
        self.counter = 0
        self.m = np.zeros(self.ndim)
        self.m2 = np.zeros((self.ndim, self.ndim))

    def update(self, sample):
        self.counter += 1
        delta = sample - self.m
        self.m += delta / self.counter
        self.m2 += (sample - self.m)[:, None] * delta[None, :]

    def finalize(self):
        if self.counter < 1:
            return
        cov = self.m2 / (self.counter - 1)
        n = self.counter
        cov *= (n / (n + 5.0))
        cov[np.diag_indices_from(cov)] += 1e-3 * (5.0 / (n + 5.0))
        self.update_variance(cov)
        self.restart()


class ConstantStepSize(object):

    def __init__(self, step_size, jitter=None):
        self.step_size = step_size
        self.jitter = jitter

    def sample_step_size(self):
        jitter = self.jitter
        eps = self.get_step_size()
        if jitter is None:
            return eps
        jitter = np.clip(jitter, 0, 1)
        return eps * (1.0 - jitter * np.random.uniform(-1, 1))

    def get_step_size(self):
        return self.step_size

    def restart(self):
        pass

    def update(self, accept_stat):
        pass

    def finalize(self):
        pass


class StepSizeEstimator(ConstantStepSize):

    def __init__(self, delta=0.5, mu=0.5, gamma=0.05, kappa=0.75, t0=10,
                 jitter=None):
        self.jitter = jitter
        self.mu = mu
        self.delta = delta
        self.gamma = gamma
        self.kappa = kappa
        self.t0 = t0
        self.restart()

    def restart(self):
        self.counter = 0
        self.s_bar = 0.0
        self.x_bar = 0.0
        self.x = 0.0

    def get_step_size(self):
        return np.exp(self.x)

    def update(self, adapt_stat):
        self.counter += 1
        adapt_stat = min(adapt_stat, 1.0)
        eta = 1.0 / (self.counter + self.t0)
        self.s_bar = (1.0 - eta) * self.s_bar + eta * (self.delta - adapt_stat)
        self.x = self.mu - self.s_bar * np.sqrt(self.counter) / self.gamma
        x_eta = self.counter ** -self.kappa
        self.x_bar = (1.0 - x_eta) * self.x_bar + x_eta * self.x

    def finalize(self):
        self.x = self.x_bar


def leapfrog(grad_log_prob_fn, metric, q, p, epsilon, dUdq=None):
    q = np.array(q, copy=True)
    p = np.array(p, copy=True)

    if dUdq is None:
        dUdq = -grad_log_prob_fn(q)
    p -= 0.5 * epsilon * dUdq
    dTdp = metric.dot(p)
    q += epsilon * dTdp
    dUdq = -grad_log_prob_fn(q)
    p -= 0.5 * epsilon * dUdq

    return q, p, dUdq


def step_hmc(log_prob_fn, grad_log_prob_fn, metric, q, log_prob, epsilon,
             L):
    initial_q = np.array(q, copy=True)
    p = metric.sample_p()
    initial_h = 0.5 * np.dot(p, metric.dot(p))
    initial_h -= log_prob
    dUdq = -grad_log_prob_fn(q)
    for l in range(L):
        q, p, dUdq = leapfrog(grad_log_prob_fn, metric, q, p, epsilon,
                              dUdq)
    p = -p
    final_log_prob = log_prob_fn(q)
    final_h = 0.5 * np.dot(p, metric.dot(p))
    final_h -= final_log_prob
    accept = np.random.rand() < np.exp(initial_h - final_h)
    if accept:
        return q, final_log_prob, accept
    return initial_q, log_prob, accept


def simple_hmc(log_prob_fn, grad_log_prob_fn, q, niter, epsilon, L,
               metric=None):
    if metric is None:
        metric = IdentityMetric(len(q))

    samples = np.empty((niter, len(q)))
    samples_lp = np.empty(niter)
    log_prob = log_prob_fn(q)
    acc_count = 0
    for n in tqdm(range(niter), total=niter):
        q, log_prob, accept = step_hmc(log_prob_fn, grad_log_prob_fn,
                                       metric, q, log_prob, epsilon, L)
        acc_count += accept
        samples[n] = q
        samples_lp[n] = log_prob

    return samples, samples_lp, acc_count / float(niter)


def tf_simple_hmc(session, log_prob_tensor, var_list, niter, epsilon, L,
                  metric=None, feed_dict=None):
    model = TFModel(log_prob_tensor, var_list, session=session,
                    feed_dict=feed_dict)
    model.setup()
    q = model.current_vector()

    # Run the HMC
    samples, samples_lp, acc_frac = simple_hmc(
        model.value, model.gradient, q, niter, epsilon, L,
        metric=metric
    )

    # Update the variables
    fd = model.vector_to_feed_dict(samples[-1])
    feed = {} if feed_dict is None else feed_dict
    session.run([tf.assign(v, fd[v]) for v in var_list], feed_dict=feed)

    return samples, samples_lp, acc_frac


Point = namedtuple("Point", ("q", "p", "U", "dUdq"))


def _nuts_criterion(p_sharp_minus, p_sharp_plus, rho):
    return np.dot(p_sharp_plus, rho) > 0 and np.dot(p_sharp_minus, rho) > 0


def _nuts_tree(log_prob_fn, grad_log_prob_fn, metric, epsilon,
               depth, z, z_propose, p_sharp_left, p_sharp_right, rho, H0,
               sign, n_leapfrog, log_sum_weight, sum_metro_prob, max_depth,
               max_delta_h):
    if depth == 0:
        q, p, dUdq = leapfrog(grad_log_prob_fn, metric, z.q, z.p,
                              sign * epsilon, z.dUdq)
        z = Point(q, p, -log_prob_fn(q), dUdq)
        n_leapfrog += 1

        h = 0.5 * np.dot(p, metric.dot(p))
        h += z.U
        if not np.isfinite(h):
            h = np.inf
        valid_subtree = (h - H0) <= max_delta_h

        log_sum_weight = np.logaddexp(log_sum_weight, H0 - h)
        sum_metro_prob += min(np.exp(H0 - h), 1.0)

        z_propose = z
        rho += z.p

        p_sharp_left = metric.dot(z.p)
        p_sharp_right = p_sharp_left

        return (
            valid_subtree, z, z_propose, p_sharp_left, p_sharp_right, rho,
            n_leapfrog, log_sum_weight, sum_metro_prob
        )

    p_sharp_dummy = np.empty_like(p_sharp_left)

    # Left
    log_sum_weight_left = -np.inf
    rho_left = np.zeros_like(rho)
    results_left = _nuts_tree(
        log_prob_fn, grad_log_prob_fn, metric, epsilon,
        depth - 1, z, z_propose, p_sharp_left, p_sharp_dummy, rho_left,
        H0, sign, n_leapfrog, log_sum_weight_left, sum_metro_prob, max_depth,
        max_delta_h
    )
    (valid_left, z, z_propose, p_sharp_left, p_sharp_dummy, rho_left,
     n_leapfrog, log_sum_weight_left, sum_metro_prob) = results_left

    if not valid_left:
        return (
            False, z, z_propose, p_sharp_left, p_sharp_right, rho,
            n_leapfrog, log_sum_weight, sum_metro_prob
        )

    # Right
    z_propose_right = Point(z.q, z.p, z.U, z.dUdq)
    log_sum_weight_right = -np.inf
    rho_right = np.zeros_like(rho)
    results_right = _nuts_tree(
        log_prob_fn, grad_log_prob_fn, metric, epsilon,
        depth - 1, z, z_propose_right, p_sharp_dummy, p_sharp_right, rho_right,
        H0, sign, n_leapfrog, log_sum_weight_right, sum_metro_prob, max_depth,
        max_delta_h
    )
    (valid_right, z, z_propose_right, p_sharp_dummy, p_sharp_right, rho_right,
     n_leapfrog, log_sum_weight_right, sum_metro_prob) = results_right

    if not valid_right:
        return (
            False, z, z_propose, p_sharp_left, p_sharp_right, rho,
            n_leapfrog, log_sum_weight, sum_metro_prob
        )

    # Multinomial sample from the right
    log_sum_weight_subtree = np.logaddexp(log_sum_weight_left,
                                          log_sum_weight_right)
    log_sum_weight = np.logaddexp(log_sum_weight, log_sum_weight_subtree)

    if log_sum_weight_right > log_sum_weight_subtree:
        z_propose = z_propose_right
    else:
        accept_prob = np.exp(log_sum_weight_right - log_sum_weight_subtree)
        if np.random.rand() < accept_prob:
            z_propose = z_propose_right

    rho_subtree = rho_left + rho_right
    rho += rho_subtree

    return (
        _nuts_criterion(p_sharp_left, p_sharp_right, rho_subtree),
        z, z_propose, p_sharp_left, p_sharp_right, rho,
        n_leapfrog, log_sum_weight, sum_metro_prob
    )


def step_nuts(log_prob_fn, grad_log_prob_fn, metric, q, log_prob, epsilon,
              max_depth, max_delta_h):
    dUdq = -grad_log_prob_fn(q)
    p = metric.sample_p()

    z_plus = Point(q, p, -log_prob, dUdq)
    z_minus = Point(q, p, -log_prob, dUdq)
    z_sample = Point(q, p, -log_prob, dUdq)
    z_propose = Point(q, p, -log_prob, dUdq)

    p_sharp_plus = metric.dot(p)
    p_sharp_dummy = np.array(p_sharp_plus, copy=True)
    p_sharp_minus = np.array(p_sharp_plus, copy=True)
    rho = np.array(p, copy=True)

    n_leapfrog = 0
    log_sum_weight = 0.0
    sum_metro_prob = 0.0
    H0 = 0.5 * np.dot(p, metric.dot(p))
    H0 -= log_prob

    for depth in range(max_depth):
        rho_subtree = np.zeros_like(rho)
        valid_subtree = False
        log_sum_weight_subtree = -np.inf

        if np.random.rand() > 0.5:
            results = _nuts_tree(
                log_prob_fn, grad_log_prob_fn, metric, epsilon,
                depth, z_plus, z_propose, p_sharp_dummy, p_sharp_plus,
                rho_subtree, H0, 1, n_leapfrog, log_sum_weight_subtree,
                sum_metro_prob, max_depth, max_delta_h)
            (valid_subtree, z_plus, z_propose, p_sharp_dummy, p_sharp_plus,
             rho_subtree, n_leapfrog, log_sum_weight_subtree, sum_metro_prob) \
                = results

        else:
            results = _nuts_tree(
                log_prob_fn, grad_log_prob_fn, metric, epsilon,
                depth, z_minus, z_propose, p_sharp_dummy, p_sharp_minus,
                rho_subtree, H0, -1, n_leapfrog, log_sum_weight_subtree,
                sum_metro_prob, max_depth, max_delta_h)
            (valid_subtree, z_minus, z_propose, p_sharp_dummy, p_sharp_minus,
             rho_subtree, n_leapfrog, log_sum_weight_subtree, sum_metro_prob) \
                = results

        if not valid_subtree:
            break

        if log_sum_weight_subtree > log_sum_weight:
            z_sample = z_propose
        else:
            accept_prob = np.exp(log_sum_weight_subtree - log_sum_weight)
            if np.random.rand() < accept_prob:
                z_sample = z_propose

        log_sum_weight = np.logaddexp(log_sum_weight, log_sum_weight_subtree)
        rho += rho_subtree

        if not _nuts_criterion(p_sharp_minus, p_sharp_plus, rho):
            break

    accept_prob = sum_metro_prob / n_leapfrog
    return z_sample.q, log_prob_fn(q), float(accept_prob)


def simple_nuts(log_prob_fn, grad_log_prob_fn, q, nsample, epsilon,
                metric=None, max_depth=5, max_delta_h=1000.0,
                tune=False, tune_step_size=False, tune_metric=False,
                initial_buffer=100, final_buffer=100, window=25,
                nwarmup=None):
    if metric is None:
        metric = IdentityMetric(len(q))
    try:
        epsilon.sample_step_size()
    except AttributeError:
        epsilon = ConstantStepSize(epsilon)

    if nwarmup is None:
        nwarmup = int(0.5 * nsample)
    assert nwarmup <= nsample

    samples = np.empty((nsample, len(q)))
    samples_lp = np.empty(nsample)
    log_prob = log_prob_fn(q)
    acc_count = 0
    pbar = tqdm(range(nsample), total=nsample)

    inner_window = nwarmup - initial_buffer - final_buffer
    windows = window * 2 ** np.arange(np.ceil(np.log2(inner_window)
                                              - np.log2(window)) + 1)
    if windows[-1] > inner_window:
        windows = np.append(windows[:-2], inner_window)
    windows += initial_buffer
    windows = set(windows.astype(int))

    for n in pbar:
        step = epsilon.sample_step_size()
        q, log_prob, accept = step_nuts(log_prob_fn, grad_log_prob_fn,
                                        metric, q, log_prob, step,
                                        max_depth, max_delta_h)
        pbar.set_description("{0:.1e}, {1:.3f}".format(step, acc_count/(n+1)))

        if n < nwarmup:
            if tune or tune_step_size:
                epsilon.update(accept)
            if n >= initial_buffer and (tune or tune_metric):
                metric.update(q)
                if (n+1) in windows:
                    print(n+1, "updating metric")
                    metric.finalize()
                    if tune or tune_step_size:
                        epsilon.restart()
                    print(epsilon.get_step_size(), epsilon.sample_step_size())

        if n == nwarmup - 1 and (tune or tune_step_size):
            epsilon.finalize()

        acc_count += accept
        samples[n] = q
        samples_lp[n] = log_prob

    if tune or tune_step_size:
        epsilon.finalize()
    if tune or tune_metric:
        metric.finalize()

    return samples, samples_lp, acc_count / float(nsample), metric, epsilon


def tf_simple_nuts(session, log_prob_tensor, var_list, niter, epsilon,
                   metric=None, max_depth=5, max_delta_h=1000.0,
                   feed_dict=None,
                   tune=False, tune_step_size=False, tune_metric=False):
    model = TFModel(log_prob_tensor, var_list, session=session,
                    feed_dict=feed_dict)
    model.setup()
    q = model.current_vector()

    results = simple_nuts(
        model.value, model.gradient, q, niter, epsilon,
        metric=metric, max_depth=max_depth, max_delta_h=max_delta_h,
        tune=tune, tune_step_size=tune_step_size, tune_metric=tune_metric
    )

    # Update the variables
    fd = model.vector_to_feed_dict(results[0][-1])
    feed = {} if feed_dict is None else feed_dict
    session.run([tf.assign(v, fd[v]) for v in var_list], feed_dict=feed)

    return results


class TFModel(object):

    def __init__(self, target, var_list, feed_dict=None, session=None):
        self.target = target
        self.var_list = var_list
        self.grad_target = tf.gradients(self.target, self.var_list)
        self.feed_dict = {} if feed_dict is None else feed_dict
        self._session = session

    @property
    def session(self):
        if self._session is None:
            return tf.get_default_session()
        return self._session

    def value(self, vector):
        feed_dict = self.vector_to_feed_dict(vector)
        return self.session.run(self.target, feed_dict=feed_dict)

    def gradient(self, vector):
        feed_dict = self.vector_to_feed_dict(vector)
        return np.concatenate([
            np.reshape(g, s) for s, g in zip(
                self.sizes,
                self.session.run(self.grad_target, feed_dict=feed_dict))
        ])

    def setup(self, session=None):
        if session is not None:
            self._session = session
        values = self.session.run(self.var_list)
        self.sizes = [np.size(v) for v in values]
        self.shapes = [np.shape(v) for v in values]

    def vector_to_feed_dict(self, vector):
        i = 0
        fd = dict(self.feed_dict)
        for var, size, shape in zip(self.var_list, self.sizes, self.shapes):
            fd[var] = np.reshape(vector[i:i+size], shape)
            i += size
        return fd

    def feed_dict_to_vector(self, feed_dict):
        return np.concatenate([
            np.reshape(feed_dict[v], s)
            for v, s in zip(self.var_list, self.sizes)])

    def current_vector(self):
        values = self.session.run(self.var_list)
        return np.concatenate([
            np.reshape(v, s)
            for v, s in zip(values, self.sizes)])
