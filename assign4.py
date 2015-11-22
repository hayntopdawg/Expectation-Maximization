#!/usr/bin/env python
from __future__ import division

import csv
import math
import numpy as np

__author__ = 'Jamie Fujimoto'


# Read input data
def read_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield row


# Splits data into feature matrix and class vector
def split_data(filename):
    X, y = [], []
    for i, row in enumerate(read_data(filename)):
        X.append(row[:-1])
        y.append(row[-1])
    return np.array(X, dtype='float'), np.array(y)


# Initialize by randomly selecting means
def init_mu(X, k):
    n = X.shape[0]
    d = X.shape[1]

    # create means list
    mu = np.zeros((k, d))

    # seed random numbers to make calculation deterministic
    np.random.seed(1)

    for i in xrange(k):
        for j in xrange(d):
            mu[i, j] = np.random.choice(X[:, j], 1)

    return mu


# Initialize each covariance matrix to the identity matrix
def init_sig(d, k):
    sig = [np.eye(d, d) for _ in xrange(k)]
    return np.array(sig)


# Initialize each prior probability to 1/k
def init_pri_prob(k):
    P = [1/k for _ in xrange(k)]
    return np.array(P)


def g_func(x, mu, sig):
    """
    Computes the D-variate Gaussian function
    :param x: single sample from data matrix D
    :param mu: mean vector for cluster i
    :param sig: covariance matrix for cluster i
    :return: f(x|mu(i), sig(i))
    """
    denom = pow(2 * math.pi, x.shape[0] / 2) * np.sqrt(np.linalg.det(sig))
    num = np.exp(-np.dot(np.dot((x - mu), np.linalg.inv(sig)), (x - mu).T) / 2)
    return num / denom


# Compute posterior probability P(Ci | x_j)
def compute_post_prob(x, mu, sig, p):
    return g_func(x, mu, sig) * p


# Compute weight w_ij
def compute_weight(x, Mu, Sig, P, i):
    denom = 0
    k = Mu.shape[0]
    for a in xrange(k):
        denom += compute_post_prob(x, Mu[a, :], Sig[a], P[a])
    return compute_post_prob(x, Mu[i, :], Sig[i], P[i]) / denom


def expectation_step(X, Mu, Sig, P):
    n = X.shape[0]
    k = Mu.shape[0]
    w = np.zeros((k, n))
    for i in xrange(k):
        for j in xrange(n):
            w[i, j] = compute_weight(X[j, :], Mu, Sig, P, i)
    return w


# re-estimate mean
def compute_mean(X, w):
    wi = w.reshape((w.shape[0], 1))
    num = np.sum(np.multiply(wi, X), 0)
    mu = num / np.sum(wi)
    return mu


# re-estimate covariance matrix
def compute_covariance(X, mu, w):
    wi = w.reshape((w.shape[0], 1))
    return (np.dot(np.multiply(wi, (X - mu)).T, (X - mu))) / np.sum(wi)


# re-estimate prior probability
def compute_pri_prob(w, n):
    wi = w.reshape((w.shape[0], 1))
    return np.sum(wi) / n


def maximization_step(X, Mu, Sig, P, W, k):
    for i in xrange(k):
        Mu[i, :] = compute_mean(X, W[i,:])
        Sig[i] = compute_covariance(X, Mu[i,:], W[i,:])
        P[i] = compute_pri_prob(W[i,:], X.shape[0])
    return Mu, Sig, P


class EM():
    def __init__(self, k=1, eps=0.001):
        self.k = k
        self.eps = eps


    def _initialize(self, X):
        self.n, self.d = X.shape
        self.Mu = init_mu(X, self.k)
        self.Sig = init_sig(X.shape[1], self.k)
        self.P = init_pri_prob(self.k)


    def train(self, X):
        self._initialize(X)

        self.t = 0
        converged = False
        while converged == False:
            Mu_prev = np.copy(self.Mu)
            # Tests
            W = expectation_step(X, self.Mu, self.Sig, self.P)
            # print W
            # print W.shape
            # mu = compute_mean(X, W[0, :])
            # print mu
            self.Mu, self.Sig, self.P = maximization_step(X, self.Mu, self.Sig, self.P, W, self.k)
            # print Mu
            # print Sig
            # print P
            for i in xrange(k):
                sq_norm = np.linalg.norm((self.Mu[i] - Mu_prev[i])) ** 2
                if sq_norm <= eps:
                    converged = True
                else:
                    converged = False
                    break
            self.t += 1
        # return Mu, Sig, P, t


    def get_mean(self):
        return self.Mu


    def get_covariance(self):
        return self.Sig


    def get_iterations(self):
        return self.t


    def predict(self):
        pass


    def purity_score(self):
        pass


def iris(k, eps):
    # Input (X) and target (y) datasets
    X, y = split_data('iris.txt')
    # print X.shape
    # print y.shape

    em = EM(k, eps)
    em._initialize(X)
    print "Initial mean:\n{}".format(em.get_mean())

    em.train(X)
    Mu = em.get_mean()
    Sig = em.get_covariance()
    t = em.get_iterations()
    return Mu, Sig, t


if __name__ == '__main__':
    k = 3
    eps = 0.001

    Mu, Sig, t = iris(k, eps)
    print "Mu:\n{}\n".format(Mu)
    print "Sig:\n{}\n".format(Sig)
    # print "P: {}\n".format(P)
    print "t: {}".format(t)