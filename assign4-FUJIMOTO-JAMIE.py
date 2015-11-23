#!/usr/bin/env python
from __future__ import division
from collections import Counter

import csv
import math
import numpy as np
import sys

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


class EM():
    def __init__(self, k=1, eps=0.001):
        self.k = k
        self.eps = eps


    # Initialize by randomly selecting means
    def _init_mu(self, X):
        mu = np.zeros((self.k, self.d))

        # seed random numbers to make calculation deterministic
        # np.random.seed(1)

        for i in xrange(self.k):
            for j in xrange(self.d):
                mu[i, j] = np.random.choice(X[:, j], 1)
        return mu


    # Initialize each covariance matrix to the identity matrix
    def _init_sig(self):
        sig = [np.eye(self.d, self.d) for _ in xrange(self.k)]
        return np.array(sig)  # should I leave it as a list?


    # Initialize each prior probability to 1/k
    def _init_pri_prob(self):
        P = [1 / self.k for _ in xrange(self.k)]
        return np.array(P)  # should I leave it as a list?


    def _g_func(self, x, mu, sig):
        """
        Computes the D-variate Gaussian function
        :param x: single sample from data matrix D
        :param mu: mean vector for cluster i
        :param sig: covariance matrix for cluster i
        :return: f(x|mu(i), sig(i))
        """
        denom = pow(2 * math.pi, x.shape[0] / 2) * np.sqrt(np.linalg.det(sig))
        if denom == 0:
            lambda_I = 0.001 * np.eye(sig.shape[0])
            sig += lambda_I
            denom = pow(2 * math.pi, x.shape[0] / 2) * np.sqrt(np.linalg.det(sig))

        num = np.exp(-np.dot(np.dot((x - mu), np.linalg.inv(sig)), (x - mu).T) / 2)
        return num / denom


    # Compute posterior probability P(Ci | x_j)
    def _compute_post_prob(self, x, mu, sig, p):
        return self._g_func(x, mu, sig) * p


    # Compute weight w_ij
    def _compute_weight(self, x, i):
        denom = 0
        for a in xrange(self.k):
            denom += self._compute_post_prob(x, self.Mu[a, :], self.Sig[a], self.P[a])
        # print denom
        return self._compute_post_prob(x, self.Mu[i, :], self.Sig[i], self.P[i]) / denom


    def expectation_step(self, X):
        self.W = np.zeros((self.k, self.n))
        for i in xrange(self.k):
            for j in xrange(self.n):
                self.W[i, j] = self._compute_weight(X[j, :], i)


    # re-estimate mean
    def _compute_mean(self, X, w):
        wi = w.reshape((w.shape[0], 1))
        num = np.sum(np.multiply(wi, X), 0)
        mu = num / np.sum(wi)
        return mu


    # re-estimate covariance matrix
    def _compute_covariance(self, X, mu, w):
        wi = w.reshape((w.shape[0], 1))
        return (np.dot(np.multiply(wi, (X - mu)).T, (X - mu))) / np.sum(wi)


    # re-estimate prior probability
    def _compute_pri_prob(self, w):
        wi = w.reshape((w.shape[0], 1))
        return np.sum(wi) / self.n


    def maximization_step(self, X):
        for i in xrange(self.k):
            self.Mu[i, :] = self._compute_mean(X, self.W[i, :])
            self.Sig[i] = self._compute_covariance(X, self.Mu[i, :], self.W[i, :])
            self.P[i] = self._compute_pri_prob(self.W[i, :])


    def _initialize(self, X):
        self.n, self.d = X.shape
        self.Mu = self._init_mu(X)
        self.Sig = self._init_sig()
        self.P = self._init_pri_prob()


    def train(self, X):
        self._initialize(X)

        self.t = 0
        converged = False
        while converged == False:
            Mu_prev = np.copy(self.Mu)
            self.expectation_step(X)
            self.maximization_step(X)
            for i in xrange(self.k):
                sq_norm = np.linalg.norm((self.Mu[i] - Mu_prev[i])) ** 2
                if sq_norm <= self.eps:
                    converged = True
                else:
                    converged = False
                    break
            self.t += 1


    def predict(self):
        self.preds = np.argmax(self.W, axis=0)


    def purity_score(self, y):
        C = [[] for _ in xrange(self.k)]
        for n, i in enumerate(self.preds):
            C[i].append(y[n])
        clusters = [[] for _ in xrange(self.k)]
        for j in xrange(self.k):
            clusters[j] = Counter(C[j])
        p = 0
        for cluster in clusters:
            p += cluster.most_common(1)[0][1]
        return clusters, p/self.n


def em_single(filename, k, eps):
    # Input (X) and target (y) datasets
    X, y = split_data(filename)

    em = EM(k, eps)
    em.train(X)
    em.predict()
    counts = np.bincount(em.preds)
    clusters, purity = em.purity_score(y)

    return em, X, y, counts, clusters, purity


def write_to(filename, em, X, y, counts, clusters, purity):
    with open(filename, "w") as f:
        f.write("Final mean for each cluster:\n")
        for i in xrange(em.k):
            f.write("Cluster {}: {}\n".format(i, em.Mu[i,:]))
        f.write("\nFinal covariance matrix for each cluster:\n")
        for i in xrange(em.k):
            f.write("Cluster {}:\n{}\n".format(i, em.Sig[i]))
        # print "W:\n{}\n".format(W)
        f.write("\nEM algorithm took {} iterations.\n".format(em.t))
        f.write("\nFinal cluster assignments of all points:\n")
        for n, i in enumerate(em.preds):
             f.write("{} {} --> Cluster: {}\n".format(X[n], y[n], i))
        f.write("\nFinal size of each cluster:\n")
        for i in xrange(em.k):
            f.write("Cluster {}: {}\n".format(i, counts[i]))
        f.write("\nCluster breakdown: {}\n".format(clusters))
        f.write("\nPurity score: {}".format(purity))


def find_purest(filename, k, eps, iterations=100):
    purest = 0
    for i in xrange(iterations):
        em, X, y, counts, clusters, purity = em_single(filename, k, eps)
        # if i % 10 == 0:
        print purity
        if purity > purest:
            purest_em = em
            purest_counts = counts
            purest_clusters = clusters
            purest = purity
    outfile = filename.split(".txt")[0] + "_output.txt"
    write_to(outfile, purest_em, X, y, purest_counts, purest_clusters, purest)


def script():
    filename = sys.argv[1]
    k = int(sys.argv[2])
    eps = float(sys.argv[3])

    em, X, y, counts, clusters, purity = em_single(filename, k, eps)
    outfile = filename.split(".txt")[0] + "_output.txt"
    write_to(outfile, em, X, y, counts, clusters, purity)


if __name__ == '__main__':
    # em_single('iris.txt', k, eps)
    # find_purest('iris.txt', 3, 0.001, 50)
    # find_purest('1R2RC_truth.txt', 3, 0.001, 50)
    # find_purest("dancing_truth.txt", 5, 0.001, 50)
    script()