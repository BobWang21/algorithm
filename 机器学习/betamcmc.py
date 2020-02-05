import random

import matplotlib.pyplot as plt
from scipy.stats import norm

T = 5000


def norm_dist_prob(theta):
    y = norm.pdf(theta, loc=3, scale=2)
    return y


pi = [0 for i in range(T)]
sigma = 1
t = 0
while t < T - 1:
    t = t + 1
    pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)[0]
    alpha = min(1, (norm_dist_prob(pi_star) / norm_dist_prob(pi[t - 1])))

    u = random.uniform(0, 1)
    if u < alpha:
        pi[t] = pi_star
    else:
        pi[t] = pi[t - 1]

plt.scatter(pi, norm.pdf(pi, loc=3, scale=2))
num_bins = 50
plt.hist(pi, num_bins, normed=1, facecolor='red', alpha=0.7)
plt.show()