import numpy as np
from scipy import stats

def norm_pdf(mu, sigma):
    return stats.norm(loc=mu, scale=sigma)

def beta_pdf(alpha, beta_p):
    return stats.beta(a=alpha, b=beta_p)

def gamma_pdf(shape, scale):
    return stats.gamma(a=shape, scale=scale)

class CoreDist:
    def __init__(self, mu = 1024, sigma = 62.25):
        self.mu = mu
        self.sigma = sigma
        self.dist = norm_pdf(mu=mu, sigma=sigma)

    def range_cdf(self, a, b):
        return self.dist.cdf(b) - self.dist.cdf(a)
    
    def failslow_prob(self, x):
        tx = self.mu - abs(x - self.mu)
        return 1 - self.dist.cdf(tx) * 2
    
    def generate(self, size=1):
        numbers = self.dist.rvs(size=size)
        return numbers[0]
    
class NoCDist:
    def __init__(self, shape = 8.0, rate = 0.5):
        self.shape = shape
        self.rate = rate
        self.dist = gamma_pdf(shape=shape, scale=1.0/rate)

    def range_cdf(self, a, b):
        return self.dist.cdf(b) - self.dist.cdf(a)
    
    def failslow_prob(self, x):
        return 1 - self.dist.cdf(x)
    
    def generate(self, size=1):
        numbers = self.dist.rvs(size=size)
        return numbers[0]