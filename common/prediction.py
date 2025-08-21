import csv
import numpy as np
from collections import defaultdict
from scipy.stats import norm

class EM_Model:
    def __init__(self, link_name, max_iter=100, tol=1e-4):
        self.link_name = link_name
        self.link_idx = {name: i for i, name in enumerate(link_name)}
        self.n_links = len(link_name)
        self.max_iter = max_iter
        self.tol = tol
        self.lr = 0.00001
        self.prev_loss = 0

        self.mu_bw_inv = np.ones(self.n_links) / 16
        self.sigma_bw_inv = np.ones(self.n_links) * 0.1

        self.mu_node = 1.0
        self.sigma_node = 0.1

        self.mu_start = 10
        self.sigma_start = 0.1

    def load_samples_from_csv(self, filepath):
        samples = []
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                row = [float(x) for x in row]
                link_sizes = {}
                for i in range(24):
                    if row[i] > 0:
                        link_id = self.link_name[i]
                        link_sizes[link_id] = row[i]

                n_nodes = int(row[24])
                T_start_coeff = float(row[25])
                T_comm = float(row[27])
                samples.append({
                    "link_sizes": link_sizes,  
                    "n_nodes": n_nodes,
                    "T_comm": T_comm
                })
        return samples
    
    def predict(self, sample):
        n_nodes = sample["n_nodes"]
        link_sizes = sample["link_sizes"]
        T_pred = self.mu_start + n_nodes * self.mu_node + sum(sz * self.mu_bw_inv[self.link_idx[l]] for l, sz in link_sizes.items())
        return T_pred
    
    def compute_loss_and_grad(self, samples):
        loss = 0.0
        grad_start = 0.0
        grad_node = 0.0
        grad_bw_inv = np.zeros_like(self.mu_bw_inv)
        size_count = np.zeros(self.n_links)

        for s in samples:
            T_pred = self.predict(s)
            T_true = s["T_comm"]
            err = T_pred - T_true
            loss += err**2

            grad_start += 2 * err
            grad_node += 2 * err * s["n_nodes"]
            for lid, size in s["link_sizes"].items():
                grad_bw_inv[self.link_idx[lid]] += 2 * err * size
                size_count[self.link_idx[lid]] += size

        grad_start /= len(samples)
        grad_node /= len(samples)
        grad_bw_inv /= size_count
        return loss, grad_start, grad_node, grad_bw_inv
        
    def fit(self, samples):
        for iteration in range(self.max_iter):
            loss, g_start, g_node, g_bw = self.compute_loss_and_grad(samples)

            self.mu_start -= self.lr * g_start
            self.mu_node -= self.lr * g_node
            self.mu_bw_inv -= self.lr * g_bw
            
            if abs(self.prev_loss - loss) < self.tol:
                print(f"[Gradient Descent] Converged at iter {iteration} with loss = {loss:.4f}")
                break
            self.prev_loss = loss

    def output(self):
        print("Estimated Parameters(EM algorithm):")
        print(f"mu_start: {self.mu_start:.4f}, sigma_start: {self.sigma_start:.4f}")
        print(f"mu_node: {self.mu_node:.4f}, sigma_node: {self.sigma_node:.4f}")
        for name, mu_inv, sigma_inv in zip(self.link_name, self.mu_bw_inv, self.sigma_bw_inv):
            print(f"{name}: mu_bw_inv = {mu_inv:.4f}, mu_bw = {1.0/mu_inv:.4f}, sigma_bw_inv = {sigma_inv:.4f}")
