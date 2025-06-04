import csv
import numpy as np
from collections import defaultdict
from scipy.stats import norm

# 基于分布预设和独立性假设
class EM_Model:
    def __init__(self, link_name, max_iter=100, tol=1e-4):
        self.link_name = link_name
        # link_name 到 id 的映射
        self.link_idx = {name: i for i, name in enumerate(link_name)}
        self.n_links = len(link_name)
        self.max_iter = max_iter
        self.tol = tol
        self.lr = 0.01
        self.prev_loss = 0

        # 初始化链路参数
        # 使用倒数作为估计参数
        self.mu_bw_inv = np.ones(self.n_links) / 16
        self.sigma_bw_inv = np.ones(self.n_links) / 10

        # 初始化节点延迟
        self.mu_node = 1.0
        self.sigma_node = 0.1

        # 初始化startup延迟
        self.mu_start = 1.0
        self.sigma_start = 0.1

    # 读入 csv 数据
    def load_samples_from_csv(self, filepath):
        samples = []
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                row = [float(x) for x in row]
                # Dict[str, float]
                link_sizes = {}
                for i in range(24):
                    if row[i] > 0:
                        link_id = self.link_name[i]
                        link_sizes[link_id] = row[i]

                n_nodes = int(row[24])
                # 始终为1，所以未放入sample
                T_start_coeff = float(row[25])
                T_comm = float(row[27])
                samples.append({
                    "link_sizes": link_sizes,  
                    "n_nodes": n_nodes,
                    "T_comm": T_comm
                })
        return samples
    
    # 即 E-step，根据当前估计值计算每条路径期望通信时间
    def predict(self, sample):
        n_nodes = sample["n_nodes"]
        link_sizes = sample["link_sizes"]
        T_pred = self.mu_start + n_nodes * self.mu_node + sum(sz * self.mu_bw_inv[self.link_idx[l]] for l, sz in link_sizes.items())
        return T_pred
    
    # 计算梯度，为 M-step 做准备
    def compute_loss_and_grad(self, samples):
        loss = 0.0
        grad_start = 0.0
        grad_node = 0.0
        grad_bw_inv = np.zeros_like(self.mu_bw_inv)

        for s in samples:
            T_pred = self.predict(s)
            T_true = s["T_comm"]
            print(f"pred: {T_pred}, True: {T_true}")
            err = T_pred - T_true
            loss += err**2

            grad_start += 2 * err
            grad_node += 2 * err * s["n_nodes"]
            for lid, size in s["link_sizes"].items():
                grad_bw_inv[self.link_idx[lid]] += 2 * err * size

        return loss, grad_start, grad_node, grad_bw_inv
        
    def fit(self, samples):
        for iteration in range(self.max_iter):
            # E-step
            loss, g_start, g_node, g_bw = self.compute_loss_and_grad(samples)
            # print(g_bw)

            # 基于梯度下降的 M-step
            self.mu_start -= self.lr * g_start
            self.mu_node -= self.lr * g_node
            self.mu_bw_inv -= self.lr * g_bw

            if iteration % 100 == 0:
                print(f"Iter {iteration}: Loss = {loss:.4f}")
            
            # 收敛检查
            if abs(self.prev_loss - loss) < self.tol:
                print(f"[Gradient Descent] Converged at iter {iteration} with loss = {loss:.4f}")
                break
            self.prev_loss = loss

    def output(self):
        print("Estimated Parameters:")
        print(f"mu_start: {self.mu_start:.4f}, sigma_start: {self.sigma_start:.4f}")
        print(f"mu_node: {self.mu_node:.4f}, sigma_node: {self.sigma_node:.4f}")
        for name, mu_inv, sigma_inv in zip(self.link_name, self.mu_bw_inv, self.sigma_bw_inv):
            print(f"{name}: mu_bw_inv = {mu_inv:.4f}, sigma_bw_inv = {sigma_inv:.4f}")
