import numpy as np
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

direct = [[-1, 0], [1, 0], [0, -1], [0, 1]]

def get_id(x: int, y: int, mesh_y: int):
    return x * mesh_y + y

def get_pos(id: int, mesh_y: int):
    return id // mesh_y, id % mesh_y

def check(x: int, y: int, mesh_x: int, mesh_y: int):
    return 0 <= x < mesh_x and 0 <= y < mesh_y

class NoCTomography:
    def __init__(self, mesh_x: int, mesh_y: int, factor):
        self.mesh_x = mesh_x
        self.mesh_y = mesh_y
        self.factor = factor

        # 路由器初始化
        self.router_num = mesh_x * mesh_y
        self.router_name = [f"router_{i}" for i in range(self.router_num)]
        
        # 链路初始化
        self.links = []
        self.link_name = []
        self.link_to_id = {}

        for i in range(mesh_x):
            row_id = i * mesh_y
            for j in range(mesh_y):
                cur_router = row_id + j
                for offset in direct:
                    nx = i + offset[0]
                    ny = j + offset[1]

                    if check(nx, ny, mesh_x, mesh_y):
                        neighbor_id = get_id(nx, ny, mesh_y)
                        start = min(cur_router, neighbor_id)
                        end = max(cur_router, neighbor_id)
                        self.links.append((start, end))
                        self.link_to_id[(start, end)] = len(self.links) - 1
                        self.link_name.append(f"link_{start}_{end}")
        
        self.link_num = len(self.links)
        # 先router 后link
        self.feature_num = self.router_num + self.link_num

        self.model = None
    
    # paths: List[(src, dst, cycle)]
    def build_feature_matrix(self, paths):
        num_paths = len(paths)
        if num_paths == 0:
            return np.array([]).reshape(0, self.feature_num), np.array([])

        X = np.zeros((num_paths, self.feature_num))
        y = np.zeros(num_paths)

        for id, (src, dst, time) in enumerate(paths):
            y[id] = time
            src_x, src_y = get_pos(src, self.mesh_y)
            dst_x, dst_y = get_pos(dst, self.mesh_y)

            while src_x != dst_x:
                X[id, get_id(src_x, src_y, self.mesh_y)] = 1
                if src_x < dst_x:
                    start_id = get_id(src_x, src_y, self.mesh_y)
                    end_id = get_id(src_x+1, src_y, self.mesh_y)
                    if start_id > end_id:
                        start_id, end_id = end_id, start_id
                    X[id, self.router_num + self.link_to_id[(start_id, end_id)]] = 1
                    src_x += 1
                else:
                    start_id = get_id(src_x, src_y, self.mesh_y)
                    end_id = get_id(src_x-1, src_y, self.mesh_y)
                    if start_id > end_id:
                        start_id, end_id = end_id, start_id
                    X[id, self.router_num + self.link_to_id[(start_id, end_id)]] = 1
                    src_x -= 1

            while src_y != dst_y:
                X[id, get_id(src_x, src_y, self.mesh_y)] = 1
                if src_y < dst_y:
                    start_id = get_id(src_x, src_y, self.mesh_y)
                    end_id = get_id(src_x, src_y+1, self.mesh_y)
                    if start_id > end_id:
                        start_id, end_id = end_id, start_id
                    X[id, self.router_num + self.link_to_id[(start_id, end_id)]] = 1
                    src_y += 1
                else:
                    start_id = get_id(src_x, src_y, self.mesh_y)
                    end_id = get_id(src_x, src_y-1, self.mesh_y)
                    if start_id > end_id:
                        start_id, end_id = end_id, start_id
                    X[id, self.router_num + self.link_to_id[(start_id, end_id)]] = 1
                    src_y -= 1

            X[id, src] = 0
        return X, y
    
    def train(self, paths, model_type = 'lasso', alpha = 0.1):
        X, y = self.build_feature_matrix(paths)

        if X.shape[0] == 0:
            print("No communication exist.")
            return
        
        if model_type == 'lasso':
            self.model = Lasso(alpha=alpha, positive=True, fit_intercept=False)
        elif model_type == 'ridge':
            print("Warning: Scikit-learn Ridge doesn't directly support 'positive=True'. Using LinearRegression(positive=True) as an alternative for non-negativity in this example if an L2 regularizer with positive constraint is not available.")
            self.model = LinearRegression(positive=True, fit_intercept=False)
        elif model_type == 'linear':
            self.model = LinearRegression(positive=True, fit_intercept=False)
        else:
            raise ValueError("Unsupported model_type. Choose 'lasso', 'ridge', or 'linear'.")
        
        try:
            self.model.fit(X, y)
        except Exception as e:
            print(f"Error during model training: {e}")
            return {name: float('nan') for name in self.feature_names}
        
        estimated_bandwidth = {}
        estimated_router_delay = {}
        coeffs = self.model.coef_ if hasattr(self.model, 'coef_') else [0.0] * self.feature_num

        for id, link in enumerate(self.links):
            estimated_bandwidth[link] = self.factor/coeffs[self.router_num+id] if self.router_num+id < len(coeffs) else 0.0
        
        for id in range(self.router_num):
            estimated_router_delay[id] = coeffs[id] if id < len(coeffs) else 0.0

        return estimated_bandwidth