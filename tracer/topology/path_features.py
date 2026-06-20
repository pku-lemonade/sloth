import numpy as np


class TopologyPathFeatures:
    def __init__(self, topology):
        self.topology = topology
        self.links = topology.physical_links()
        self.link_to_id = {link: idx for idx, link in enumerate(self.links)}
        self.link_name = [f"link_{src}_{dst}" for src, dst in self.links]
        self.link_num = len(self.links)
        self.feature_num = self.link_num + 3

    def build_weighted_matrix(self, paths):
        num_paths = len(paths)
        if num_paths == 0:
            return np.array([]).reshape(0, self.feature_num), np.array([])

        X = np.zeros((num_paths, self.feature_num))
        y = np.zeros(num_paths)

        for row, (data_size, time, src, dst) in enumerate(paths):
            y[row] = time
            route_links = self.topology.route_links(src, dst)
            for link in route_links:
                X[row, self.link_to_id[self.topology.normalize_link(*link)]] = data_size
            X[row, self.link_num] = len(route_links)
            X[row, self.link_num + 1] = 1
            X[row, self.link_num + 2] = data_size

        return X, y
