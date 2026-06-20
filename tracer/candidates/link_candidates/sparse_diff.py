import numpy as np
from sklearn.linear_model import Lasso, LinearRegression

from tracer.candidates.link_candidates.interface import LinkCandidate, LinkCandidateDetector, normalize_link
from tracer.topology import TopologyPathFeatures


class SparseDifferentialLinkCandidateDetector(LinkCandidateDetector):
    name = "sparse-diff"

    def __init__(self, threshold: float = 5.0, alpha: float = 0.001):
        self.threshold = threshold
        self.alpha = alpha

    def _all_paths(self, summary):
        paths = []
        for inst_trace in summary.trace:
            exe_time = inst_trace.end_time - inst_trace.start_time
            paths.append((inst_trace.inference_time, inst_trace.data_size, exe_time, inst_trace.src_id, inst_trace.dst_id))
        return paths

    def _group_detect_paths(self, summary):
        grouped = {}
        for inst_trace in summary.trace:
            exe_time = inst_trace.end_time - inst_trace.start_time
            grouped.setdefault(inst_trace.inference_time, []).append(
                (inst_trace.data_size, exe_time, inst_trace.src_id, inst_trace.dst_id)
            )
        return grouped

    def _fit_baseline(self, summary, context):
        paths = [(data_size, exe_time, src, dst) for _, data_size, exe_time, src, dst in self._all_paths(summary)]
        path_model = TopologyPathFeatures(context.topology)
        X, y = path_model.build_weighted_matrix(paths)
        if X.shape[0] == 0:
            return path_model, np.zeros(path_model.feature_num)

        baseline_model = LinearRegression(positive=True, fit_intercept=False)
        baseline_model.fit(X, y)
        coeffs = baseline_model.coef_ if hasattr(baseline_model, "coef_") else np.zeros(path_model.feature_num)
        return path_model, coeffs

    def detect(self, normal_summary, detect_summary, context) -> list[LinkCandidate]:
        path_model, baseline_coeffs = self._fit_baseline(normal_summary, context)
        baseline_link_inv = baseline_coeffs[: path_model.link_num] if baseline_coeffs.size else np.zeros(path_model.link_num)
        detect_periods = self._group_detect_paths(detect_summary)

        candidates = []
        for period, paths in detect_periods.items():
            X_detect, y_detect = path_model.build_weighted_matrix(paths)
            if X_detect.shape[0] == 0:
                continue

            y_pred = X_detect @ baseline_coeffs
            residual = np.maximum(y_detect - y_pred, 0)
            if not np.any(residual > 0):
                continue

            link_matrix = X_detect[:, : path_model.link_num]
            sparse_solver = Lasso(alpha=self.alpha, positive=True, fit_intercept=False, max_iter=10000)
            sparse_solver.fit(link_matrix, residual)
            link_deltas = sparse_solver.coef_ if hasattr(sparse_solver, "coef_") else np.zeros(path_model.link_num)

            for link_id, delta in enumerate(link_deltas):
                if delta <= 0:
                    continue
                baseline_inv = baseline_link_inv[link_id]
                if baseline_inv > 0:
                    score = 1.0 + delta / baseline_inv
                else:
                    score = delta
                if score <= self.threshold:
                    continue

                src_id, dst_id = context.topology.normalize_link(*path_model.links[link_id])
                candidates.append(
                    LinkCandidate(
                        period=period,
                        src_id=src_id,
                        dst_id=dst_id,
                        score=float(score),
                    )
                )

        return candidates
