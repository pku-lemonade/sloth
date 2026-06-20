from collections import defaultdict

from tracer.candidates.link_candidates.interface import LinkCandidate, LinkCandidateDetector, group_paths_by_inference, normalize_link
from tracer.topology import TopologyPathFeatures
from sklearn.linear_model import LinearRegression


class TomographyLinkCandidateDetector(LinkCandidateDetector):
    name = "tomography"

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

    def _estimate_period_bandwidths(self, summary, context) -> dict[int, dict[tuple[int, int], float]]:
        grouped_paths = group_paths_by_inference(summary)
        period_bandwidths = {}
        path_model = TopologyPathFeatures(context.topology)

        for period, paths in grouped_paths.items():
            X, y = path_model.build_weighted_matrix(paths)
            if X.shape[0] == 0:
                period_bandwidths[period] = {}
                continue

            model = LinearRegression(positive=True, fit_intercept=False)
            model.fit(X, y)
            coeffs = model.coef_ if hasattr(model, "coef_") else [0.0] * path_model.feature_num

            current_period = {}
            for link_id, link in enumerate(path_model.links):
                inv_bandwidth = coeffs[link_id]
                if inv_bandwidth == 0:
                    continue
                current_period[context.topology.normalize_link(link[0], link[1])] = 1.0 / inv_bandwidth
            period_bandwidths[period] = current_period

        return period_bandwidths

    def _baseline_bandwidths(self, summary, context) -> dict[tuple[int, int], float]:
        period_bandwidths = self._estimate_period_bandwidths(summary, context)
        aggregated = defaultdict(list)
        for period in period_bandwidths.values():
            for link, bandwidth in period.items():
                aggregated[link].append(bandwidth)
        return {link: sum(values) / len(values) for link, values in aggregated.items() if values}

    def detect(self, normal_summary, detect_summary, context) -> list[LinkCandidate]:
        baseline = self._baseline_bandwidths(normal_summary, context)
        candidates = []

        for period, bandwidths in self._estimate_period_bandwidths(detect_summary, context).items():
            for link, bandwidth in bandwidths.items():
                if link not in baseline or bandwidth == 0:
                    continue
                variance = baseline[link] / bandwidth
                if variance > self.threshold:
                    candidates.append(
                        LinkCandidate(
                            period=period,
                            src_id=link[0],
                            dst_id=link[1],
                            score=variance,
                        )
                    )
        return candidates
