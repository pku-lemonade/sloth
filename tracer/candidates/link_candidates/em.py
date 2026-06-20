import math

import numpy as np

from common.prediction import EM_Model
from tracer.candidates.link_candidates.interface import LinkCandidate, LinkCandidateDetector, group_paths_by_inference, normalize_link
from tracer.topology import TopologyPathFeatures


class EMLinkCandidateDetector(LinkCandidateDetector):
    name = "em"

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

    def _build_samples(self, paths, context, link_name_map) -> list[dict]:
        samples = []
        for data_size, time, src, dst in paths:
            route_links = context.topology.route_links(src, dst)
            if not route_links:
                continue
            link_sizes = {link_name_map[link]: data_size for link in route_links if link in link_name_map}
            if not link_sizes:
                continue
            samples.append(
                {
                    "link_sizes": link_sizes,
                    "n_nodes": len(route_links),
                    "T_comm": time,
                }
            )
        return samples

    def _estimate_period_bandwidths(self, summary, context) -> dict[int, dict[tuple[int, int], float]]:
        grouped_paths = group_paths_by_inference(summary)
        period_bandwidths = {}
        path_model = TopologyPathFeatures(context.topology)
        link_name_map = {link: name for link, name in zip(path_model.links, path_model.link_name)}

        for period, paths in grouped_paths.items():
            samples = self._build_samples(paths, context, link_name_map)
            if not samples:
                period_bandwidths[period] = {}
                continue

            model = EM_Model(link_name=path_model.link_name)
            model.fit(samples)

            current_period = {}
            for link, name in zip(path_model.links, path_model.link_name):
                mu_inv = model.mu_bw_inv[model.link_idx[name]]
                if not np.isfinite(mu_inv) or mu_inv <= 0:
                    continue
                bandwidth = 1.0 / mu_inv
                if math.isfinite(bandwidth):
                    current_period[context.topology.normalize_link(link[0], link[1])] = bandwidth
            period_bandwidths[period] = current_period

        return period_bandwidths

    def _baseline_bandwidths(self, summary, context) -> dict[tuple[int, int], float]:
        period_bandwidths = self._estimate_period_bandwidths(summary, context)
        aggregated = {}
        counts = {}
        for period in period_bandwidths.values():
            for link, bandwidth in period.items():
                aggregated[link] = aggregated.get(link, 0.0) + bandwidth
                counts[link] = counts.get(link, 0) + 1
        return {link: aggregated[link] / counts[link] for link in aggregated}

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
