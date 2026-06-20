from tracer.candidates.link_candidates.em import EMLinkCandidateDetector
from tracer.candidates.link_candidates.interface import LinkCandidate
from tracer.candidates.link_candidates.sparse_diff import SparseDifferentialLinkCandidateDetector
from tracer.candidates.link_candidates.tomography import TomographyLinkCandidateDetector

DEFAULT_LINK_DETECTOR = "tomography"

BACKENDS = {
    TomographyLinkCandidateDetector.name: TomographyLinkCandidateDetector,
    EMLinkCandidateDetector.name: EMLinkCandidateDetector,
    SparseDifferentialLinkCandidateDetector.name: SparseDifferentialLinkCandidateDetector,
}


def available_link_detectors() -> tuple[str, ...]:
    return tuple(BACKENDS.keys())


def detect_link_candidates(backend_name: str, normal_summary, detect_summary, context, threshold: float = 5.0) -> list[LinkCandidate]:
    if backend_name not in BACKENDS:
        raise ValueError(
            f"Unsupported link detector backend '{backend_name}'. "
            f"Choose one of: {', '.join(sorted(BACKENDS))}"
        )

    detector = BACKENDS[backend_name](threshold=threshold)
    try:
        return detector.detect(normal_summary, detect_summary, context)
    except Exception as exc:
        raise RuntimeError(f"Link detector backend '{backend_name}' failed: {exc}") from exc
