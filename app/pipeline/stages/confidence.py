from __future__ import annotations


class ConfidenceScorer:
    def email_regex(self) -> float:
        return 0.99

    def phone_regex(self) -> float:
        return 0.95

    def header_name(self) -> float:
        return 0.90

    def location_heuristic(self) -> float:
        return 0.90

    def url(self) -> float:
        return 0.90

    def skill(self) -> float:
        return 0.90

    def section_extraction(self) -> float:
        return 0.85

    def ambiguous(self) -> float:
        return 0.55
