from __future__ import annotations

from src.services.metrics import EmbeddingMetricService


def test_hash_backend_scr_and_rdi(monkeypatch):
    monkeypatch.setenv("SCR_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("SCR_HASH_DIM", "512")

    svc = EmbeddingMetricService()
    assert svc.embedding_backend == "hash"

    scr_identical = svc.calculate_scr(["alpha beta gamma"] * 4)
    assert scr_identical is not None
    assert scr_identical < 1e-6

    scr_divergent = svc.calculate_scr(
        [
            "alpha beta gamma",
            "delta epsilon zeta",
            "eta theta iota",
            "kappa lambda mu",
        ]
    )
    assert scr_divergent is not None
    assert scr_divergent > 0.5

    rdi_same = svc.calculate_rdi("foo bar baz", "foo bar baz")
    assert rdi_same is not None
    assert rdi_same < 1e-6

    rdi_diff = svc.calculate_rdi("foo bar baz", "qux quux corge")
    assert rdi_diff is not None
    assert rdi_diff > 0.5

