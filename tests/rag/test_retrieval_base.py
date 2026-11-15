
"""Tests for semantic retrieval interface."""

import pytest

from researchpilot.rag.retrieval.base import RetrievalService


def test_retrieval_service_is_abstract():
    with pytest.raises(TypeError):
        RetrievalService()
