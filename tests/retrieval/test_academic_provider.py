import pytest

from researchpilot.retrieval.academic.base import AcademicSearchProvider

def test_academic_provider_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AcademicSearchProvider()