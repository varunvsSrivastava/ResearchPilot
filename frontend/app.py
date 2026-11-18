
"""Main Streamlit application for ResearchPilot."""

import os

import streamlit as st

from frontend.components.layout import (
    load_custom_css,
    render_header,
    render_section_header,
)
from frontend.services.api_client import (
    ResearchPilotAPIError,
    ResearchPilotClient,
)
from frontend.components.research_display import (
    render_research_response,
)

st.set_page_config(
    page_title="ResearchPilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_custom_css()
render_header()


# ---------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## Configuration")

    api_url = st.text_input(
        "Backend API URL",
        value=os.getenv(
            "RESEARCHPILOT_API_URL",
            "http://127.0.0.1:8000",
        ),
    ).strip().rstrip("/")

    check_health = st.button(
        "Check API Connection",
        use_container_width=True,
    )

    if check_health:
        try:
            client = ResearchPilotClient(base_url=api_url)
            health = client.health_check()

            if health.get("status") == "healthy":
                st.success("Backend connected successfully.")
            else:
                st.warning(f"Unexpected health response: {health}")

        except ResearchPilotAPIError as exc:
            st.error(str(exc))

    st.divider()

    st.caption("ResearchPilot v0.1.0")
    st.caption("Evidence-Driven AI Research Assistant")


# ---------------------------------------------------------
# Main workspace
# ---------------------------------------------------------

st.markdown(
    "### Your research workspace"
)

st.markdown(
    """
    Investigate research questions using academic sources,
    evidence retrieval, quality evaluation, and AI-powered synthesis.
    """
)

render_section_header(
    "Start a research task",
    "Enter a research question to begin an evidence-driven investigation.",
)

research_question = st.text_area(
    "Research question",
    placeholder=(
        "Example: How does retrieval-augmented generation "
        "improve factual accuracy in large language models?"
    ),
    height=130,
    max_chars=2000,
)

character_count = len(research_question.strip())

st.caption(f"{character_count}/2000 characters")

can_submit = 3 <= character_count <= 2000

if st.button(
    "Start Research",
    type="primary",
    use_container_width=True,
    disabled=not can_submit,
):
    client = ResearchPilotClient(base_url=api_url)

    try:
        with st.spinner(
            "ResearchPilot is investigating your question..."
        ):
            result = client.execute_research(research_question)

        st.session_state["research_result"] = result

        st.success("Research workflow execution finished.")

    except ResearchPilotAPIError as exc:
        st.error(f"Research execution failed: {exc}")

    except ValueError as exc:
        st.error(str(exc))


# ---------------------------------------------------------
# Initial workspace state
# ---------------------------------------------------------


if "research_result" in st.session_state:
    render_research_response(
        st.session_state["research_result"]
    )