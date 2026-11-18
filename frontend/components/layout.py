
"""Shared layout components for the ResearchPilot frontend."""

from pathlib import Path

import streamlit as st


CSS_PATH = (
    Path(__file__).resolve().parents[1]
    / "styles"
    / "custom.css"
)


def load_custom_css() -> None:
    """Load the global frontend stylesheet."""
    if not CSS_PATH.exists():
        st.warning("Custom stylesheet not found.")
        return

    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the ResearchPilot application header."""
    st.markdown(
        """
        <div class="rp-header">
            <div class="rp-brand">
                Research<span>Pilot</span>
            </div>
            <div class="rp-subtitle">
                Your intelligent research companion
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(
    title: str,
    description: str | None = None,
) -> None:
    """Render a consistent section heading."""
    st.markdown(
        f'<div class="rp-section-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if description:
        st.markdown(
            f'<div class="rp-muted">{description}</div>',
            unsafe_allow_html=True,
        )


def render_status(status: str) -> None:
    """Render the current research workflow status."""
    status_config = {
        "pending": ("⏳", "Pending"),
        "running": ("⚙️", "Running"),
        "completed": ("✓", "Completed"),
        "failed": ("✕", "Failed"),
    }

    icon, label = status_config.get(
        status,
        ("?", status.title()),
    )

    if status == "completed":
        st.success(f"{icon} Workflow {label}")

    elif status == "failed":
        st.error(f"{icon} Workflow {label}")

    elif status == "running":
        st.info(f"{icon} Workflow {label}")

    else:
        st.warning(f"{icon} Workflow {label}")
