
"""Research result visualization components."""

import streamlit as st

from frontend.components.layout import render_status


def render_research_plan(plan: dict) -> None:
    """Display the generated research plan."""

    if not plan:
        st.info("No research plan available.")
        return

    st.markdown(f"**Research objective:** {plan.get('research_objective', 'N/A')}")

    st.markdown("**Sub-questions**")
    for index, question in enumerate(plan.get("sub_questions", []), 1):
        st.markdown(f"{index}. {question}")

    st.markdown("**Search queries**")

    for query in plan.get("search_queries", []):
        with st.expander(query.get("query", "Search query")):
            st.write("Purpose:", query.get("purpose", "N/A"))
            st.write("Source type:", query.get("source_type", "N/A"))

    st.markdown("**Research strategy**")
    st.write(plan.get("research_strategy", "N/A"))

    preferences = plan.get("source_preferences", {})
    if preferences:
        st.markdown("**Source preferences**")
        st.write("Primary sources:", preferences.get("primary_sources", []))
        st.write("Preferred evidence:", preferences.get("preferred_evidence", []))
        st.write("Supplementary sources:", preferences.get("supplementary_sources", []))


def render_execution_steps(steps: list) -> None:
    """Display agent execution history."""

    if not steps:
        st.info("No execution steps available.")
        return

    for step in steps:
        step_number = step.get("step_number", "?")
        action = step.get("action", "Unnamed action")
        status = step.get("status", "pending")

        with st.expander(f"Step {step_number}: {action}"):
            st.write("Status:", status)
            st.write("Tool:", step.get("tool_name") or "N/A")

            if step.get("tool_input"):
                st.markdown("**Tool input**")
                st.json(step["tool_input"])

            if step.get("tool_output"):
                st.markdown("**Tool output**")
                st.text(step["tool_output"])

            if step.get("error"):
                st.error(step["error"])


def render_evidence(pipeline: dict) -> None:
    """Display evaluated evidence and quality scores."""

    evidence_items = pipeline.get("evaluated_evidence", [])

    if not evidence_items:
        st.info("No evaluated evidence available.")
        return

    st.metric("Evidence items", len(evidence_items))

    for index, item in enumerate(evidence_items, 1):
        evidence = item.get("evidence", {})
        quality = item.get("quality", {})

        with st.expander(
            f"Evidence {index}: {evidence.get('source_id', 'Unknown source')}"
        ):
            st.write(evidence.get("content", ""))

            st.write("Retrieval score:", evidence.get("retrieval_score", "N/A"))

            st.markdown("**Quality assessment**")

            columns = st.columns(4)

            score_fields = [
                ("Relevance", "relevance"),
                ("Completeness", "completeness"),
                ("Clarity", "clarity"),
                ("Overall", "overall_score"),
            ]

            for column, (label, field) in zip(columns, score_fields):
                score = quality.get(field)
                column.metric(
                    label,
                    f"{score:.1%}" if isinstance(score, (int, float)) else "N/A",
                )


def render_sources(pipeline: dict) -> None:
    """Display academic source credibility."""

    sources = pipeline.get("evaluated_sources", [])

    if not sources:
        st.info("No evaluated sources available.")
        return

    for source in sources:
        credibility = source.get("credibility", {})

        with st.expander(source.get("title", "Untitled source")):
            st.write("Paper ID:", source.get("paper_id", "N/A"))
            st.write("Provider:", source.get("provider", "N/A"))

            columns = st.columns(5)

            fields = [
                ("DOI", "doi_score"),
                ("Citations", "citation_score"),
                ("Recency", "recency_score"),
                ("Metadata", "metadata_score"),
                ("Overall", "overall_score"),
            ]

            for column, (label, field) in zip(columns, fields):
                score = credibility.get(field)
                column.metric(
                    label,
                    f"{score:.1%}" if isinstance(score, (int, float)) else "N/A",
                )


def render_coverage(pipeline: dict) -> None:
    """Display research evidence coverage."""

    assessment = pipeline.get("coverage", {})
    coverage = assessment.get("coverage", {})

    if not assessment:
        st.info("No coverage assessment available.")
        return

    score = coverage.get("coverage_score", 0.0)

    st.metric("Coverage score", f"{score:.1%}")
    st.progress(max(0.0, min(1.0, score)))

    st.write("Evidence count:", assessment.get("evidence_count", 0))
    st.write("Total aspects:", coverage.get("total_aspects", 0))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Covered aspects**")
        for aspect in coverage.get("covered_aspects", []):
            st.write(f"✓ {aspect}")

    with col2:
        st.markdown("**Uncovered aspects**")
        for aspect in coverage.get("uncovered_aspects", []):
            st.write(f"• {aspect}")


def render_contradictions(pipeline: dict) -> None:
    """Display conflicting research claims."""

    assessment = pipeline.get("contradictions", {})

    if not assessment:
        st.info("No contradiction assessment available.")
        return

    st.metric(
        "Detected contradictions",
        assessment.get("contradiction_count", 0),
    )

    contradictions = assessment.get("contradictions", [])

    if not contradictions:
        st.success("No contradictions were detected.")
        return

    for index, item in enumerate(contradictions, 1):
        with st.expander(f"Contradiction {index}"):
            st.write("First source:", item.get("first_source_id"))
            st.write("First claim:", item.get("first_claim"))

            st.divider()

            st.write("Second source:", item.get("second_source_id"))
            st.write("Second claim:", item.get("second_claim"))

            score = item.get("contradiction_score")
            if isinstance(score, (int, float)):
                st.metric("Contradiction score", f"{score:.1%}")

            st.write("Explanation:", item.get("explanation"))


def render_synthesis(pipeline: dict) -> None:
    """Display synthesized research findings."""

    synthesis = pipeline.get("synthesis", {})

    if not synthesis:
        st.info("No research synthesis available.")
        return

    st.markdown("**Synthesis summary**")
    st.write(synthesis.get("synthesis_summary", "N/A"))

    st.write("Evidence count:", synthesis.get("evidence_count", 0))

    for index, finding in enumerate(synthesis.get("findings", []), 1):
        with st.expander(f"Finding {index}"):
            st.write(finding.get("finding", ""))

            confidence = finding.get("confidence_score")
            if isinstance(confidence, (int, float)):
                st.metric("Confidence", f"{confidence:.1%}")

            st.write("Supporting sources:", finding.get("supporting_sources", []))
            st.write("Evidence count:", finding.get("evidence_count", 0))


def render_report(pipeline: dict) -> None:
    """Display the generated research report."""

    report = pipeline.get("report", {})

    if not report:
        st.info("No research report available.")
        return

    st.markdown(f"### {report.get('title', 'Research Report')}")

    st.caption(f"Generated: {report.get('generated_at', 'N/A')}")

    for section in report.get("sections", []):
        st.markdown(f"#### {section.get('title', 'Section')}")
        st.markdown(section.get("content", ""))

        citations = section.get("citations", [])
        if citations:
            st.caption(
                "Citations: " + ", ".join(map(str, citations))
            )

    st.markdown("#### Conclusion")
    st.write(report.get("conclusion", "N/A"))


def render_citation_validation(pipeline: dict) -> None:
    """Display citation validation results."""

    validation = pipeline.get("citation_validation", {})

    if not validation:
        st.info("No citation validation available.")
        return

    if validation.get("is_valid"):
        st.success("All report citations passed validation.")
    else:
        st.error("Citation validation identified issues.")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Cited sources",
        len(validation.get("cited_sources", [])),
    )

    col2.metric(
        "Invalid sources",
        len(validation.get("invalid_sources", [])),
    )

    col3.metric(
        "Unused sources",
        len(validation.get("unused_sources", [])),
    )

    missing = validation.get("missing_citations", [])

    if missing:
        st.warning("Missing citations")
        for item in missing:
            st.write(f"- {item}")


def render_research_response(response: dict) -> None:
    """Render the complete research workflow response."""

    st.divider()
    st.markdown("## Research Results")

    status = response.get("status", "pending")
    render_status(status)

    if response.get("error"):
        st.error(response["error"])

    if response.get("final_answer"):
        st.markdown("### Final Answer")
        st.markdown(response["final_answer"])

    plan = response.get("research_plan")
    if plan:
        with st.expander("Research Plan", expanded=True):
            render_research_plan(plan)

    with st.expander("Execution History"):
        render_execution_steps(response.get("steps", []))

    pipeline = response.get("research_result")

    if not pipeline:
        st.info("Detailed research results are not available.")
        return

    tabs = st.tabs([
        "Evidence",
        "Sources",
        "Coverage",
        "Contradictions",
        "Synthesis",
        "Final Report",
        "Citations",
    ])

    renderers = [
        render_evidence,
        render_sources,
        render_coverage,
        render_contradictions,
        render_synthesis,
        render_report,
        render_citation_validation,
    ]

    for tab, renderer in zip(tabs, renderers):
        with tab:
            renderer(pipeline)
