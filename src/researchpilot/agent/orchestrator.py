
"""Research agent orchestration dependencies."""

from researchpilot.agent.research_agent import ResearchAgent
from researchpilot.planner.research_planner import generate_research_plan
from researchpilot.tools.registry import ToolRegistry
from researchpilot.rag.pipeline.service import RAGPipelineService
from researchpilot.retrieval.pipeline import RetrievalPipeline
from researchpilot.rag.indexing.service import EvidenceIndexingService
from researchpilot.rag.pipeline.models import RAGPipelineRequest
from researchpilot.generation.report_pipeline import ResearchReportPipeline
from researchpilot.rag.retrieval.models import RetrievalRequest
from researchpilot.evidence.aspect_coverage import AspectCoverageAssessor


class ResearchOrchestrator:
    """Coordinate planning and research tool execution."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
        rag_pipeline_service: RAGPipelineService | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
        indexing_service: EvidenceIndexingService | None = None,
        report_pipeline: ResearchReportPipeline | None = None,
        aspect_coverage_assessor: AspectCoverageAssessor | None = None,
    ):
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.rag_pipeline_service = rag_pipeline_service
        self.retrieval_pipeline = retrieval_pipeline
        self.indexing_service = indexing_service
        self.report_pipeline = report_pipeline
        self.aspect_coverage_assessor = (
            aspect_coverage_assessor or AspectCoverageAssessor()
        )

    def create_agent(self, research_question: str) -> ResearchAgent:
        """Create a research agent for the given question."""
        return ResearchAgent(
            research_question=research_question,
            max_steps=self.max_steps,
        )

    def plan_research(self, research_question: str):
        """Generate a structured research plan."""
        return generate_research_plan(research_question)


    def execute_research(self, research_question: str) -> ResearchAgent:
        """Generate a research plan and execute research tools."""
        agent = self.create_agent(research_question)
        agent.start()

        try:
            plan = self.plan_research(research_question)
            agent.state.research_plan = plan

            agent.record_step(
                action="Generate research plan",
                tool_name="research_planner",
                tool_input={"research_question": research_question},
                tool_output=str(plan),
                status="completed",
            )

            academic_queries = [
                search_query.query
                for search_query in plan.search_queries
                if search_query.source_type == "academic"
            ]

            if len(academic_queries) >= 2:
                self.execute_tool(
                    agent,
                    "academic_search",
                    {"queries": academic_queries},
                )

            retrieval_output = self.execute_tool(
                agent,
                "rag_retrieval",
                {"query": research_question},
            )

            # Preserve retrieved evidence.
            agent.state.steps[-1].tool_output = str(retrieval_output)

            if self.rag_pipeline_service is not None:
                answer = self.generate_answer(research_question, retrieval_output)

                agent.record_step(
                    action="Generate grounded answer",
                    tool_name="answer_generation",
                    tool_input={"research_question": research_question},
                    tool_output=answer,
                    status="completed",
                )

                agent.complete(answer)


        except Exception as exc:
            if agent.state.status == "running":
                agent.fail(str(exc))

        return agent


    def execute_full_research(self, research_question: str) -> ResearchAgent:
        """Execute the complete research and report generation workflow."""
        agent = self.create_agent(research_question)
        agent.start()

        try:
            # 1. Planning
            plan = self.plan_research(research_question)
            agent.state.research_plan = plan

            agent.record_step(
                action="Generate research plan",
                tool_name="research_planner",
                tool_input={"research_question": research_question},
                tool_output=str(plan),
                status="completed",
            )

            # Validate required dependencies
            if self.retrieval_pipeline is None:
                raise RuntimeError("Retrieval pipeline is not configured.")

            if self.indexing_service is None:
                raise RuntimeError("Indexing service is not configured.")

            if self.rag_pipeline_service is None:
                raise RuntimeError("RAG pipeline service is not configured.")

            if self.report_pipeline is None:
                raise RuntimeError("Report pipeline is not configured.")

            # 2. Academic retrieval and full-text acquisition
            retrieval_result = self.retrieval_pipeline.execute(plan)

            agent.record_step(
                action="Acquire academic papers and evidence",
                tool_name="retrieval_pipeline",
                tool_input={"research_plan": str(plan)},
                tool_output=str(retrieval_result),
                status="completed",
            )

            # 3. Evidence indexing
            indexing_result = self.indexing_service.index_evidence(
                retrieval_result.evidence
            )

            agent.record_step(
                action="Index acquired evidence",
                tool_name="evidence_indexing",
                tool_input={
                    "evidence_count": len(retrieval_result.evidence)
                },
                tool_output=str(indexing_result),
                status="completed",
            )

            if not indexing_result.is_successful:
                raise RuntimeError(
                    "Evidence indexing failed: "
                    f"{indexing_result.errors}"
                )

            # 4. Semantic retrieval
            rag_result = self.rag_pipeline_service.execute(
                RAGPipelineRequest(
                    query=research_question,
                    top_k=10,
                    generate_answer=False,
                )
            )

            agent.record_step(
                action="Retrieve indexed evidence",
                tool_name="semantic_retrieval",
                tool_input={"query": research_question, "top_k": 10},
                tool_output=str(rag_result.retrieval_response),
                status="completed",
            )

            # 5. Assess research aspect coverage
            evidence_texts = [
                result.record.text
                for result in rag_result.retrieval_response.results
            ]

            covered_aspects = self.aspect_coverage_assessor.assess(
                aspects=plan.sub_questions,
                evidence_texts=evidence_texts,
            )

            agent.record_step(
                action="Assess research aspect coverage",
                tool_name="aspect_coverage",
                tool_input={"aspects": plan.sub_questions},
                tool_output=str(covered_aspects),
                status="completed",
            )

            # 6. Evaluate evidence and generate report
            report_result = self.report_pipeline.run(
                research_question=research_question,
                retrieved_results=rag_result.retrieval_response.results,
                papers=retrieval_result.papers,
                context=rag_result.context,
                covered_aspects=covered_aspects,
                all_aspects=plan.sub_questions,
            )

            agent.record_step(
                action="Evaluate evidence and generate research report",
                tool_name="research_report_pipeline",
                tool_input={"research_question": research_question},
                tool_output=str(report_result.report),
                status="completed",
            )

            # 7. Preserve structured result and complete agent
            agent.research_result = report_result

            final_report = str(report_result.report)
            agent.complete(final_report)

        except Exception as exc:
            if agent.state.status == "running":
                agent.fail(str(exc))

        return agent


    def execute_tool(
        self,
        agent: ResearchAgent,
        tool_name: str,
        tool_input: dict,
    ):
        """Validate and execute a registered research tool."""

        if agent.state.is_terminal:
            raise RuntimeError(
                "Cannot execute tools for a terminal agent."
            )

        try:
            tool = self.tool_registry.get(tool_name)
            arguments = tool.args_schema.model_validate(tool_input)

            output = tool.execute(arguments)

            agent.record_step(
                action=f"Execute tool: {tool_name}",
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=str(output),
                status="completed",
            )

            return output

        except Exception as exc:
            agent.record_step(
                action=f"Execute tool: {tool_name}",
                tool_name=tool_name,
                tool_input=tool_input,
                status="failed",
                error=str(exc),
            )
            agent.fail(str(exc))
            raise

    def generate_answer(
        self,
        research_question: str,
        retrieval_output,
    ) -> str:
        """Generate a grounded answer from retrieved evidence."""
        if self.rag_pipeline_service is None:
            raise RuntimeError("RAG pipeline service is not configured.")

        context = self.rag_pipeline_service.context_builder.build(
            retrieval_output.results
        )

        answer = self.rag_pipeline_service.answer_generation_service.generate(
            query=research_question,
            context=context,
        )

        return answer
