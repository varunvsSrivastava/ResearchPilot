
"""Research agent orchestration dependencies."""

from researchpilot.agent.research_agent import ResearchAgent
from researchpilot.planner.research_planner import generate_research_plan
from researchpilot.tools.registry import ToolRegistry
from researchpilot.rag.pipeline.service import RAGPipelineService

class ResearchOrchestrator:
    """Coordinate planning and research tool execution."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
        rag_pipeline_service: RAGPipelineService | None = None,
    ):
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.rag_pipeline_service = rag_pipeline_service

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