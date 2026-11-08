# ResearchPilot

ResearchPilot is an evidence-driven AI research assistant that investigates research questions using academic papers and web sources, then generates structured reports with verifiable citations.

## Core Capabilities

### Implemented
- LLM integration using OpenAI APIs
- Reliable LLM client with timeout and retry handling
- API error classification and response validation
- Token usage tracking
- Research question decomposition into focused sub-questions
- Structured research plan generation using Pydantic
- Purpose-driven search queries with source-type classification
- Academic-first source prioritization
- Research strategy generation

### Planned
- Multi-source research and tool calling
- Academic and web evidence retrieval
- Retrieval-augmented generation (RAG)
- Semantic search and evidence ranking
- Evidence-grounded reports with verifiable citations
- Source evaluation and citation validation
- Retrieval and answer quality evaluation

## Project Status

- **Phase 1: Project Foundation** - Completed
- **Phase 2: Core LLM Engineering** - In Progress
  - LLM client architecture and API reliability - Completed
  - Structured output and research plan engineering - Completed
- **Phase 3: Retrieval Engineering** - Planned
- **Phase 4: Evidence Processing and RAG** - Planned
- **Phase 5: Research Report Generation** - Planned
- **Phase 6: Evaluation and Application Integration** - Planned

## Technology Stack

### Implemented
- Python
- OpenAI API
- Pydantic
- pytest

### Planned
- Academic and web search integrations
- Embeddings and FAISS
- Retrieval and evidence ranking
- FastAPI
- Streamlit

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install project dependencies.
4. Configure API keys using environment variables.
5. Run tests to verify the setup.

## Security

Never commit API keys, credentials, or private research documents.
