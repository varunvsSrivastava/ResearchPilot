
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
- Academic paper search using OpenAlex
- Paper deduplication and retrieval pipeline
- PDF downloading and text extraction
- Text normalization and chunking
- Evidence generation with source metadata
- OpenAI embedding integration using `text-embedding-3-small`
- FAISS vector store integration
- Cosine similarity-based vector search
- Vector persistence and loading
- Evidence indexing with batch processing
- Evidence ID filtering and duplicate handling
- Indexing failure handling and result tracking

### Planned

- Semantic retrieval pipeline
- Metadata filtering
- Retrieval ranking and optimization
- End-to-end RAG pipeline integration
- Retrieval evaluation
- Multi-source web search integration
- Tool calling and agentic research workflows
- Evidence-grounded report generation
- Citation validation and source evaluation
- Answer quality evaluation

## Project Status

- **Phase 1: Project Foundation** - Completed
- **Phase 2: Core LLM Engineering** - Completed
  - LLM client architecture and API reliability - Completed
  - Structured output and research plan engineering - Completed
- **Phase 3: Academic Search** - Completed
  - OpenAlex integration and academic search - Completed
- **Phase 4: Retrieval Engineering** - Completed
  - PDF acquisition, extraction, normalization, chunking, and evidence generation - Completed
  - Academic retrieval pipeline integration - Completed
- **Phase 5: RAG Engineering** - In Progress
  - 5.1 Embedding Model Integration - Completed
  - 5.2 Vector Store Integration - Completed
  - 5.3 Evidence Indexing - Completed
  - 5.4 Semantic Retrieval - Planned
  - 5.5 Metadata Filtering - Planned
  - 5.6 Retrieval Ranking - Planned
  - 5.7 RAG Pipeline Integration - Planned
  - 5.8 Retrieval Evaluation - Planned
- **Phase 6: Research Report Generation** - Planned
- **Phase 7: Evaluation and Application Integration** - Planned

## Technology Stack

### Implemented

- Python
- OpenAI API
- Pydantic
- pytest
- OpenAlex API
- httpx
- FAISS
- NumPy

### Planned

- Web search integrations
- FastAPI
- Streamlit
- LLM-based report generation
- Agentic orchestration

## Testing

ResearchPilot uses pytest for unit and integration testing.

Latest validation:

- **167 tests passed**

Run the complete test suite:

```bash
pytest
```

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install project dependencies.
4. Configure API keys using environment variables.
5. Run tests to verify the setup.

## Security

Never commit API keys, credentials, or private research documents.
