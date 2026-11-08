from researchpilot.planner.research_planner import (
    generate_research_plan,
)


def main():
    question = "How is generative AI transforming healthcare?"

    research_plan = generate_research_plan(question)

    print("\nGenerate Research Plan\n")
    print(research_plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
