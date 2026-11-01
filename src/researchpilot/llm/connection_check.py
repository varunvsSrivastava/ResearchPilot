from researchpilot.llm.client import generate_response

def main():
    prompt = "Reply with exactly: ResearchPilot LLM connection successful."
    
    response = generate_response(prompt)
    
    print("LLM response:")
    print(response)
    
if __name__=="__main__":
    main()