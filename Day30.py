import re

print("---  Building an AI ReAct Framework ---\n")


def calculator_tool(expression):
    """A simple tool that safely evaluates a math expression."""
    print(f"\n[TOOL RUNNING: Calculator] -> Calculating: {expression}")
    try:
        result = eval(expression)
        print(f"[TOOL OUTPUT] -> {result}\n")
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"


tools = {
    "Calculator": calculator_tool
}


def mock_llm_generation(prompt_history):
    """Simulates what a real LLM would generate when following the ReAct prompt."""

    if "Action: Calculator[15 * 4]" not in prompt_history:
        return """Thought: I need to multiply 15 by 4 first, then add 20. I should use the Calculator tool.
Action: Calculator[15 * 4]"""

    elif "Action: Calculator[60 + 20]" not in prompt_history:
        return """Thought: The tool told me 15 * 4 is 60. Now I need to add 20 to that result.
Action: Calculator[60 + 20]"""

    else:
        return """Thought: The tool told me 60 + 20 is 80. I now have the final answer.
Final Answer: 15 multiplied by 4, plus 20 is 80."""


def run_agent(user_question):
    print(f"USER: {user_question}\n")

    prompt = f"""You are an AI Agent. You have access to the following tools: [Calculator]. 
Use the following format:
Thought: think about what to do
Action: ToolName[input]
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Final Answer: the final answer to the original input question

Question: {user_question}
"""

    for step in range(5):
        ai_response = mock_llm_generation(prompt)
        print(f"\033[94m{ai_response}\033[0m")

        prompt += "\n" + ai_response + "\n"

        if "Final Answer:" in ai_response:
            print("\n[SYSTEM] Agent has finished the task.\n")
            break

        action_match = re.search(r"Action: (.*?)\[(.*?)\]", ai_response)
        if action_match:
            tool_name = action_match.group(1)
            tool_input = action_match.group(2)

            if tool_name in tools:
                observation = tools[tool_name](tool_input)
                prompt += f"Observation: {observation}\n"
            else:
                prompt += f"Observation: Tool {tool_name} not found.\n"


run_agent("What is 15 multiplied by 4, and then add 20?")
