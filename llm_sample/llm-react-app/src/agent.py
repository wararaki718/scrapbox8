from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from models import get_model
from tools.github_tools import get_repository_info, analyze_code_structure

def get_agent():
    model = get_model()
    tools = [get_repository_info, analyze_code_structure]
    checkpointer = MemorySaver()
    
    system_prompt = "あなたは Chain of Thought を用いて GitHub 操作を行うシニアエンジニアです"
    
    agent = create_react_agent(
        model, 
        tools=tools, 
        checkpointer=checkpointer,
        prompt=system_prompt
    )
    return agent
