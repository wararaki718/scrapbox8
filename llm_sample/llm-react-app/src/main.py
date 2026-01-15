import sys
import os

from agent import get_agent

def main():
    agent = get_agent()
    
    query = "リポジトリ scrapbox8 の構造を分析し、README の改善案を出してください。"
    config = {"configurable": {"thread_id": "1"}}
    
    print(f"--- Query ---\n{query}\n")
    
    # 思考プロセスと最終回答を出力するためにストリーミングを使用
    for chunk in agent.stream({"messages": [("user", query)]}, config=config, stream_mode="values"):
        if "messages" in chunk:
            last_message = chunk["messages"][-1]
            last_message.pretty_print()

if __name__ == "__main__":
    main()
