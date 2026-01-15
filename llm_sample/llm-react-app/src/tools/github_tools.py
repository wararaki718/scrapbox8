from langchain_core.tools import tool

@tool
def get_repository_info(repo_name: str) -> str:
    """指定されたリポジトリの概要を返します。"""
    # モック実装
    return f"Repository: {repo_name}\nDescription: A sample repository for GitHub operations testing.\nMain Language: Python"

@tool
def analyze_code_structure(path: str) -> str:
    """指定パスのコード構造を分析します。"""
    # モック実装
    return f"Analyzing structure at {path}...\nFound directories: src, tests, docs.\nKey components detected: ReAct agent, GitHub tools."
