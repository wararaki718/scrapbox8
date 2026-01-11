import os
from mcp.server.fastmcp import FastMCP

# "MemoAssistant" という名前のMCPサーバーを作成
mcp = FastMCP("MemoAssistant")

# 読み書きを許可するディレクトリ（適宜変更してください）
# MEMO_DIR = "./my_memos"
# os.makedirs(MEMO_DIR, exist_ok=True)

# @mcp.tool()
# def read_memo(filename: str) -> str:
#     """指定されたメモファイルの内容を読み取ります。"""
#     path = os.path.join(MEMO_DIR, filename)
#     if not os.path.exists(path):
#         return f"エラー: {filename} が見つかりません。"
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read()


# @mcp.tool()
# def save_memo(filename: str, content: str) -> str:
#     """指定された内容をメモファイルとして保存します。"""
#     path = os.path.join(MEMO_DIR, filename)
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(content)
#     return f"{filename} に保存しました。"


# if __name__ == "__main__":
#     mcp.run()

import sys

# 起動時に標準エラー出力にメッセージを出す（これは切断の原因にならない）
print("--- MCP Server Starting ---", file=sys.stderr)

mcp = FastMCP("MemoAssistant")

@mcp.tool()
def echo_test(text: str) -> str:
    return f"Echo: {text}"

if __name__ == "__main__":
    mcp.run()