import os
from google import genai


def test_connection():
    # 環境変数からキーを読み込み
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not set.")
        return

    client = genai.Client(api_key=api_key)

    # 利用可能なモデル一覧を表示
    print("--- Available Models ---")
    for m in client.models.list():
        # generateContent に対応しているモデルだけを表示
        if 'generateContent' in m.supported_actions:
            print(f"Model ID: {m.name}")
    
    # Gemini 2.5 Flash Lite (最新の安定版) でテスト
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="「ターミナルからの接続に成功しました」と英語で出力して。"
    )
    
    print("-" * 30)
    print(f"Gemini Response: {response.text}")
    print("-" * 30)

if __name__ == "__main__":
    test_connection()
