import os
from google import genai  # 最新の google-genai を使用
from datasets import Dataset
from ragas import evaluate
# インポートパスを .collections ではなく .metrics から直接に変更
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings


def main() -> None:
    # 1. 評価用データの準備 (変更なし)
    data = {
        "question": [
            "Qdrantでのベクトル検索のメリットは何ですか？",
            "SPLADEとはどのような手法ですか？"
        ],
        "answer": [
            "Qdrantは高速なベクトル検索が可能で、フィルタリング機能も強力です。",
            "SPLADEはスパースベクトルを用いた情報検索手法で、キーワードの重要度を考慮します。"
        ],
        "contexts": [
            ["Qdrantはスケーラビリティに優れたベクトルデータベースであり、HNSWインデックスにより高速な検索を実現します。"],
            ["SPLADE (Sparse Lexical and Expansion) は、BERTを用いて文書をスパースな語彙表現に変換し、検索精度を高めます。"]
        ],
        "ground_truth": [
            "Qdrantのメリットは高速なHNSW検索、スケーラビリティ、速度、高度なフィルタリング機能です。",
            "SPLADEはBERTを活用して文書をスパースなベクトル空間にマッピングする手法です。"
        ]
    }

    dataset = Dataset.from_dict(data)

    # 2. クライアントとLLMの設定
    api_key = os.getenv("GEMINI_API_KEY")
    
    # 【重要】ChatGoogleGenerativeAI ではなく google.genai.Client を使う
    native_client = genai.Client(api_key=api_key)
    
    # llm_factory にネイティブクライアントを渡す
    evaluator_llm = llm_factory(
        model="gemini-2.0-flash", # 2026年現在の安定版。liteが必要なら適宜変更
        provider="google", 
        client=native_client
    )
    
    # Embedding の設定
    evaluator_embeddings = GoogleEmbeddings(api_key=api_key, model="models/text-embedding-004")

    # 3. 評価指標の設定
    # AnswerRelevancy に embeddings を渡し、ContextPrecision にも渡すのが一般的です
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm)
    ]

    # 4. 評価の実行
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # 5. 結果の表示
    print(result)
    df = result.to_pandas()
    print(df.head())


if __name__ == "__main__":
    main()
