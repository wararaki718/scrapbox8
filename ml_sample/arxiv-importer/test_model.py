import torch
import pytest
import importlib.util
from pathlib import Path

# このテストファイルの親ディレクトリを取得 (ml_sample/arxiv-importer/)
BASE_DIR = Path(__file__).resolve().parent

@pytest.fixture(scope="module")
def model_info():
    """
    テスト対象のモデル情報をpytestのフィクスチャとして提供する。
    """
    # "Attention Is All You Need" 論文のベースモデルのパラメータ
    # エラーメッセージに合わせてキーを調整
    model_params = {
        "vocab_size": 30000,  # 仮の語彙サイズ
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "ff_dim": 2048,
        "dropout_rate": 0.1
    }
    
    return {
        "file_path": BASE_DIR / "generated_models/Model_1706_03762.py",
        "class_name": "Model_1706_03762",
        "input_shape": (2, 128),  # Transformerの入力は (batch, seq_len) のIDテンソル
        "expected_output_shape": (2, 128, model_params["vocab_size"]), # 出力は (batch, seq_len, embed_dim)
        "model_params": model_params
    }

def load_model_from_file(file_path: Path, class_name: str):
    """
    指定されたパスのPythonファイルから動的にクラスを読み込む。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found at: {file_path}")

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module at {file_path}")
        
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    ModelClass = getattr(module, class_name, None)
    if ModelClass is None:
        raise AttributeError(f"Class '{class_name}' not found in {file_path}")
        
    return ModelClass

def test_model_execution(model_info):
    """
    モデルの読み込み、実行、出力形状の検証を行うpytestテストケース。
    """
    # 1. モデルクラスを動的に読み込み
    try:
        ModelClass = load_model_from_file(model_info["file_path"], model_info["class_name"])
        print(f"\nSuccessfully loaded model class '{model_info['class_name']}'.")
    except (FileNotFoundError, ImportError, AttributeError) as e:
        pytest.fail(f"Failed to load model: {e}")

    # 2. デバイスの決定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. モデルのインスタンス化とデバイスへの移動
    try:
        # フィクスチャからモデルパラメータを取得してインスタンス化
        model_params = model_info.get("model_params", {})
        model = ModelClass(**model_params).to(device)
        model.eval()
        print("Model instantiated with parameters and moved to device.")
    except Exception as e:
        pytest.fail(f"Error during model instantiation: {e}")

    # 4. ダミー入力データの作成 (TransformerはIDのテンソルを入力とする)
    dummy_input = torch.randint(0, model_info["model_params"]["vocab_size"], model_info["input_shape"]).to(device)
    print(f"Created dummy input with shape: {dummy_input.shape}")

    # 5. フォワードパスの実行と出力形状の検証
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Model executed successfully. Output shape: {output.shape}")
        
        assert output.shape == model_info["expected_output_shape"], \
            f"Output shape mismatch! Expected {model_info['expected_output_shape']}, but got {output.shape}"
            
        print("Output shape is correct.")
        
    except Exception as e:
        pytest.fail(f"An error occurred during the model's forward pass: {e}")

# pytestをコマンドラインから実行する場合、この部分は不要ですが、
# スクリプトとして直接実行する際のために残しておきます。
if __name__ == "__main__":
    pytest.main([__file__])

