"""生成されたPyTorchモデルのテスト."""

import importlib.util
from pathlib import Path
from typing import Any, cast

import pytest
import torch


def load_model_from_file(file_path: Path, class_name: str) -> type[torch.nn.Module]:
    """指定されたパスのPythonファイルから動的にクラスを読み込みます."""
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

    return cast(type[torch.nn.Module], ModelClass)


@pytest.fixture(scope="module")
def model_info() -> dict[str, Any]:
    """テスト対象のモデル情報をpytestのフィクスチャとして提供します."""
    return {
        "file_path": Path("generated_models/Model_1706_03762.py"),
        "class_name": "Model_1706_03762",
        "input_shape": (2, 128),
        "model_params": {
            "d_model": 512,
            "nhead": 8,
            "num_encoder_layers": 6,
            "dim_feedforward": 2048,
        },
        "vocab_size": 10000,
        "expected_output_shape": (2, 128, 512),
    }


def test_model_execution(model_info: dict[str, Any]) -> None:
    """モデルの読み込み、実行、出力形状の検証を行います."""
    # Skip if model file doesn't exist
    if not model_info["file_path"].exists():
        pytest.skip(f"Model file not found: {model_info['file_path']}")

    # 1. モデルクラスを動的に読み込み
    try:
        ModelClass = load_model_from_file(
            model_info["file_path"], model_info["class_name"]
        )
        print(
            f"\nSuccessfully loaded model class '{model_info['class_name']}' ."
        )
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
    dummy_input = torch.randint(
        0,
        model_info["vocab_size"],
        model_info["input_shape"],
    ).to(device)
    print(f"Created dummy input with shape: {dummy_input.shape}")

    # 5. フォワードパスの実行と出力形状の検証
    try:
        with torch.no_grad():
            output = model(dummy_input)

        print(f"Model executed successfully. Output shape: {output.shape}")

        expected_shape = model_info["expected_output_shape"]
        assert output.shape == expected_shape, (
            f"Output shape mismatch! Expected {expected_shape}, "
            f"but got {output.shape}"
        )

        print("Output shape is correct.")

    except Exception as e:
        pytest.fail(f"An error occurred during the model's forward pass: {e}")
