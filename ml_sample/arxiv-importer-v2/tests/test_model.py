"""生成されたPyTorchモデルのテスト."""

import importlib.util
import inspect
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch

# `src`ディレクトリをシステムパスに追加
# sys.path.append(str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GENERATED_MODELS_DIR = Path("generated_models")


def load_module_from_file(file_path: Path) -> Any:
    """指定されたファイルパスからPythonモジュールを動的に読み込みます."""
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_generated_model_files() -> list[Path]:
    """生成されたモデルファイルの一覧を検索します."""
    if not GENERATED_MODELS_DIR.exists():
        return []
    return list(GENERATED_MODELS_DIR.glob("Model_*.py"))


@pytest.fixture(scope="module", params=find_generated_model_files())
def model_file(request: Any) -> Path | None:
    """各生成モデルファイルに対するフィクスチャ."""
    param: Path = request.param
    if not param.exists():
        pytest.skip(f"Model file not found: {param}")
    return param


@pytest.fixture(scope="module")
def loaded_model_and_class(
    model_file: Path | None,
) -> tuple[Any, type[torch.nn.Module]] | None:
    """モデルファイルからモジュールとクラスをロードします."""
    if model_file is None:
        pytest.skip("Model file is None.")
        return None
    logger.info("Testing model file: %s", model_file)
    module = load_module_from_file(model_file)
    model_class_name = model_file.stem
    model_class = getattr(module, model_class_name)
    return module, model_class


def get_dummy_input_for_model(
    model_class: type[torch.nn.Module],
) -> dict[str, Any]:
    """モデルのシグネチャに基づいてダミーの入力と引数を生成します."""
    init_signature = inspect.signature(model_class.__init__)
    init_params = init_signature.parameters

    dummy_init_args = {}
    # __init__の引数を設定
    for name, param in init_params.items():
        if name in ("self", "args", "kwargs"):
            continue
        if param.default is inspect.Parameter.empty:
            # デフォルト値がない引数にダミー値を設定
            if "vocab_size" in name:
                dummy_init_args[name] = 1000
            elif "dim" in name or "d_model" in name:
                dummy_init_args[name] = 512
            elif "nhead" in name or "layers" in name:
                dummy_init_args[name] = 4
            else:
                dummy_init_args[name] = 10  # Fallback

    # forwardの引数に基づいてダミー入力テンソルを作成
    forward_signature = inspect.signature(model_class.forward)
    forward_params = forward_signature.parameters
    dummy_forward_args = {}
    batch_size = 2
    seq_len = 128
    for name, param in forward_params.items():
        if name in ("self", "args", "kwargs"):
            continue
        if param.default is inspect.Parameter.empty:
            # デフォルト値のない必須引数に対してテンソルを作成
            if "mask" in name:
                # マスクはNone許容の場合が多いが、ここでは一旦作成しない
                continue
            # 型ヒントや名前に基づいてテンソル形状を推測
            if "input" in name or "src" in name or "tgt" in name:
                vocab_size_key = next(
                    (k for k in dummy_init_args if "vocab_size" in k), ""
                )
                vocab_size = dummy_init_args.get(vocab_size_key, 1000)
                dummy_forward_args[name] = torch.randint(
                    0, vocab_size, (batch_size, seq_len)
                )
            else:
                # デフォルトの画像様データ
                dummy_forward_args[name] = torch.randn(batch_size, 3, 224, 224)

    return {
        "init_args": dummy_init_args,
        "forward_args": dummy_forward_args,
    }


def test_model_forward_pass(
    loaded_model_and_class: tuple[Any, type[torch.nn.Module]] | None,
) -> None:
    """ダミーデータで順伝播（Forward Pass）テストを実行します."""
    if loaded_model_and_class is None:
        pytest.skip("No model to test.")
        return

    _, model_class = loaded_model_and_class

    try:
        dummy_data = get_dummy_input_for_model(model_class)
        init_args = dummy_data["init_args"]
        forward_args = dummy_data["forward_args"]

        # モデルのインスタンス化
        model = model_class(**init_args)
        logger.info("Instantiated %s with args: %s", model_class.__name__, init_args)

        # ダミー入力で順伝播
        output = model(**forward_args)
        assert output is not None
        logger.info("Forward pass successful for %s.", model_class.__name__)
        logger.info("Output shape: %s", output.shape)
    except Exception as e:
        pytest.fail(
            f"Forward pass failed for {model_class.__name__}: {e}\n"
            f"Model __init__ signature: {inspect.signature(model_class.__init__)}\n"
            f"Model forward signature: {inspect.signature(model_class.forward)}"
        )


# このジェネレータは現在使用されていませんが、将来的な拡張のために残しておきます
def model_test_generator() -> Generator[Any, Any, None]:
    """テストケースを動的に生成します（現在は未使用）."""
    model_files = find_generated_model_files()
    for model_file in model_files:
        yield pytest.param(model_file, id=model_file.name)
