# sample openai agent app

## specification

- See simple specification: requirements.md

## setup

```shell
pip install openai-agents
```

Ollama の準備:

```shell
ollama pull gemma4:e2b
```

Ollama の起動（別ターミナルで実行し、起動したままにする）:

```shell
ollama serve
```

起動確認:

```shell
curl http://127.0.0.1:11434/api/tags
```

## run

```shell
python main.py --input README.md
```

出力ファイルを保存する場合:

```shell
python main.py --input README.md --output guide.md --lang ja --verbose
```
