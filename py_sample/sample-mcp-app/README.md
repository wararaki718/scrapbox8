# sample mcp app

## setup

```shell
pip install mcp
```

## run

download claude desktop and add config.

```json
{
  "mcpServers": {
    "my-memo-tool": {
      "command": "python",
      "args": ["<your working directory>/scrapbox8/py_sample/sample-mcp-app/main.py"]
    }
  }
}
```

after that, launch claude desktop and call this application.

sample input

```text
echo_test ツールを使って、『こんにちは』とテスト送信してみて
```
