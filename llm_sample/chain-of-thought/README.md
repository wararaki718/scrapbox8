# chain of thought

## setup

install `claude-desktop` and `npx`.
set `github mcp` to `claude_desktop_config.json`.

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your_github_access_token"
      }
    }
  }
}
```

## sample output

- https://github.com/wararaki718/scrapbox8/pull/2
- https://github.com/wararaki718/scrapbox8/issues/3
