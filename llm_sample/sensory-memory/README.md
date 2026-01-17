# sensory memory

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
