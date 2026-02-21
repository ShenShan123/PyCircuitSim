# OpenClaw Discord Bot Integration Design

**Date:** 2026-02-21
**Author:** Claude Sonnet 4.5
**Status:** Approved

## Overview

Install and configure OpenClaw as a system-wide AI assistant daemon that connects Discord to OpenRouter API using the minimax/minimax-m2.5 model for cost-optimized responses.

## Architecture

### Components

1. **OpenClaw Daemon** - Runs as systemd service, manages AI connections and Discord bot
2. **Discord Bot** - Listens for @mentions in configured channels, forwards messages to OpenClaw
3. **OpenRouter API** - Routes requests to minimax/minimax-m2.5 model (cost-optimized)
4. **Configuration Store** - JSON/YAML config file with API keys, channel IDs, model settings

### Data Flow

```
Discord User (@mention bot) → Discord Bot → OpenClaw Daemon → OpenRouter API → minimax-m2.5 → Response → Discord
```

### System Layout

- **Install location:** `/opt/openclaw/` or `~/.local/share/openclaw/`
- **Config:** `~/.config/openclaw/config.yml`
- **Logs:** `~/.local/state/openclaw/logs/`
- **Service:** systemd user unit

## Configuration

### Discord Bot Configuration

- **Bot Token:** `REDACTED_DISCORD_BOT_TOKEN`
- **Server ID:** `1469701279445159956`
- **Channel ID:** `1469594375964328080`
- **Command Type:** Mention-based (@bot_name)

### OpenRouter API Configuration

- **API Key:** `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`
- **Model:** `minimax/minimax-m2.5` (cost-optimized)
- **API Endpoint:** `https://openrouter.ai/api/v1`

### Config Structure

```yaml
# Discord Bot Configuration
discord:
  bot_token: "REDACTED_DISCORD_BOT_TOKEN"
  server_ids:
    - "1469701279445159956"
  channel_ids:
    - "1469594375964328080"
  command_type: "mention"  # Responds to @bot_name

# OpenRouter API Configuration
openrouter:
  api_key: "sk-or-v1-0a8e88d1a..."
  model: "minimax/minimax-m2.5"
  api_endpoint: "https://openrouter.ai/api/v1"

# Daemon Settings
daemon:
  auto_start: true
  log_level: "info"
  max_concurrent_requests: 5
  timeout_seconds: 30
```

## Implementation Steps

### Phase 1: Install OpenClaw

1. Run one-click installer script: `curl -fsSL https://openclaw.bot/install.sh | bash`
2. Complete onboarding wizard (or skip for manual config)
3. Verify installation: `openclaw --version`

### Phase 2: Configure APIs

1. Create/edit `~/.config/openclaw/config.yml`
2. Add OpenRouter API key
3. Add Discord bot token
4. Set model to `minimax/minimax-m2.5`

### Phase 3: Discord Setup

1. Add server ID `1469701279445159956` to config
2. Add channel ID `1469594375964328080` to config
3. Set command type to "mention"
4. Test bot connection: `openclaw test discord`

### Phase 4: Start Daemon

1. Enable systemd service: `systemctl --user enable openclaw`
2. Start service: `systemctl --user start openclaw`
3. Check logs: `journalctl --user -u openclaw -f`
4. Verify bot responds to @mentions in Discord

### Phase 5: Testing

1. Send test message: "@OpenClaw hello"
2. Verify response from minimax model
3. Check logs for errors
4. Adjust timeout/concurrency settings if needed

## Error Handling & Monitoring

### Error Scenarios

- **API key invalid:** Log error, retry with exponential backoff, notify admin
- **Discord connection lost:** Auto-reconnect with 30s intervals
- **Rate limits (OpenRouter):** Queue requests, implement 1 req/sec limit
- **Model timeout:** Fall back to error message after 30s, log to file

### Monitoring

- **Logs:** `~/.local/state/openclaw/logs/openclaw.log` with rotation
- **Health check:** `openclaw status` command
- **Metrics:** Request count, success rate, avg response time

### Backup/Recovery

- Config backed up to `~/.config/openclaw/config.yml.bak`
- Daemon auto-restart on crash (systemd Restart=always)

## Security Considerations

- API keys stored in user-writable config file (chmod 600)
- Bot token never logged
- Support for environment variable overrides (OPENROUTER_API_KEY, DISCORD_BOT_TOKEN)
- Regular security updates via package manager

## Success Criteria

- [ ] OpenClaw installed and accessible via command line
- [ ] Daemon running as systemd service
- [ ] Discord bot online and responds to @mentions
- [ ] OpenRouter API integration working with minimax model
- [ ] Logs show successful requests/responses
- [ ] Bot responds in configured channel only

## References

- OpenClaw GitHub: https://github.com/openclaw/openclaw
- OpenClaw Docs: https://clawd.bot/docs
- OpenRouter API: https://openrouter.ai/docs
