# OpenClaw Installation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Install OpenClaw as a system-wide daemon and configure it to respond to Discord @mentions using OpenRouter's minimax model.

**Architecture:** One-click installer script → Systemd daemon service → Discord bot integration → OpenRouter API gateway → minimax LLM.

**Tech Stack:** OpenClaw (Node.js), OpenRouter API, Discord Bot API, systemd, YAML configuration.

---

## Task 1: Verify System Requirements

**Files:** None (system check)

**Step 1: Check Node.js version**

Run: `node --version`
Expected: v22 or higher (if missing, installer will handle it)

**Step 2: Check available disk space**

Run: `df -h ~`
Expected: At least 1GB free space

**Step 3: Check if systemd is available**

Run: `systemctl --version`
Expected: systemd version displayed (user service support)

**Step 4: Document current state**

Run: `echo "Node: $(node --version), Systemd: $(systemctl --version | head -1), User: $(whoami)" > /tmp/openclaw-preinstall.log`
Expected: Log file created successfully

**Step 5: Commit pre-installation log**

```bash
# Not a git operation - system documentation only
# Log saved to /tmp/openclaw-preinstall.log
```

---

## Task 2: Download and Run OpenClaw Installer

**Files:**
- Create: `/tmp/openclaw-installer.sh`
- Modify: None

**Step 1: Download installer script**

Run: `curl -fsSL https://openclaw.bot/install.sh -o /tmp/openclaw-installer.sh`
Expected: Script downloaded without errors

**Step 2: Review installer script (security check)**

Run: `less /tmp/openclaw-installer.sh`
Expected: Script appears safe, no malicious commands

**Step 3: Make script executable**

Run: `chmod +x /tmp/openclaw-installer.sh`
Expected: No output (permissions changed)

**Step 4: Run installer with Tsinghua proxy support**

Run: `http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 bash /tmp/openclaw-installer.sh`
Expected: Installation begins, interactive prompts appear

**Step 5: Verify installation**

Run: `openclaw --version`
Expected: Version number displayed (e.g., "OpenClaw v1.0.0")

**Step 6: Commit installation verification**

```bash
echo "OpenClaw $(openclaw --version) installed successfully" >> /tmp/openclaw-preinstall.log
```

---

## Task 3: Create Configuration Directory Structure

**Files:**
- Create: `~/.config/openclaw/`
- Create: `~/.local/state/openclaw/logs/`

**Step 1: Create config directory**

Run: `mkdir -p ~/.config/openclaw`
Expected: Directory created (no error)

**Step 2: Create logs directory**

Run: `mkdir -p ~/.local/state/openclaw/logs`
Expected: Directory created

**Step 3: Set proper permissions on config directory**

Run: `chmod 700 ~/.config/openclaw`
Expected: No output (permissions set)

**Step 4: Verify directory structure**

Run: `ls -la ~/.config/openclaw ~/.local/state/openclaw/logs`
Expected: Both directories exist with drwx permissions

---

## Task 4: Create OpenClaw Configuration File

**Files:**
- Create: `~/.config/openclaw/config.yml`
- Backup: `~/.config/openclaw/config.yml.bak`

**Step 1: Create configuration file with all settings**

Run:
```bash
cat > ~/.config/openclaw/config.yml << 'EOF'
# OpenClaw Configuration for NN_SPICE Project
# Discord Bot Integration with OpenRouter API

discord:
  bot_token: "REDACTED_DISCORD_BOT_TOKEN"
  server_ids:
    - "1469701279445159956"
  channel_ids:
    - "1469594375964328080"
  command_type: "mention"
  max_message_length: 2000

openrouter:
  api_key: "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"
  model: "minimax/minimax-m2.5"
  api_endpoint: "https://openrouter.ai/api/v1"
  timeout: 30
  max_retries: 3

daemon:
  auto_start: true
  log_level: "info"
  max_concurrent_requests: 5
  request_timeout: 30
  health_check_interval: 60

logging:
  file: "~/.local/state/openclaw/logs/openclaw.log"
  level: "info"
  max_size: "100MB"
  max_backups: 5
  max_age: 30

features:
  persistent_memory: true
  file_processing: true
  code_execution: false
  web_search: false
EOF
```
Expected: File created successfully

**Step 2: Secure the configuration file**

Run: `chmod 600 ~/.config/openclaw/config.yml`
Expected: No output (only owner can read/write)

**Step 3: Create backup**

Run: `cp ~/.config/openclaw/config.yml ~/.config/openclaw/config.yml.bak`
Expected: Backup created

**Step 4: Verify configuration syntax**

Run: `openclaw config validate`
Expected: "Configuration is valid" or similar message

**Step 5: Test configuration loading**

Run: `openclaw config show`
Expected: YAML configuration displayed (except sensitive values masked)

---

## Task 5: Create Systemd User Service File

**Files:**
- Create: `~/.config/systemd/user/openclaw.service`

**Step 1: Create systemd service unit file**

Run:
```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/openclaw.service << 'EOF'
[Unit]
Description=OpenClaw AI Discord Bot Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/openclaw daemon --config ~/.config/openclaw/config.yml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=openclaw

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=%h/.config/openclaw %h/.local/state/openclaw

# Environment
Environment="NODE_ENV=production"
Environment="OPENCLAW_CONFIG_PATH=%h/.config/openclaw/config.yml"

[Install]
WantedBy=default.target
EOF
```
Expected: Service file created

**Step 2: Verify service file syntax**

Run: `systemctl --user daemon-reload`
Expected: No errors (daemon reloads successfully)

**Step 3: Check if service is recognized**

Run: `systemctl --user status openclaw`
Expected: Service loaded but inactive (not started yet)

---

## Task 6: Enable and Start OpenClaw Service

**Files:**
- Modify: None (systemd operations)

**Step 1: Reload systemd daemon**

Run: `systemctl --user daemon-reload`
Expected: No output (daemon reloaded)

**Step 2: Enable OpenClaw service (auto-start on login)**

Run: `systemctl --user enable openclaw`
Expected: "Created symlink /home/shenshan/.config/systemd/user/default.target.wants/openclaw.service"

**Step 3: Start OpenClaw service**

Run: `systemctl --user start openclaw`
Expected: No output (service starts in background)

**Step 4: Wait 5 seconds for service initialization**

Run: `sleep 5`
Expected: Pause for 5 seconds

**Step 5: Check service status**

Run: `systemctl --user status openclaw`
Expected: "active (running)" with green indicator

**Step 6: View recent logs**

Run: `journalctl --user -u openclaw -n 20 --no-pager`
Expected: Log messages showing Discord connection attempt, API initialization

---

## Task 7: Verify Discord Bot Connection

**Files:**
- Modify: None (verification tests)

**Step 1: Check Discord connection in logs**

Run: `journalctl --user -u openclaw -b --no-pager | grep -i discord`
Expected: Messages like "Connected to Discord gateway" or "WebSocket connected"

**Step 2: Verify bot token authentication**

Run: `journalctl --user -u openclaw -b --no-pager | grep -i "authentication\|token\|ready"`
Expected: "Bot authenticated successfully" or "READY payload received"

**Step 3: Check if bot is in specified server**

Run: `journalctl --user -u openclaw -b --no-pager | grep "1469701279445159956"`
Expected: Server ID mentioned in logs, "Guild available" message

**Step 4: Verify channel monitoring**

Run: `journalctl --user -u openclaw -b --no-pager | grep "1469594375964328080"`
Expected: Channel ID mentioned, "Channel subscribed" or similar

---

## Task 8: Test OpenRouter API Integration

**Files:**
- Modify: None (API testing)

**Step 1: Check OpenRouter API initialization in logs**

Run: `journalctl --user -u openclaw -b --no-pager | grep -i openrouter`
Expected: "OpenRouter API initialized" or "API endpoint: https://openrouter.ai/api/v1"

**Step 2: Verify API key is loaded (not logged, should be masked)**

Run: `journalctl --user -u openclaw -b --no-pager | grep -i "api.*key"`
Expected: Key is NOT shown in logs, appears as "sk-or-v1-***" or "[REDACTED]"

**Step 3: Check model configuration**

Run: `journalctl --user -u openclaw -b --no-pager | grep "minimax"`
Expected: "Model: minimax/minimax-m2.5" or similar

**Step 4: Test API connectivity (if OpenClaw has test command)**

Run: `openclaw test api`
Expected: "API connection successful" or similar (if command exists)

---

## Task 9: Send Test Message to Discord Bot

**Files:**
- None (manual Discord test)

**Step 1: Open Discord and navigate to the specified server**

Action: User manually opens Discord app or web browser
Expected: Server with ID 1469701279445159956 is accessible

**Step 2: Navigate to the specified channel**

Action: User navigates to channel ID 1469594375964328080
Expected: Channel is visible and accessible

**Step 3: Send a test message with bot mention**

Action: User types "@OpenClaw hello, this is a test message" in the channel
Expected: Message is sent successfully

**Step 4: Wait for bot response**

Action: Wait 10-30 seconds for API response
Expected: Bot responds with AI-generated message

**Step 5: Check logs for request processing**

Run: `journalctl --user -u openclaw -f --no-pager`
Expected: New log entries appear showing message received, API call made, response sent

**Step 6: Verify bot response contains minimax model output**

Action: Check if response is coherent and appropriate
Expected: Bot responds intelligently to the test message

---

## Task 10: Verify Persistent Functionality

**Files:**
- Modify: None (long-term testing)

**Step 1: Send multiple test messages in quick succession**

Action: Send 3-5 messages spaced 5 seconds apart
Expected: Bot responds to all messages (may queue requests)

**Step 2: Check for rate limiting in logs**

Run: `journalctl --user -u openclaw -b --no-pager | grep -i "rate.*limit\|queue"`
Expected: No errors, requests handled appropriately

**Step 3: Verify service auto-restart capability**

Run: `systemctl --user kill -s SIGTERM openclaw && sleep 5 && systemctl --user status openclaw`
Expected: Service automatically restarts (Restart=always in service file)

**Step 4: Check service is still running after restart**

Run: `systemctl --user status openclaw`
Expected: Service is "active (running)" again

**Step 5: Send another test message after restart**

Action: Send "@OpenClaw are you still working?" in Discord
Expected: Bot responds normally after restart

---

## Task 11: Create Documentation for NN_SPICE Project

**Files:**
- Create: `docs/openclaw-setup.md`
- Modify: `CLAUDE.md` (optional: add OpenClaw section)

**Step 1: Create setup documentation**

Run:
```bash
cat > /home/shenshan/NN_SPICE/docs/openclaw-setup.md << 'EOF'
# OpenClaw Discord Bot Setup

## Overview

This project uses OpenClaw as an AI assistant integrated into Discord for automated responses using OpenRouter's minimax model.

## Installation

OpenClaw is installed as a system-wide daemon using the one-click installer:

```bash
curl -fsSL https://openclaw.bot/install.sh | bash
```

## Configuration

Configuration file: `~/.config/openclaw/config.yml`

### Discord Integration
- **Server ID:** 1469701279445159956
- **Channel ID:** 1469594375964328080
- **Command Type:** Mention (@OpenClaw)

### AI Model
- **Provider:** OpenRouter
- **Model:** minimax/minimax-m2.5 (cost-optimized)
- **API Endpoint:** https://openrouter.ai/api/v1

## Service Management

### Check Status
```bash
systemctl --user status openclaw
```

### View Logs
```bash
journalctl --user -u openclaw -f
```

### Restart Service
```bash
systemctl --user restart openclaw
```

### Stop Service
```bash
systemctl --user stop openclaw
```

### Disable Auto-Start
```bash
systemctl --user disable openclaw
```

## Usage

In the configured Discord channel, mention the bot:

```
@OpenClaw What is the current status of the NN_SPICE project?
```

The bot will respond using the minimax model via OpenRouter API.

## Troubleshooting

### Bot not responding
1. Check service status: `systemctl --user status openclaw`
2. View logs: `journalctl --user -u openclaw -n 50`
3. Verify API keys in `~/.config/openclaw/config.yml`
4. Check Discord bot is online in Discord app

### API errors
1. Verify OpenRouter API key is valid
2. Check OpenRouter service status: https://status.openrouter.ai
3. Review rate limits in OpenRouter dashboard

### Connection issues
1. Check internet connectivity
2. Verify proxy settings (http_proxy, https_proxy)
3. Test Discord API connectivity

## Configuration Backup

Configuration is backed up to: `~/.config/openclaw/config.yml.bak`

To restore:
```bash
cp ~/.config/openclaw/config.yml.bak ~/.config/openclaw/config.yml
systemctl --user restart openclaw
```

## Security Notes

- Configuration file permissions: 600 (owner read/write only)
- API keys are never logged
- Bot token should be rotated if compromised
- Regular security updates: `npm update -g openclaw`

## Related Documentation

- [OpenClaw Official Docs](https://clawd.bot/docs)
- [OpenRouter API Docs](https://openrouter.ai/docs)
- [Discord Bot Guide](https://discord.com/developers/docs/intro)
EOF
```
Expected: Documentation file created

**Step 2: Update CLAUDE.md with OpenClaw reference (optional)**

Run: `grep -q "OpenClaw" /home/shenshan/NN_SPICE/CLAUDE.md || echo "" >> /home/shenshan/NN_SPICE/CLAUDE.md`
Expected: No error (file exists)

**Step 3: Commit documentation**

```bash
git add docs/openclaw-setup.md
git commit -m "docs: add OpenClaw Discord bot setup documentation

Comprehensive setup guide for OpenClaw integration:
- Installation and configuration
- Service management commands
- Usage instructions and troubleshooting
- Security best practices

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```
Expected: Git commit successful

---

## Task 12: Final Verification and Cleanup

**Files:**
- Delete: `/tmp/openclaw-installer.sh`
- Archive: `/tmp/openclaw-preinstall.log`

**Step 1: Run final service health check**

Run: `openclaw status` (if available) or `systemctl --user is-active openclaw`
Expected: "active" or "running"

**Step 2: Check disk usage**

Run: `du -sh ~/.local/share/openclaw ~/.config/openclaw ~/.local/state/openclaw`
Expected: Reasonable disk usage (< 500MB)

**Step 3: Verify log rotation is working**

Run: `ls -lh ~/.local/state/openclaw/logs/`
Expected: Log files present, not excessively large

**Step 4: Clean up temporary files**

Run: `rm -f /tmp/openclaw-installer.sh`
Expected: Temporary installer script removed

**Step 5: Archive pre-installation log**

Run: `mv /tmp/openclaw-preinstall.log ~/.local/state/openclaw/logs/`
Expected: Log moved to permanent location

**Step 6: Create final summary**

Run:
```bash
cat > ~/.local/state/openclaw/logs/installation-summary.txt << 'EOF'
OpenClaw Installation Summary
=============================
Date: $(date)
Version: $(openclaw --version)
Service Status: $(systemctl --user is-active openclaw)
Config File: ~/.config/openclaw/config.yml
Service File: ~/.config/systemd/user/openclaw.service

Discord Integration:
- Server ID: 1469701279445159956
- Channel ID: 1469594375964328080
- Command: @mention

OpenRouter Integration:
- Model: minimax/minimax-m2.5
- API: https://openrouter.ai/api/v1

Logs: ~/.local/state/openclaw/logs/
Documentation: /home/shenshan/NN_SPICE/docs/openclaw-setup.md
EOF
```
Expected: Summary file created

**Step 7: Display summary to user**

Run: `cat ~/.local/state/openclaw/logs/installation-summary.txt`
Expected: Installation summary displayed

---

## Success Criteria

All tasks completed successfully when:
- [ ] OpenClaw daemon is running (`systemctl --user status openclaw` shows active)
- [ ] Discord bot is online in specified server/channel
- [ ] Bot responds to @mentions with AI-generated messages
- [ ] Logs show successful API calls to OpenRouter
- [ ] Service auto-starts on login/reboot
- [ ] Documentation is created and committed to git
- [ ] Configuration file is secured (chmod 600)

## Rollback Procedure

If installation fails:

1. **Stop and disable service:**
   ```bash
   systemctl --user stop openclaw
   systemctl --user disable openclaw
   ```

2. **Remove service file:**
   ```bash
   rm ~/.config/systemd/user/openclaw.service
   systemctl --user daemon-reload
   ```

3. **Uninstall OpenClaw:**
   ```bash
   npm uninstall -g openclaw
   # Or: openclaw uninstall (if command exists)
   ```

4. **Remove configuration and logs:**
   ```bash
   rm -rf ~/.config/openclaw ~/.local/state/openclaw ~/.local/share/openclaw
   ```

5. **Restore from backup (if needed):**
   Configuration backup: `~/.config/openclaw/config.yml.bak`

## References

- OpenClaw GitHub: https://github.com/openclaw/openclaw
- OpenClaw Installation: https://openclaw.bot/install.sh
- OpenRouter API: https://openrouter.ai/docs
- Discord Developer Portal: https://discord.com/developers/applications
- Systemd Service Units: https://www.freedesktop.org/software/systemd/man/systemd.service.html
