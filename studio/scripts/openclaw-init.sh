#!/bin/sh

set -eu

has_openclaw_cli() {
  command -v openclaw >/dev/null 2>&1
}

ensure_openclaw_config_exists() {
  CONFIG_PATH="${HOME}/.openclaw/openclaw.json"
  mkdir -p "$(dirname "$CONFIG_PATH")"

  if [ ! -s "$CONFIG_PATH" ]; then
    printf '{}\n' > "$CONFIG_PATH"
  fi
}

resolve_installed_dip_plugin_dir() {
  PLUGIN_INFO_OUTPUT="$(openclaw plugins info dip --json 2>&1 || true)"

  printf '%s' "$PLUGIN_INFO_OUTPUT" \
    | node -e '
        let input = "";
        process.stdin.on("data", chunk => (input += chunk));
        process.stdin.on("end", () => {
          const start = input.indexOf("{");
          if (start < 0) {
            process.exit(1);
          }

          let payload;
          try {
            payload = JSON.parse(input.slice(start));
          } catch {
            process.exit(1);
          }

          if (typeof payload.source !== "string" || payload.source.length === 0) {
            process.exit(1);
          }

          process.stdout.write(require("node:path").dirname(payload.source));
        });
      '
}

install_dip_plugin() {
  if ! has_openclaw_cli; then
    echo "openclaw CLI is unavailable, skipping dip plugin installation"
    return 0
  fi

  if PLUGIN_INSTALL_DIR="$(resolve_installed_dip_plugin_dir)"; then
    echo "dip plugin already installed, replacing existing installation"
    if [ -n "$PLUGIN_INSTALL_DIR" ]; then
      rm -rf "$PLUGIN_INSTALL_DIR"
    fi
  fi

  echo "installing dip plugin from /app/extensions/dip"
  openclaw plugins install /app/extensions/dip
}

install_feishu_openclaw_skills() {
  if ! has_openclaw_cli; then
    echo "openclaw CLI is unavailable, skipping openclaw-lark installation"
    return 0
  fi

  echo "installing openclaw-lark from /app/extensions/larksuite-openclaw-lark-2026.3.24.tgz"
  openclaw plugins install /app/extensions/larksuite-openclaw-lark-2026.3.24.tgz
}

ensure_openclaw_config_exists
install_dip_plugin || echo "dip plugin installation failed; continuing init flow"
install_feishu_openclaw_skills || echo "openclaw-lark installation failed; continuing init flow"
node /app/scripts/init_agents/index.mjs
