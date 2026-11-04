#!/usr/bin/env sh
# Install convoviz on Linux or macOS:
#   curl -LsSf https://raw.githubusercontent.com/mohamed-chs/convoviz/main/install.sh | sh
# It installs uv (https://docs.astral.sh/uv/) if needed, then convoviz as a uv tool.
set -eu

if ! command -v uv >/dev/null 2>&1; then
	echo "Installing uv..."
	if command -v curl >/dev/null 2>&1; then
		curl -LsSf https://astral.sh/uv/install.sh | sh
	else
		wget -qO- https://astral.sh/uv/install.sh | sh
	fi
	PATH="$HOME/.local/bin:$PATH"
fi

uv tool install --upgrade convoviz
echo
echo "Done. Open a new terminal and run: convoviz"
