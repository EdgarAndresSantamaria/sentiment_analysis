# Add default vscode settings if not existing
SETTINGS_FILE=./.vscode/settings.json
SETTINGS_TEMPLATE_FILE=./.vscode/settings.default.json
echo "Copy SETTINGS_TEMPLATE_FILE"
cp "$SETTINGS_TEMPLATE_FILE" "$SETTINGS_FILE"
rm -rf ./.venv
uv venv --clear .venv
source ./.venv/bin/activate
echo "Build venv"
uv pip install -r ./requirements.txt
