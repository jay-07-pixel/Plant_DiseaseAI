#!/usr/bin/env bash
# Install desktop autostart so PlantDiseaseAI opens when the Pi boots to GUI.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$PROJECT_ROOT/scripts/start_plant_disease_pi.sh"
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/plantdiseaseai.desktop"

chmod +x "$LAUNCHER"
mkdir -p "$AUTOSTART_DIR"

cat >"$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=PlantDiseaseAI
Comment=Auto-start Plant Disease Detection on boot
Exec=$LAUNCHER
Path=$PROJECT_ROOT
Terminal=false
X-GNOME-Autostart-enabled=true
StartupNotify=false
EOF

echo "Installed autostart: $DESKTOP_FILE"
echo "Launcher: $LAUNCHER"
echo
echo "Reboot to test:"
echo "  sudo reboot"
echo
echo "To disable later:"
echo "  rm \"$DESKTOP_FILE\""
