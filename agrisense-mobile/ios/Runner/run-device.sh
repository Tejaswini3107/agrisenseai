#!/usr/bin/env bash
set -euo pipefail

# Runner script to prepare and build the iOS app for a connected device.
# Usage:
#   ./run-device.sh <device-udid>
# or
#   IOS_UDID=<device-udid> ./run-device.sh
# If no UDID is provided the script will open the workspace in Xcode for GUI signing.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[runner] Working directory: $ROOT_DIR"

echo "[runner] Installing CocoaPods dependencies..."
if ! command -v pod >/dev/null 2>&1; then
  echo "CocoaPods not found. Install with: sudo gem install cocoapods or brew install cocoapods"
  exit 1
fi
pod install --repo-update

UDID="${1:-${IOS_UDID:-}}"
if [ -z "$UDID" ]; then
  echo "[runner] No device UDID provided. Opening Xcode workspace for GUI run so you can manage signing/provisioning."
  open -a Xcode AgriSenseMobile.xcworkspace
  echo "Then press Run ▶ in Xcode with your device connected."
  exit 0
fi

echo "[runner] Building and installing to device $UDID"
set -x
DERIVED_DATA_DIR="$ROOT_DIR/build/derived-data"
mkdir -p "$DERIVED_DATA_DIR"
# Build using an explicit derived data path so we can find the .app easily
xcodebuild -workspace AgriSenseMobile.xcworkspace -scheme AgriSenseMobile -configuration Debug -destination "id=$UDID" -derivedDataPath "$DERIVED_DATA_DIR" -allowProvisioningUpdates clean build
set +x

APP_PATH="$DERIVED_DATA_DIR/Build/Products/Debug-iphoneos/AgriSenseMobile.app"
if [ -d "$APP_PATH" ]; then
  echo "[runner] Built .app at: $APP_PATH"
  if command -v ios-deploy >/dev/null 2>&1; then
    echo "[runner] Installing to device via ios-deploy..."
    set -x
    ios-deploy --id "$UDID" --bundle "$APP_PATH" --justlaunch
    set +x
    echo "[runner] ios-deploy finished. Check device for the app."
  else
    echo "[runner] ios-deploy not found. To install the app on device run:"
    echo "  brew install ios-deploy"
    echo "  ios-deploy --id $UDID --bundle \"$APP_PATH\" --justlaunch"
    echo "Or open the workspace in Xcode and press Run ▶ to install the built app."
  fi
else
  echo "[runner] ERROR: expected .app not found at: $APP_PATH"
  echo "Check xcodebuild output above for errors or provisioning problems."
fi

echo "[runner] Build+install finished (check xcodebuild/ios-deploy output above). If signing failed, open the workspace in Xcode and enable automatic signing for your Personal Team, then press Run ▶ to create the provisioning profile." 
