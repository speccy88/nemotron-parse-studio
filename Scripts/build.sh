#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
python3 Scripts/generate_project.py
xcodebuild -project NemotronParseStudio.xcodeproj -scheme NemotronParseStudio -configuration Release -derivedDataPath build CODE_SIGN_IDENTITY=- build
printf '\nBuilt: %s/build/Build/Products/Release/Nemotron Parse Studio.app\n' "$PWD"
