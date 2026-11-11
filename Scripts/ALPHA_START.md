# Nemotron Parse Studio alpha

This ZIP includes a prebuilt app for Apple silicon Macs. Extract it and open **Nemotron Parse Studio.app**. You can preview documents before setting up the local model.

1. For local parsing, install Python 3.12 and run **Set Up Local Parsing.command** from the extracted folder. The command installs the Python dependencies and downloads the pinned NVIDIA model, about 2.5 GB.
2. Open or reopen **Nemotron Parse Studio.app**. Import **Sample.png** with **⌘O**, then parse it with **⌘R**. New setups use MPS for inference.
3. The app can be moved after setup. The Python environment and model live in `~/Library/Application Support/NemotronParseStudio/Runtime`.

To use an existing copy of the pinned model, run `./Set\ Up\ Local\ Parsing.command --python-only` in Terminal from the extracted folder, then select that model directory in Settings.

The alpha is ad hoc signed. macOS may require approval for a downloaded app. [Apple's guidance on opening downloaded apps](https://support.apple.com/en-us/102445)

Native Core AI uses separate converted assets. Follow the source repository's Core AI guide if you need that experimental backend.
