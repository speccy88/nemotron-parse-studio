#!/usr/bin/env python3
"""Build a signed Apple silicon alpha ZIP from a committed source snapshot."""

import argparse
import hashlib
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "Nemotron Parse Studio.app"


def build_release(tag, commit, output):
    with tempfile.TemporaryDirectory(prefix="parse-studio-release-") as work:
        work = Path(work)
        source = work / "source"
        source.mkdir()
        tarball = subprocess.check_output(["git", "archive", commit], cwd=ROOT)
        subprocess.run(["tar", "-xf", "-"], input=tarball, cwd=source, check=True)
        subprocess.run(["Scripts/build.sh", "ARCHS=arm64"], cwd=source, check=True)

        package = work / f"Nemotron Parse Studio {tag}"
        package.mkdir()
        app = package / APP_NAME
        built_app = source / "build/Build/Products/Release" / APP_NAME
        subprocess.run(["ditto", str(built_app), str(app)], check=True)
        resources = app / "Contents/Resources"
        shutil.copytree(source / "Backend", resources / "Backend")
        plist = app / "Contents/Info.plist"
        info = plistlib.loads(plist.read_bytes())
        if info["CFBundleShortVersionString"] != tag[1:].split("-", 1)[0]:
            raise SystemExit("The release tag and app version must match.")
        info.pop("StudioSourceRoot", None)
        info["StudioReleaseTag"] = tag
        info["StudioReleaseCommit"] = commit
        plist.write_bytes(plistlib.dumps(info))

        subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True)
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)
        executable = app / "Contents/MacOS" / info["CFBundleExecutable"]
        if subprocess.check_output(["lipo", "-archs", str(executable)], text=True).strip() != "arm64":
            raise SystemExit("The alpha download must contain an Apple silicon executable.")

        for name in ("Set Up Local Parsing.command", "ALPHA_START.md"):
            shutil.copy2(source / "Scripts" / name, package / name)
        shutil.copy2(source / "Conversion/evidence/pages/mixed.png", package / "Sample.png")

        archive = output / f"Nemotron-Parse-Studio-{tag}-macOS-arm64.zip"
        output.mkdir(parents=True)
        subprocess.run(
            ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(package), str(archive)],
            check=True,
        )
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        (output / "SHA256SUMS").write_text(f"{checksum}  {archive.name}\n")
        print(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Release tag, for example v1.0.0-alpha.1")
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args()
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?", args.tag):
        parser.error("expected a version tag like v1.0.0-alpha.1")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    output = ROOT / "Artifacts/Releases" / args.tag
    if output.exists():
        parser.error(f"{output} already exists")
    build_release(args.tag, commit, output)


if __name__ == "__main__":
    main()
