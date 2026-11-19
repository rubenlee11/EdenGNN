"""Initialize ~/.edengnn from a source checkout and download universal models.

Run automatically during Meson builds/installs or manually with:
    python scripts/install_user_data.py /path/to/EdenGNN-main
Existing files are preserved, including user-configured pseudopotential maps.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request
from urllib.parse import quote


MODEL_TREE_URL = "https://huggingface.co/api/models/TrueSavage/EdenGNN/tree/main/mp?recursive=true"
MODEL_URL = "https://huggingface.co/TrueSavage/EdenGNN/resolve/main/mp"


def iter_model_files():
    """Yield all file paths under the repository's mp/ tree, including new files."""
    url = MODEL_TREE_URL
    while url:
        with urllib.request.urlopen(url, timeout=120) as response:
            entries = json.load(response)
            next_page = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link", ""))
        for entry in entries:
            if entry["type"] == "file":
                yield entry["path"].removeprefix("mp/")
        url = next_page.group(1) if next_page else None


def install_user_data(source_dir):
    """Copy missing resources and models to the current user's ~/.edengnn."""
    target_dir = Path.home() / ".edengnn"
    for source in (source_dir / "scripts").rglob("*"):
        if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".pyc":
            continue
        target = target_dir / source.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)

    target = target_dir / "config.yaml"
    if not target.exists():
        shutil.copy2(source_dir / "config.yaml", target)

    for filename in iter_model_files():
        target = target_dir / "edengnn-uni" / filename
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {filename} to {target}", flush=True)
        # Publish only complete downloads so an interrupted install can be retried.
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary_dir:
            temporary_path = Path(temporary_dir) / target.name
            with urllib.request.urlopen(f"{MODEL_URL}/{quote(filename)}", timeout=120) as response:
                with temporary_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            temporary_path.replace(target)
    print(f"EdenGNN user files: {target_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="EdenGNN source checkout")
    parser.add_argument("--stamp", type=Path, help="Meson build completion stamp")
    args = parser.parse_args()
    install_user_data(args.source_dir.resolve())
    if args.stamp is not None:
        args.stamp.touch()


if __name__ == "__main__":
    main()
