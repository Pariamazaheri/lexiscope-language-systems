"""Execute the report notebook without retraining or network downloads."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = next((root / "notebooks").glob("*.ipynb"))
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    nbformat.write(notebook, path)
    print("Executed", path.name)


if __name__ == "__main__":
    main()
