
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class Document:
    document_id: str
    source: str
    content: str


def load_documents(directory: str | Path) -> list[Document]:
    """Load supported text documents from a directory recursively."""
    root = Path(directory)

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    documents = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        raw_text = path.read_text(encoding="utf-8")
        content = raw_text.strip()

        if not content:
            print(f"Skipping empty file: {path}")
            continue

        source = path.relative_to(root).as_posix()
        document_id = sha256(source.encode("utf-8")).hexdigest()[:16]

        documents.append(
            Document(
                document_id=document_id,
                source=source,
                content=content,
            )
        )

    return documents
