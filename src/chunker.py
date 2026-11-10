from dataclasses import dataclass
from src.document_loader import Document

@dataclass
class Chunk:
    chunk_id: str
    source: str
    chunk_index: int
    content: str


def fixed_size_chunks(
        document: Document,
        chunk_size: int = 300,
        overlap: int = 50,

) -> list[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be greater than 0 and less than chunk size")

    text = document.content
    chunks = []
    step = chunk_size - overlap


    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        content = text[start:end].strip()

        if content:
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id= f"{document.document_id}_{index}",
                    source=document.source,
                    chunk_index=index,
                    content=content


                )
            )

        if end == len(text):
            break

    return chunks

def paragraph_chunks(
        document: Document,
        max_chars: int = 300,

) -> list[Chunk]:

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    paragraphs = [
        paragraph.strip()
        for paragraph in document.content.split("\n\n")
        if paragraph.strip()
    ]

    groups = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                groups.append(current)
                current = ""

            for start in range(0, len(paragraph), max_chars):
                piece = paragraph[start: start + max_chars].strip()
                if piece:
                    groups.append(piece)

        elif not current:
            current = paragraph

        elif len(current) + 2 + len(paragraph) <= max_chars:
            current += "\n\n" + paragraph

        else:
            groups.append(current)
            current = paragraph

    if current:
        groups.append(current)

    return[
        Chunk(
            chunk_id=f"{document.document_id}_{index}",
            source=document.source,
            chunk_index=index,
            content=content,
        )
        for index, content in enumerate(groups)
    ]