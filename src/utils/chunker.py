"""Smart text chunker with table-aware splitting."""
import re


def smart_chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text into overlapping chunks.
    Keeps table sections together — does not split mid-table.
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    # Separate OCR text and Visual description sections
    sections = re.split(r'\[(OCR TEXT|VISUAL DESCRIPTION)\]', text)

    all_chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # If section is small enough — keep as one chunk
        if len(section) <= chunk_size:
            all_chunks.append(section)
            continue

        # Split by double newline (paragraph breaks)
        paragraphs = re.split(r'\n\n+', section)

        current = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            # If single paragraph too large, split by lines
            if para_len > chunk_size:
                lines = para.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if current_len + len(line) > chunk_size and current:
                        all_chunks.append('\n'.join(current))
                        # Keep last few lines for overlap
                        current = current[-3:]
                        current_len = sum(len(l) for l in current)
                    current.append(line)
                    current_len += len(line)
                continue

            if current_len + para_len > chunk_size and current:
                all_chunks.append('\n\n'.join(current))
                # Overlap: keep last paragraph
                current = current[-1:]
                current_len = sum(len(p) for p in current)

            current.append(para)
            current_len += para_len

        if current:
            all_chunks.append('\n\n'.join(current))

    return [c for c in all_chunks if c.strip()]