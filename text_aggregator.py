import re


GENERIC_CAPTION_PHRASES = {
    "a person standing",
    "a man speaking",
    "a woman speaking",
    "person speaking",
    "man speaking",
    "woman speaking",
    "news anchor speaking",
    "news anchor speaking in studio",
    "anchor speaking",
    "person standing",
    "person sitting",
    "studio",
    "news studio",
    "a close up",
    "a screenshot",
}


CONTENT_PRIORITY = {
    "article_text": 0,
    "article": 0,
    "transcript": 1,
    "audio_transcript": 1,
    "frame_ocr": 2,
    "ocr": 2,
    "caption": 3,
    "frame_caption": 3,
}


def normalize_text(text):
    text = str(text or "").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_piece(piece):
    content_type = piece.get("type") or piece.get("content_type") or "text"
    return {
        "type": content_type,
        "content_type": content_type,
        "text": normalize_text(piece.get("text", "")),
        "timestamp_sec": piece.get("timestamp_sec"),
    }


def is_meaningful_caption(text):
    normalized = normalize_text(text).lower().strip(" .")
    if len(normalized) < 12:
        return False
    if normalized in GENERIC_CAPTION_PHRASES:
        return False
    words = normalized.split()
    if len(words) <= 3 and all(word in {"person", "man", "woman", "standing", "sitting", "speaking", "studio"} for word in words):
        return False
    return True


def is_meaningful_piece(piece):
    text = piece["text"]
    if not text:
        return False
    content_type = piece["content_type"]
    if content_type in {"caption", "frame_caption"}:
        return is_meaningful_caption(text)
    return True


def prepare_content_pieces(pieces):
    prepared = []
    seen = set()
    for raw_piece in pieces or []:
        piece = normalize_piece(raw_piece)
        if not is_meaningful_piece(piece):
            continue
        dedupe_key = (piece["content_type"], piece["text"].lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        prepared.append(piece)
    return prepared


def aggregate_text(pieces, source_type=None):
    prepared = prepare_content_pieces(pieces)
    prepared.sort(key=lambda piece: CONTENT_PRIORITY.get(piece["content_type"], 10))
    return normalize_text(" ".join(piece["text"] for piece in prepared))


def chunk_text(text, max_chars=1200, overlap=150):
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sentence_end = max(text.rfind(".", start, end), text.rfind("?", start, end), text.rfind("!", start, end))
            if sentence_end > start + max_chars // 2:
                end = sentence_end + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]
