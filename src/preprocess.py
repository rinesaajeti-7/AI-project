import re

_url = re.compile(r"(https?://\S+|www\.\S+)")
_html = re.compile(r"<[^>]+>")
_nonalpha = re.compile(r"[^a-z\s]")

def basic_clean(text: str) -> str:
    """Lightweight text normalization.

    - Lowercases

    - Strips URLs and HTML tags

    - Removes non-letters

    - Collapses whitespace

    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _url.sub(" ", t)
    t = _html.sub(" ", t)
    t = _nonalpha.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
