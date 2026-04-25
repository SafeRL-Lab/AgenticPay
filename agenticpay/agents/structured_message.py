"""Helpers to extract <message>...</message> from model outputs with robust fallbacks."""

import re
from typing import Optional

# Match <message> or <message ...> and paired </message>
_MESSAGE_PAIRED = re.compile(
    r"<message[^>]*>(.*?)</message>", re.DOTALL | re.IGNORECASE
)
_MESSAGE_OPEN = re.compile(r"<message[^>]*>\s*", re.DOTALL | re.IGNORECASE)
_MESSAGE_CLOSE = re.compile(r"</message>", re.IGNORECASE)


def extract_message_content(response: str) -> Optional[str]:
    """Try structured extraction only (no tag stripping).

    Order:
    1. First paired ``<message>...</message>`` (``<message>`` may have attributes).
    2. First ``<message>`` ... optional ``</message>`` (handles missing closing tag
       or cases where the non-greedy pair match failed e.g. unusual nesting).

    Returns None if no ``<message`` opening tag is found in the text.
    """
    if not response or not response.strip():
        return None

    m = _MESSAGE_PAIRED.search(response)
    if m:
        return m.group(1).strip()

    m_open = _MESSAGE_OPEN.search(response)
    if m_open:
        tail = response[m_open.end() :]
        m_close = _MESSAGE_CLOSE.search(tail)
        if m_close:
            return tail[: m_close.start()].strip()
        return tail.strip()

    return None
