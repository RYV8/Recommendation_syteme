from __future__ import annotations

"""Legacy re-export module.

This module is kept for backward compatibility. The canonical implementation
now lives in infrastructure.data_processing.
"""

from ..infrastructure.data_processing import (  # noqa: F401
    categorize_tag,
    handle_rating,
    handler_genre,
    handler_tag,
)

__all__ = [
    "categorize_tag",
    "handle_rating",
    "handler_genre",
    "handler_tag",
]
