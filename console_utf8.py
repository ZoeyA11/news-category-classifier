"""
Console output encoding fix for Windows.

The problem: on Windows, stdout defaults to the cp1252 code page, which cannot
represent most non-ASCII characters. Any `print` containing such a character
raises `UnicodeEncodeError` and kills the script.

This matters here even though the project's own output is English, because the
dataset text is not: HuffPost headlines contain typographic apostrophes,
em dashes, accented names and the occasional emoji. Printing a raw headline -
which Part 2 does when showing before/after cleaning examples - is enough to
trigger the crash.

The fix: switch stdout/stderr to UTF-8 at import time. Import this module
*before* anything is printed.
"""

import sys


def enable_utf8_console() -> None:
    """Switches stdout and stderr to UTF-8, where the environment allows it."""
    for stream in (sys.stdout, sys.stderr):
        # `reconfigure` exists from Python 3.7 onward. errors="replace" means a
        # character that still cannot be represented degrades to a placeholder
        # instead of raising.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


enable_utf8_console()
