def select_science_header(original_header, solved_header):
    """Return solved header when available, otherwise keep original header."""
    return solved_header if solved_header is not None else original_header


def copy_header_or_none(header):
    """Return a shallow copy of header or None if header is missing."""
    return header.copy() if header is not None else None
