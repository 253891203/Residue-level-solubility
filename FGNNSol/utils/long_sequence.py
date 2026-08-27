def make_windows(length: int, chunk_size: int = 1022, overlap: int = 128) -> list[tuple[int, int]]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    if length <= chunk_size:
        return [(0, length)]
    windows, start = [], 0
    while start < length:
        end = min(start + chunk_size, length)
        windows.append((start, end))
        if end == length:
            break
        start = end - overlap
    return windows
