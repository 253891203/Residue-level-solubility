from pathlib import Path


def normalize_sequence(value: str) -> str:
    return "".join(str(value).split()).upper()


def read_fasta_file(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    headers = [x[1:].strip() for x in lines if x.startswith(">")]
    sequence = normalize_sequence("".join(x.strip() for x in lines if x and not x.startswith(">")))
    if not sequence:
        raise ValueError(f"No sequence in FASTA: {path}")
    return (headers[0] if headers else path.stem), sequence


def read_fasta_directory(directory: Path) -> list[dict]:
    if not directory.is_dir():
        raise FileNotFoundError(f"FASTA directory not found: {directory}")
    records = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        header, sequence = read_fasta_file(path)
        records.append({"fasta_filename": path.name, "fasta_path": str(path.resolve()),
                        "fasta_header": header, "sequence": sequence})
    return records
