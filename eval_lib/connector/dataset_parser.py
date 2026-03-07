import csv
import json
import io
from typing import List, Dict, Any, Tuple


def parse_csv(content: bytes) -> Tuple[List[str], List[Dict[str, Any]]]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    columns = reader.fieldnames or []
    rows = [dict(row) for row in reader]
    return list(columns), rows


def parse_json(content: bytes) -> Tuple[List[str], List[Dict[str, Any]]]:
    data = json.loads(content.decode("utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                rows = v
                break
        else:
            rows = [data]
    else:
        raise ValueError("JSON must be an array or object containing an array")
    if not rows:
        return [], []
    columns = list(rows[0].keys())
    return columns, rows


def parse_jsonl(content: bytes) -> Tuple[List[str], List[Dict[str, Any]]]:
    text = content.decode("utf-8")
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        return [], []
    columns = list(rows[0].keys())
    return columns, rows


def parse_dataset(content: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "csv":
        return parse_csv(content)
    elif ext == "json":
        return parse_json(content)
    elif ext == "jsonl":
        return parse_jsonl(content)
    else:
        raise ValueError(f"Unsupported file format: .{ext}. Use .csv, .json, or .jsonl")
