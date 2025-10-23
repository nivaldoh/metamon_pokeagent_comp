#!/usr/bin/env python3
"""
Infer a complete JSON Schema (draft 2020-12) from a replay file (.json or .json.lz4)
and print one example value for every field.

Features
- Reads .json.lz4 using lz4.frame; plain .json also supported.
- Scans the entire JSON (not just a sample) to union types/keys.
- Computes "required" keys (present in all instances of a given object).
- Attaches a single example at every node via the "examples" keyword.
- Prints a flat "path -> example" table for quick inspection.

Usage
  pip install lz4
  python inspect_replay_schema.py /path/to/replay.json.lz4 --schema-out schema.json

Options
  --schema-out FILE   Write the inferred JSON Schema to FILE (otherwise print to stdout).
  --print-table       Print a flat table of path -> example (default on).
  --no-print-table    Disable the table.
  --truncate N        Truncate printed example JSON values to N chars (default 140).
"""

from __future__ import annotations
import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import lz4.frame as lz4f  # required for .lz4 files
except Exception:
    lz4f = None  # soft import for non-.lz4 usage


# ---------- IO ----------

def read_json(path: str) -> Any:
    """Read JSON from .json.lz4 (LZ4 frame) or plain .json."""
    if path.lower().endswith(".lz4"):
        if lz4f is None:
            raise RuntimeError("lz4 is required for .lz4 files. Install with: pip install lz4")
        with lz4f.open(path, "rb") as f:
            raw = f.read()
        return json.loads(raw.decode("utf-8"))
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ---------- Schema inference core ----------

JSONType = str  # one of: "object","array","string","integer","number","boolean","null"

def _json_type_of(value: Any) -> JSONType:
    # Order matters: bool is subclass of int in Python
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    # Fallback (shouldn't happen for valid JSON)
    return "string"


class SchemaNode:
    """
    Accumulator node for schema inference.
    - types: set of JSON types seen at this node
    - properties: for object nodes
    - required_counts: how many times each property appeared
    - object_instances: how many object instances merged into this node
    - items: for array node (single "items" schema; unions happen inside it)
    - example: first non-null example value seen (or constructed later)
    """
    __slots__ = (
        "types",
        "properties",
        "required_counts",
        "object_instances",
        "items",
        "example",
        "saw_empty_array",
    )

    def __init__(self):
        self.types: Set[JSONType] = set()
        self.properties: Dict[str, SchemaNode] = {}
        self.required_counts: Dict[str, int] = {}
        self.object_instances: int = 0
        self.items: Optional[SchemaNode] = None
        self.example: Any = None
        self.saw_empty_array: bool = False

    def merge(self, value: Any):
        jt = _json_type_of(value)
        self.types.add(jt)

        # capture first non-null example if possible
        if self.example is None and value is not None:
            # To avoid giant examples, keep small structures; full value is fine for schema "examples".
            self.example = value

        if jt == "object":
            self.object_instances += 1
            obj: Dict[str, Any] = value  # type: ignore[assignment]
            for k, v in obj.items():
                child = self.properties.get(k)
                if child is None:
                    child = self.properties[k] = SchemaNode()
                child.merge(v)
                self.required_counts[k] = self.required_counts.get(k, 0) + 1

        elif jt == "array":
            arr: List[Any] = value  # type: ignore[assignment]
            if not arr:
                self.saw_empty_array = True
            else:
                if self.items is None:
                    self.items = SchemaNode()
                for el in arr:
                    self.items.merge(el)

        # primitives don't need special handling

    # ----- Schema emission -----

    def _type_field(self) -> Any:
        # If multiple types were seen, emit union list (e.g., ["integer","null"])
        ts = sorted(self.types)
        if not ts:
            return "object"  # fallback
        if len(ts) == 1:
            return ts[0]
        return ts

    def _required_list(self) -> List[str]:
        # A property is "required" iff it appeared in every object instance merged.
        if self.object_instances == 0:
            return []
        req = [k for k, c in self.required_counts.items() if c == self.object_instances]
        req.sort()
        return req

    def to_schema(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"type": self._type_field()}

        # Attach a single example (JSON Schema "examples" is an annotation and can be any JSON value)
        example_value = self.build_example_value()
        if example_value is not None:
            schema["examples"] = [example_value]

        if "object" in self.types:
            props: Dict[str, Any] = {}
            for k, child in sorted(self.properties.items()):
                props[k] = child.to_schema()
            schema["properties"] = props
            req = self._required_list()
            if req:
                schema["required"] = req

        if "array" in self.types:
            if self.items is not None:
                schema["items"] = self.items.to_schema()
            else:
                # empty arrays only observed; allow any items
                schema["items"] = {}
        return schema

    # ----- Example generation; also used for the printed table -----

    def build_example_value(self) -> Any:
        """
        Produce one representative example for this node.
        For objects/arrays where no direct example was captured, synthesize from children.
        """
        jt_list = sorted(self.types)
        if not jt_list:
            return None

        # Prefer a captured example if it matches observed types (and is not huge).
        if self.example is not None:
            return self.example

        # Synthesize structured examples if needed
        if "object" in self.types:
            out = {}
            for k, child in self.properties.items():
                child_ex = child.build_example_value()
                if child_ex is not None:
                    out[k] = child_ex
            return out

        if "array" in self.types:
            if self.items is not None:
                item_ex = self.items.build_example_value()
                return [item_ex] if item_ex is not None else []
            return []

        # Primitive but we only saw nulls
        if jt_list == ["null"]:
            return None

        return None  # shouldn't reach for primitives if example was captured


# ---------- Flatten for table ----------

def flatten_examples(schema_node: SchemaNode, prefix: str = "") -> List[Tuple[str, Any]]:
    rows: List[Tuple[str, Any]] = []

    # If this node has a primitive example, include it under the current prefix (for objects/arrays we expand children)
    example = schema_node.build_example_value()

    # For objects, recurse into properties
    if "object" in schema_node.types:
        for k, child in sorted(schema_node.properties.items()):
            child_prefix = f"{prefix}.{k}" if prefix else k
            rows.extend(flatten_examples(child, child_prefix))
        return rows

    # For arrays, show the items example under "prefix[]"
    if "array" in schema_node.types:
        if schema_node.items:
            child_prefix = f"{prefix}[]" if prefix else "[]"
            rows.extend(flatten_examples(schema_node.items, child_prefix))
        else:
            # empty array only; show placeholder
            rows.append((f"{prefix}[]", []))
        return rows

    # Primitive (or null-only) example
    if prefix:
        rows.append((prefix, example))
    return rows


# ---------- CLI ----------

def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else (s[: n - 1] + "…")


def main():
    ap = argparse.ArgumentParser(description="Infer JSON Schema (draft 2020-12) and examples from a replay JSON(.lz4) file.")
    ap.add_argument("path", help="Path to .json.lz4 (or plain .json)")
    ap.add_argument("--schema-out", help="Write schema to this file (JSON). If omitted, schema is printed.")
    ap.add_argument("--print-table", dest="print_table", action="store_true", default=True, help="Print a flat path->example table (default on).")
    ap.add_argument("--no-print-table", dest="print_table", action="store_false", help="Disable the table.")
    ap.add_argument("--truncate", type=int, default=140, help="Truncate long example values when printing the table.")
    args = ap.parse_args()

    data = read_json(args.path)

    # Build the accumulator and merge the entire instance
    root = SchemaNode()
    root.merge(data)

    # Emit JSON Schema (draft 2020-12)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"Inferred schema for {args.path}",
        **root.to_schema(),
    }

    if args.schema_out:
        with open(args.schema_out, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        print(f"Wrote schema to: {args.schema_out}")
    else:
        print(json.dumps(schema, ensure_ascii=False, indent=2))

    # Print a handy "path -> example" table
    if args.print_table:
        rows = flatten_examples(root)
        if rows:
            col = max(len(k) for k, _ in rows)
            print("\n=== Examples (path -> example) ===")
            for k, v in rows:
                try:
                    js = json.dumps(v, ensure_ascii=False)
                except Exception:
                    js = str(v)
                print(f"{k.ljust(col)}  : {truncate(js, args.truncate)}")
        else:
            print("\n(no examples found)")

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
