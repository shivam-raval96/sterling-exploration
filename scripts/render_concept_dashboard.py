from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from sterling_exploration.visualization import concept_distribution_html


def add_catalog_labels(
    distribution: dict[str, Any], catalog_path: Path
) -> dict[str, Any]:
    catalog = pd.read_parquet(catalog_path).set_index(["head", "concept_id"])
    for groups in distribution.values():
        for head, rows in groups.items():
            for row in rows:
                key = (head, row["concept_id"])
                if key not in catalog.index:
                    continue
                concept = catalog.loc[key]
                row["concept_name"] = str(concept["concept_name"])
                row["concept_description"] = str(concept["concept_description"])
                row["group_name"] = (
                    None if pd.isna(concept["group_name"]) else str(concept["group_name"])
                )
    return distribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("distribution", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    distribution = json.loads(args.distribution.read_text())
    labeled = add_catalog_labels(distribution, args.catalog)
    output = args.output or args.distribution.with_suffix(".html")
    output.write_text(concept_distribution_html(labeled, top_n=args.top_n))


if __name__ == "__main__":
    main()
