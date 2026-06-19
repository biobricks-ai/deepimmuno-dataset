#!/usr/bin/env python3
"""
Download and build DeepImmuno curated peptide-MHC immunogenicity dataset.

Source: https://github.com/frankligy/DeepImmuno
DOI: 10.1093/bib/bbab160
License: MIT

Paper: DeepImmuno: deep learning-empowered prediction and generation of
immunogenic peptides for T-cell immunity. Briefings in Bioinformatics, 2021.

Data files:
  - reproduce/data/df/df_all.csv           : IEDB-derived training set
  - data/remove_low_negative/remove0123.csv: Curated benchmark (IEDB+filtering)
  - reproduce/data/dengue_test.csv          : Dengue virus validation set
  - reproduce/data/deephlapan_result_cell.csv: TESLA tumor neoantigen benchmark
  - reproduce/data/covid_predicted_result.csv: SARS-CoV-2 epitope predictions
"""

from pathlib import Path
import pandas as pd
import requests

GITHUB_RAW = "https://raw.githubusercontent.com/frankligy/DeepImmuno/main"

FILES = {
    "training": "reproduce/data/df/df_all.csv",
    "benchmark_curated": "data/remove_low_negative/remove0123.csv",
    "dengue_validation": "reproduce/data/dengue_test.csv",
    "tesla_neoantigen": "reproduce/data/deephlapan_result_cell.csv",
    "sars_cov2_predictions": "reproduce/data/covid_predicted_result.csv",
}

PARQUET_NAMES = {
    "training": "training.parquet",
    "benchmark_curated": "benchmark_curated.parquet",
    "dengue_validation": "dengue_validation.parquet",
    "tesla_neoantigen": "tesla_neoantigen.parquet",
    "sars_cov2_predictions": "sars_cov2_predictions.parquet",
}


def download_csv(path: str) -> pd.DataFrame:
    url = f"{GITHUB_RAW}/{path}"
    print(f"  Fetching {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text))


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and underscore column names, convert objects to str."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
    return df


def main():
    brick_path = Path("brick")
    brick_path.mkdir(exist_ok=True)

    print("=" * 60)
    print("DeepImmuno dataset builder")
    print("DOI: 10.1093/bib/bbab160")
    print("=" * 60)

    total_records = 0

    for key, github_path in FILES.items():
        print(f"\n[{key}]")
        df = download_csv(github_path)
        df = standardize_columns(df)
        out = brick_path / PARQUET_NAMES[key]
        df.to_parquet(out, index=False)
        print(f"  -> {out.name}: {len(df):,} rows x {len(df.columns)} cols")
        print(f"     columns: {list(df.columns)}")
        print(f"     sample:")
        print(df.head(3).to_string(index=False))
        total_records += len(df)

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for f in sorted(brick_path.glob("*.parquet")):
        df = pd.read_parquet(f)
        print(f"  {f.name}: {len(df):,} rows, {list(df.columns)}")
    print(f"\nTotal records: {total_records:,}")


if __name__ == "__main__":
    main()
