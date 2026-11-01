"""
analyzer.py
-----------
Loads a dataset and produces a structural profile:
    - Shape (rows, columns)
    - Column dtypes
    - Missing value counts
    - Unique value counts
    - Basic numeric statistics
"""

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame.
    Raise a clear error if the format isn't supported.
    """
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    elif lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    else:
        raise ValueError("Only .csv, .xls, and .xlsx files are supported.")


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return a structural profile of the DataFrame."""
    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": {col: int(df[col].isna().sum()) for col in df.columns},
        "missing_pct": {col: round(df[col].isna().mean() * 100, 2) for col in df.columns},
        "unique": {col: int(df[col].nunique()) for col in df.columns},
        "numeric_summary": {},
        "sample": df.head(5).to_dict(orient="records"),
    }

    # Numeric stats (only for numeric columns)
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        profile["numeric_summary"] = numeric_df.describe().round(2).to_dict()

    return profile


def build_documentation_prompt(df: pd.DataFrame, metadata: dict, profile: dict) -> str:
    """
    Build a prompt for the LLM asking it to generate documentation text.
    """
    col_info = "\n".join(
        f"  - {c} (dtype={profile['dtypes'][c]}, "
        f"missing={profile['missing_pct'][c]}%, "
        f"unique={profile['unique'][c]})"
        for c in profile["column_names"]
    )

    prompt = f"""You are a data documentation expert. Write a clear, professional
dataset documentation report based on the information below.

DATASET METADATA
Name:      {metadata.get('dataset_name', 'N/A')}
Source:    {metadata.get('source', 'N/A')}
Purpose:   {metadata.get('purpose', 'N/A')}
Owner:     {metadata.get('owner', 'N/A')}
License:   {metadata.get('license', 'N/A')}

STRUCTURE
Rows: {profile['rows']}
Columns: {profile['columns']}

COLUMNS
{col_info}

Write a report with these sections:
1. Overview
2. Intended Use
3. Column Descriptions
4. Known Limitations
5. Ethical Considerations

Be concise and factual. Do not invent details that weren't provided.
"""
    return prompt