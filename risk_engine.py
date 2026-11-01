"""
risk_engine.py
--------------
Rule-based risk analysis for datasets.

Checks for:
    - Missing documentation fields
    - Potential PII columns (name, email, ssn, etc.)
    - Sensitive attributes (race, gender, religion, etc.)
    - Class imbalance in a target column
    - High missing-value ratios

Returns a list of risk dictionaries with severity levels.
"""

import pandas as pd


# Keywords that usually indicate personally identifiable info
PII_KEYWORDS = [
    "name", "email", "phone", "ssn", "social", "passport",
    "address", "dob", "birth", "credit", "card", "account",
    "ip", "location", "zip", "postal",
]

# Sensitive/protected attributes
SENSITIVE_KEYWORDS = [
    "race", "ethnicity", "gender", "sex", "religion",
    "disability", "age", "marital", "nationality", "orientation",
]


def _match_columns(columns, keywords):
    """Return columns whose lowercase name contains any keyword."""
    hits = []
    for col in columns:
        low = str(col).lower()
        for kw in keywords:
            if kw in low:
                hits.append(col)
                break
    return hits


def analyze_risks(df: pd.DataFrame, metadata: dict = None) -> list:
    """
    Run all risk checks and return a list of findings.

    Each finding is a dict:
        {
            "type": str,        # short category
            "severity": str,    # "High" | "Medium" | "Low"
            "message": str,     # human-readable description
            "columns": list,    # affected columns (if any)
        }
    """
    metadata = metadata or {}
    findings = []
    cols = list(df.columns)

    # ---- 1. Missing documentation ----
    required_docs = ["dataset_name", "source", "purpose", "owner", "license"]
    missing_docs = [k for k in required_docs if not metadata.get(k)]
    if missing_docs:
        findings.append({
            "type": "Missing Documentation",
            "severity": "Medium",
            "message": f"Missing metadata fields: {', '.join(missing_docs)}",
            "columns": [],
        })

    # ---- 2. PII columns ----
    pii_cols = _match_columns(cols, PII_KEYWORDS)
    if pii_cols:
        findings.append({
            "type": "Potential PII",
            "severity": "High",
            "message": f"Columns that may contain personal data: {', '.join(map(str, pii_cols))}",
            "columns": pii_cols,
        })

    # ---- 3. Sensitive attributes ----
    sens_cols = _match_columns(cols, SENSITIVE_KEYWORDS)
    if sens_cols:
        findings.append({
            "type": "Sensitive Attribute",
            "severity": "Medium",
            "message": f"Protected/sensitive attributes present: {', '.join(map(str, sens_cols))}",
            "columns": sens_cols,
        })

    # ---- 4. High missing values ----
    high_missing = []
    for col in cols:
        ratio = df[col].isna().mean()
        if ratio > 0.30:
            high_missing.append(f"{col} ({ratio:.0%})")
    if high_missing:
        findings.append({
            "type": "High Missing Values",
            "severity": "Medium",
            "message": f"Columns with >30% missing: {', '.join(high_missing)}",
            "columns": [],
        })

    # ---- 5. Class imbalance in target column ----
    target = metadata.get("target_column")
    if target and target in df.columns:
        counts = df[target].value_counts(normalize=True)
        if not counts.empty and counts.iloc[0] > 0.80:
            findings.append({
                "type": "Class Imbalance",
                "severity": "Medium",
                "message": (
                    f"Target '{target}' is imbalanced — "
                    f"top class is {counts.iloc[0]:.0%} of rows."
                ),
                "columns": [target],
            })

    # ---- 6. Tiny dataset ----
    if len(df) < 100:
        findings.append({
            "type": "Small Dataset",
            "severity": "Low",
            "message": f"Only {len(df)} rows — may not be representative.",
            "columns": [],
        })

    return findings


def summarize_risks(findings: list) -> dict:
    """Return counts by severity."""
    summary = {"High": 0, "Medium": 0, "Low": 0}
    for f in findings:
        summary[f["severity"]] = summary.get(f["severity"], 0) + 1
    return summary