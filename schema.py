# schema.py
from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class Schema:
    # Categorical inputs
    CAT_COLS: List[str] = (
        "CUSTOMER_ID",
        "GEOGRAPHY",
    )

    # Yes/No flags (we will feed as float32 0.0/1.0 for safe deployment)
    YESNO_COLS: List[str] = (
        "ACTIVE_FLAG",
        "IN_TIME_PAYMENT",
        "INVOICE_ON_TIME",
        "EU_VAT_ISSUE",
    )

    # Numeric inputs
    NUM_COLS: List[str] = (
        "CUSTOMER_CLAIMS",
        "INCIDENT_COUNT",
        "INCIDENT_RESOLVED_IN_TIME",
    )

    # Excel column: YYYY-MM-DD (e.g., 2026-01-16)
    DATE_COL: str = "POSTING_MONTH"

    # Derived columns used by model (we will feed as float32 too)
    DERIVED_COLS: List[str] = (
        "POSTING_YEAR",
        "POSTING_MON",
    )

    # Targets (training only)
    TARGET_REG: str = "CUSTOMER_SATISFACTION"
    TARGET_CLS: str = "CUSTOMER_RETAINED"

    # Model outputs
    OUT_SAT: str = "sat"
    OUT_RET: str = "ret"

    def model_input_names(self) -> List[str]:
        return (
            list(self.CAT_COLS)
            + list(self.YESNO_COLS)
            + list(self.NUM_COLS)
            + list(self.DERIVED_COLS)
        )

SCHEMA = Schema()

# Single source of truth for inference/training dtypes
# IMPORTANT: Flags + derived date fields are float32 to avoid Cast/Lambda layers.
TF_DTYPES: Dict[str, str] = {
    "CUSTOMER_ID": "string",
    "GEOGRAPHY": "string",

    "ACTIVE_FLAG": "float32",
    "IN_TIME_PAYMENT": "float32",
    "INVOICE_ON_TIME": "float32",
    "EU_VAT_ISSUE": "float32",

    "CUSTOMER_CLAIMS": "float32",
    "INCIDENT_COUNT": "float32",
    "INCIDENT_RESOLVED_IN_TIME": "float32",

    "POSTING_YEAR": "float32",
    "POSTING_MON": "float32",
}

def assert_schema_consistency():
    names = SCHEMA.model_input_names()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate column names in schema.")
    missing = [n for n in names if n not in TF_DTYPES]
    if missing:
        raise ValueError(f"Missing dtype definitions for: {missing}")