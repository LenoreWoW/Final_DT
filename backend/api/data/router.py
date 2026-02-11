"""
Data Upload API Router - Upload and analyse datasets for twin creation.

Endpoints:
    POST /api/data/upload  - Upload CSV, JSON, or Excel files.
                             Returns detected columns, inferred types,
                             and suggested entity/property mappings.
"""

import io
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/data", tags=["data"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ColumnInfo(BaseModel):
    """Metadata about a single detected column."""
    name: str
    dtype: str  # "numeric", "categorical", "datetime", "text", "boolean"
    sample_values: List[Any] = Field(default_factory=list)
    null_count: int = 0
    unique_count: int = 0


class SuggestedMapping(BaseModel):
    """A suggested entity/property mapping based on column analysis."""
    column: str
    suggested_role: str  # "entity_id", "property", "timestamp", "label", "feature"
    confidence: float = Field(ge=0.0, le=1.0)


class DataUploadResponse(BaseModel):
    """Response after a successful data upload and analysis."""
    upload_id: str
    filename: str
    file_type: str  # "csv", "json", "excel"
    row_count: int
    column_count: int
    columns: List[ColumnInfo]
    suggested_mappings: List[SuggestedMapping]
    preview: List[Dict[str, Any]] = Field(default_factory=list)
    detected_domain: Optional[str] = None
    uploaded_at: datetime


# ---------------------------------------------------------------------------
# Allowed extensions and MIME types
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".csv": "csv",
    ".json": "json",
    ".xlsx": "excel",
    ".xls": "excel",
}

MAX_FILE_SIZE_MB = 50


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _infer_file_type(filename: str) -> str:
    """Determine the file type from its extension."""
    for ext, ftype in ALLOWED_EXTENSIONS.items():
        if filename.lower().endswith(ext):
            return ftype
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported file type. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS.keys())}",
    )


def _infer_dtype(series) -> str:
    """Map a pandas series dtype to a human-readable string."""
    import numpy as np

    dtype = series.dtype
    if np.issubdtype(dtype, np.number):
        return "numeric"
    if np.issubdtype(dtype, np.datetime64):
        return "datetime"
    if np.issubdtype(dtype, np.bool_):
        return "boolean"

    # Heuristic: if few unique values relative to total, treat as categorical.
    # Guard against unhashable types (e.g., lists nested inside cells).
    try:
        if series.nunique() < min(20, max(len(series) * 0.05, 2)):
            return "categorical"
    except TypeError:
        # Unhashable values (lists, dicts inside cells) → treat as text
        return "text"
    return "text"


def _suggest_role(col_name: str, dtype: str) -> tuple:
    """Heuristically suggest a role and confidence for a column."""
    name_lower = col_name.lower()

    # ID-like columns
    if name_lower in ("id", "uid", "uuid") or name_lower.endswith("_id"):
        return "entity_id", 0.9

    # Timestamps
    if dtype == "datetime" or any(kw in name_lower for kw in ("date", "time", "timestamp", "created", "updated")):
        return "timestamp", 0.85

    # Labels / targets
    if any(kw in name_lower for kw in ("label", "target", "class", "outcome", "result", "status")):
        return "label", 0.8

    # Features / properties (default for numeric)
    if dtype == "numeric":
        return "feature", 0.7

    return "property", 0.5


def _detect_domain_from_columns(columns: List[str]) -> Optional[str]:
    """Attempt to detect the domain from column names."""
    col_text = " ".join(c.lower() for c in columns)

    domain_signals = {
        "healthcare": ["patient", "diagnosis", "treatment", "drug", "dosage", "symptom", "vitals", "tumor", "cancer"],
        "sports": ["athlete", "pace", "distance", "heart_rate", "stamina", "race", "training"],
        "finance": ["stock", "price", "volume", "portfolio", "return", "ticker", "market"],
        "environment": ["species", "population", "temperature", "co2", "emission", "wildfire"],
        "logistics": ["shipment", "warehouse", "delivery", "route", "inventory"],
    }

    best_domain = None
    best_count = 0
    for domain, keywords in domain_signals.items():
        count = sum(1 for kw in keywords if kw in col_text)
        if count > best_count:
            best_count = count
            best_domain = domain

    return best_domain if best_count >= 1 else None


def _parse_json_to_dataframe(contents: bytes):
    """Parse JSON content into a pandas DataFrame.

    Handles four shapes:
    1. JSON array of objects  ``[{...}, ...]``  -> direct DataFrame
    2. Flat object            ``{"col": val}``  -> single-row DataFrame
    3. Nested object with arrays of dicts ``{"key": [{...}, ...]}``
       -> pick the largest array and normalise it, merging others.
    4. Mixed nested object (arrays of arrays, scalars, nested dicts, and
       arrays of dicts) -> extract the best array of dicts; fall back to
       flattening scalar/dict fields into a single-row DataFrame.
    """
    import pandas as pd

    raw = json.loads(contents)

    # Case 1: top-level list
    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict):
            return pd.json_normalize(raw)
        # List of primitives / lists -> wrap into a DataFrame
        return pd.DataFrame({"value": raw})

    # Case 2 / 3 / 4: top-level dict
    if isinstance(raw, dict):
        # Find all top-level keys whose values are lists of dicts
        array_keys = {
            k: v for k, v in raw.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)
        }

        if array_keys:
            # Pick the largest array of dicts
            primary_key = max(array_keys, key=lambda k: len(array_keys[k]))
            df = pd.json_normalize(array_keys[primary_key])

            # If there are other arrays of the same length, merge them
            for k, arr in array_keys.items():
                if k == primary_key:
                    continue
                try:
                    other_df = pd.json_normalize(arr)
                    if len(other_df) == len(df):
                        other_df.columns = [f"{k}_{c}" for c in other_df.columns]
                        df = pd.concat([df, other_df], axis=1)
                except Exception:
                    pass  # Skip un-parseable sub-arrays

            return df

        # No arrays of dicts found.  Flatten scalars and simple nested
        # dicts into a single-row DataFrame (skip deeply nested structures
        # like 2-D arrays that cannot be flattened).
        flat = {}
        for k, v in raw.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                flat[k] = v
            elif isinstance(v, dict):
                # Flatten one level: grid_size.rows, grid_size.cols, etc.
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (str, int, float, bool)) or sub_v is None:
                        flat[f"{k}.{sub_k}"] = sub_v
            # Skip lists-of-lists, deeply nested structures, etc.

        if flat:
            return pd.DataFrame([flat])

        # Last resort: try pd.json_normalize on the whole thing
        return pd.json_normalize(raw)

    raise ValueError("JSON root must be an array or object")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=DataUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_data(file: UploadFile = File(...)):
    """
    Upload a CSV, JSON, or Excel file for analysis.

    The endpoint reads the file, infers column types, suggests
    entity/property mappings, and returns a structured summary
    that the frontend can use to configure twin creation.
    """
    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    file_type = _infer_file_type(file.filename)

    # Read file contents
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # Parse into a pandas DataFrame
    try:
        import pandas as pd
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="pandas is required for data upload but is not installed",
        )

    try:
        if file_type == "csv":
            df = pd.read_csv(io.BytesIO(contents))
        elif file_type == "json":
            df = _parse_json_to_dataframe(contents)
        elif file_type == "excel":
            try:
                df = pd.read_excel(io.BytesIO(contents))
            except ImportError:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="openpyxl is required for Excel files but is not installed",
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse file: {str(exc)}",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file contains no data rows",
        )

    # Build column info
    columns: List[ColumnInfo] = []
    suggested_mappings: List[SuggestedMapping] = []

    for col in df.columns:
        series = df[col]
        dtype = _infer_dtype(series)

        # Grab a few non-null sample values
        sample = series.dropna().head(5).tolist()
        # Ensure JSON-serialisable
        sample = [str(v) if not isinstance(v, (int, float, bool, str)) else v for v in sample]

        try:
            unique_count = int(series.nunique())
        except TypeError:
            unique_count = 0  # Unhashable types (lists, dicts in cells)

        columns.append(ColumnInfo(
            name=str(col),
            dtype=dtype,
            sample_values=sample,
            null_count=int(series.isnull().sum()),
            unique_count=unique_count,
        ))

        role, confidence = _suggest_role(str(col), dtype)
        suggested_mappings.append(SuggestedMapping(
            column=str(col),
            suggested_role=role,
            confidence=confidence,
        ))

    # Preview (first 5 rows, all columns)
    preview = json.loads(df.head(5).to_json(orient="records", date_format="iso"))

    # Domain detection
    detected_domain = _detect_domain_from_columns([str(c) for c in df.columns])

    return DataUploadResponse(
        upload_id=str(uuid.uuid4()),
        filename=file.filename,
        file_type=file_type,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        suggested_mappings=suggested_mappings,
        preview=preview,
        detected_domain=detected_domain,
        uploaded_at=datetime.utcnow(),
    )
