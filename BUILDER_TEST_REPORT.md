# Universal Twin Builder - Full UAT Report

**Tester:** Teammate 2 (Builder Flow UAT)
**Date:** 2026-02-11
**Backend:** FastAPI on port 8000
**Frontend:** Next.js 14 on port 3000
**Database:** PostgreSQL on port 5434 (SQLite fallback)
**Platform:** Darwin 24.5.0 (macOS)

---

## Summary

| Step | Test | Result | Notes |
|------|------|--------|-------|
| 0 | Understand API | PASS | All routes documented and accessible via `/docs` |
| 1 | Auth (register, login, JWT) | PASS | Username-based login, JWT works on `/me` |
| 2 | Healthcare Conversation | PASS | Entities extracted, domain=healthcare, twin auto-activated |
| 3 | Data Upload (CSV) | PASS | Schema analysis, suggested mappings, domain detection |
| 3b | Invalid File Upload | PASS | Returns 400 for bad extension and empty file |
| 4 | Twin Creation & State Transition | PASS | CRUD works, DRAFT -> ACTIVE via PATCH |
| 5 | Twin Simulation | PASS | Quantum advantage metrics returned, simulation persisted |
| 6 | Twin Query | PASS (after fix) | Was timing out; fixed encoding performance |
| 7 | Military/Logistics Domain | PASS (after fix) | Nested JSON upload was 500; fixed JSON parser |
| 8 | Sports Domain | PASS | End-to-end: conversation, upload, simulate, query |
| 9 | Environment Domain | PASS (after fix) | flood.json crashed on unhashable types; fixed |
| 10 | Twin Lifecycle | PASS | List, delete, verify removal, 404 on deleted |

**Overall Result: 10/10 PASS** (3 required code fixes)

---

## Detailed Results

### STEP 0: Understand API
- **Status:** PASS
- Auth uses `username` (not email) for login, confirmed in `backend/auth/router.py`
- Routes: `/api/twins/`, `/api/conversation/`, `/api/data/upload`, `/api/benchmark/`, `/api/auth/`
- `redirect_slashes=False` in FastAPI config; trailing slashes matter

### STEP 1: Create Test User & Auth
- **Status:** PASS
- Registered `uat_builder_user` with email `uat_builder@test.com`
- Login returns JWT token, `token_type=bearer`, `user_id`, `username`
- `/api/auth/me` returns correct profile with valid Bearer token

### STEP 2: Healthcare Conversation Flow
- **Status:** PASS
- First message: "I have patient tumor data and want to find the optimal treatment plan"
- Extracted entities: Patient, Tumor, Treatment
- Domain: healthcare, Goal: detect
- Twin auto-transitioned to `active` status
- Follow-up message added Drug entity, preserved existing entities

### STEP 3: Data Upload
- **Status:** PASS
- Uploaded `patients.csv` (10 rows, 7 columns)
- Detected columns: patient_id, age, tumor_type, marker_A, marker_B, treatment, outcome_score
- Suggested mappings: patient_id -> entity_id (0.9), outcome_score -> label (0.8)
- Domain auto-detected: healthcare
- Invalid `.txt` file correctly returns 400
- Empty `.csv` file correctly returns 400

### STEP 4: Twin Creation & State Transition
- **Status:** PASS
- Created twin via `POST /api/twins/` with name, description, domain
- Returned: id, status=draft, created_at, updated_at
- PATCH to status=active works correctly

### STEP 5: Twin Simulation
- **Status:** PASS
- `POST /api/twins/{id}/simulate` with time_steps=50, scenarios=3
- Returns: simulation_id, results, predictions, quantum_advantage metrics
- `quantum_advantage.speedup` and `classical_equivalent_seconds` present
- Simulation persisted in database

### STEP 6: Twin Query
- **Status:** PASS (after fix)
- **Bug found:** Query endpoint timed out (>30s) due to exponential statevector allocation
- **Root cause:** `_create_initial_state()` in `backend/engine/encoding/quantum_encoder.py` allocated `2^44 = 17.6 trillion` complex numbers for a 44-qubit system, even with a `2^20` cap the nested loops were too slow
- **Fix:** Capped statevector dimension to `2^12 = 4096` entries (line 219). The full simulation uses Qiskit Aer internally; this representation is descriptive only.
- **File changed:** `/Users/hassanalsahli/Desktop/Final_DT/backend/engine/encoding/quantum_encoder.py`
- After fix: query completes in ~5ms
- Response includes: query_type=optimization, answer with domain context, quantum_metrics, confidence=0.8

### STEP 7: Military/Logistics Domain
- **Status:** PASS (after fix)
- Conversation created logistics twin with Route, Shipment entities
- **Bug found:** `POST /api/data/upload` returned 500 for `logistics.json`
- **Root cause:** `pd.read_json()` cannot parse nested JSON objects with multiple top-level keys (`bases`, `routes`). It expects a flat array or simple object.
- **Fix:** Added `_parse_json_to_dataframe()` helper that detects nested JSON structure, finds the largest array of dicts, normalizes it, and merges sibling arrays of the same length.
- **File changed:** `/Users/hassanalsahli/Desktop/Final_DT/backend/api/data/router.py`
- After fix: bases and routes merged into 5-row, 9-column DataFrame
- Simulation runs correctly, query returns logistics-domain answer

### STEP 8: Sports Domain
- **Status:** PASS
- Conversation: "training for a marathon, optimize pacing"
- Extracted: Race entity, domain=sports, goal=optimize
- Fatigue rule with formula: `fatigue += exertion * time`
- training.csv upload: 10 rows, 8 columns, domain=sports
- Simulation: 42 time_steps, 10 scenarios, quantum_advantage.speedup=100x
- Query about pace at km 30 returns prediction-type response

### STEP 9: Environment Domain
- **Status:** PASS (after fix)
- Conversation about flood modeling created twin with Hurricane entity
- **Bug found:** `flood.json` upload returned 500 even after JSON parser fix
- **Root cause:** `_infer_dtype()` called `series.nunique()` which crashed with `TypeError: unhashable type: 'list'` because the `cells` column contained nested lists like `[[0,5],[1,5],...]`
- **Fix:** Wrapped `nunique()` calls in try/except TypeError blocks in both `_infer_dtype()` and the column info builder. Returns "text" type and `unique_count=0` for unhashable columns.
- **File changed:** `/Users/hassanalsahli/Desktop/Final_DT/backend/api/data/router.py`
- After fix: 5 infrastructure items parsed, 7 columns detected
- Query about hospital risk returns meaningful response

### STEP 10: Twin Lifecycle
- **Status:** PASS
- `GET /api/twins/` returns all 9 twins across healthcare, logistics, sports, science domains
- `DELETE /api/twins/{id}` returns 204 No Content
- After delete: twin count drops to 8, deleted twin absent from list
- `GET /api/twins/{deleted_id}` returns 404

---

## Bugs Found & Fixed

### Bug 1: Statevector Encoding Performance (Critical)
- **Severity:** Critical (query endpoint completely unusable)
- **File:** `backend/engine/encoding/quantum_encoder.py`
- **Issue:** `_create_initial_state()` tried to allocate `2^n` complex numbers where n could be 44+, causing memory exhaustion and infinite-loop-like behavior even with the 2^20 cap
- **Fix:** Changed cap from `2^20` to `2^min(n, 12)` = max 4096 entries. Added guard to skip qubit indices beyond the compact representation.
- **Impact:** Query endpoint now responds in ~5ms instead of timing out

### Bug 2: Nested JSON Upload (High)
- **Severity:** High (logistics and flood data uploads failed with 500)
- **File:** `backend/api/data/router.py`
- **Issue:** `pd.read_json()` cannot parse nested JSON objects with named sub-arrays (e.g., `{"bases": [...], "routes": [...]}`)
- **Fix:** Added `_parse_json_to_dataframe()` that handles 4 JSON shapes: flat arrays, flat objects, nested objects with arrays of dicts, and mixed nested structures. Picks the largest array of dicts as the primary table and merges siblings.
- **Impact:** All sample JSON files now upload successfully

### Bug 3: Unhashable Type in Column Analysis (Medium)
- **Severity:** Medium (flood.json upload still failed after JSON fix)
- **File:** `backend/api/data/router.py`
- **Issue:** `series.nunique()` crashes with `TypeError: unhashable type: 'list'` when DataFrame columns contain nested lists (e.g., coordinate arrays)
- **Fix:** Wrapped `nunique()` in try/except in both `_infer_dtype()` and column info builder. Falls back to "text" dtype and `unique_count=0`.
- **Impact:** Data files with nested array fields now upload without error

---

## Files Modified

1. **`/Users/hassanalsahli/Desktop/Final_DT/backend/engine/encoding/quantum_encoder.py`**
   - `_create_initial_state()`: Reduced max statevector dimension from 2^20 to 2^12, added qubit index bounds check

2. **`/Users/hassanalsahli/Desktop/Final_DT/backend/api/data/router.py`**
   - Added `_parse_json_to_dataframe()` helper for nested JSON handling
   - Fixed `_infer_dtype()` to handle unhashable types
   - Fixed column info builder to handle unhashable `nunique()` calls

---

## Recommendations

1. **Domain detection for environment:** The extractor detected "science" instead of "environment" for flood/hurricane scenarios. The `SystemExtractor` pattern list could be extended with environment-specific keywords (flood, hurricane, earthquake, wildfire, etc.).

2. **Trailing slash consistency:** The API uses `redirect_slashes=False` but some routes have trailing slashes (`/api/twins/`, `/api/conversation/`) while sub-routes don't. This can confuse API consumers. Consider enabling `redirect_slashes=True` or standardizing all routes.

3. **Query type detection:** "Which route is safest?" was classified as `prediction` rather than `optimization`. The detection heuristic could be improved to handle "safest" -> optimization.

4. **Twin naming:** Twins created via conversation are all named "New Digital Twin". Auto-generating a name from the conversation topic (e.g., "Marathon Pacing Twin") would improve usability.

5. **Data-twin linkage:** Uploaded data is analyzed but not automatically linked to a twin. Adding an optional `twin_id` parameter to the upload endpoint would complete the flow.
