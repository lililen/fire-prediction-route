# CA Fire Detection & Personalized Evacuation

A 24-hour hackathon project built for **Hack for Humanity 2026**. Given a user's GPS location in California, the app predicts real-time wildfire burn probability and air quality hazard, then computes a personalized A\* evacuation route that avoids fire zones and accounts for disabilities, pets, kids, and medications.

> **Status: Unfinished hackathon prototype.** Core ML pipeline and routing are functional; the frontend map panel and some integrations are incomplete.

---

## What It Does

1. **Takes the user's GPS location** via a React form with personal constraints (disability, pets, kids, medications).
2. **Fetches real-time data** — nearest CAL FIRE perimeter, live wind forecast, and terrain elevation.
3. **Runs two ML models in sequence:**
   - Model 1 predicts the probability that fire reaches the user's location.
   - Model 2 predicts air quality / smoke hazard class (low / medium / high).
4. **Builds a 25×25 hazard grid** (~1.1 km radius) around the user using the model outputs.
5. **Runs A\* pathfinding** on the grid to find the safest evacuation route to the nearest safe zone, treating fire and high-hazard cells as blocked and smoke cells as costly (or blocked for disabled users).
6. **Returns AI-generated evacuation guidance** (via OpenAI GPT-4o) tailored to the user's constraints.

---

## Project Structure

```
fire-prediction-route/
├── backend/
│   ├── app.py                          # Flask entry point
│   ├── ai.py                           # OpenAI GPT-4o evacuation guidance
│   ├── pathfinder.py                   # Hazard grid builder + A* router
│   ├── routes/
│   │   ├── predict.py                  # POST /api/predict endpoint
│   │   ├── inference.py                # FireHazardService (runs both ML models)
│   │   ├── realtime_data.py            # CAL FIRE / Visual Crossing / USGS APIs
│   │   └── feature_utils.py            # Shared feature engineering (both models)
│   ├── model_1/
│   │   ├── fire_predict_model.py       # Model 1 training script (XGBoost + RF)
│   │   ├── fire_spread_model.pkl       # Trained XGBoost — burn probability
│   │   └── fire_spread_features.pkl    # Feature column order
│   ├── model_2/
│   │   ├── hazard_score_model.py       # Model 2 training script (XGBoost + RF)
│   │   ├── hazard_model.pkl            # Trained XGBoost — hazard class
│   │   └── hazard_features.pkl         # Feature column order
│   ├── datasets_train/                 # 30-fire training CSVs (48h each)
│   ├── test_mock2.py                   # Full mock test suite (no live APIs needed)
│   └── requirements.txt
└── frontend/
    └── src/
        ├── App.js
        ├── components/
        │   ├── EvacuationForm.js       # User constraints form
        │   ├── GoogleMapPanel.js       # Map display (unfinished)
        │   ├── LocationDisplay.js      # GPS location via cookies
        │   └── ResultsDisplay.js       # Renders ML output + route
        └── index.js
```

---

## Lillian Le — Contributions

All backend ML and data pipeline work. Below is a breakdown by commit/component.

### ML Model 1 — Fire Spread / Burn Probability
**File:** [backend/model_1/fire_predict_model.py](backend/model_1/fire_predict_model.py)

Trained two classifiers (XGBoost + Random Forest) on `FINAL_fire_spread_ml_30fires_48h.csv`, a dataset covering 30 California wildfires over 48-hour windows. The target label is `burned_label` (binary: did this grid cell burn?).

Key engineering decisions:
- **Wind features:** squared wind speed (captures nonlinear fire acceleration per Rothermel model), sin/cos encoding of wind direction (to preserve circularity), wind-hour interaction (daytime wind variation).
- **Proximity features:** wind speed × downwind alignment (energy directed at the point), wind speed / distance-to-fire-front (close high-wind is more dangerous).
- **Class imbalance:** handled via `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (RF) since most cells don't burn.
- Trained on the full 30-fire dataset. Output: `fire_spread_model.pkl`, `fire_spread_rf_model.pkl`, `fire_spread_with_probs.csv` (burn probabilities used as input to Model 2).

### ML Model 2 — Hazard / Air Quality Predictor
**File:** [backend/model_2/hazard_score_model.py](backend/model_2/hazard_score_model.py)

Three-class classifier (0=low, 1=medium, 2=high hazard) trained on `FINAL_hazard_ml_30fires_48h.csv`. Model 1's burn probability feeds into this model as a smoke proxy.

Key engineering decisions:
- **Smoke proxy formula:** `burn_prob × (0.25 + 0.75 × downwind_factor) × exp(-dist / plume_length)` — exponential decay from the fire front, amplified downwind.
- **Smoke interactions:** `smoke × wind_speed` (ember transport), `smoke × slope_proxy` (fire spreads faster uphill), `smoke / dist_to_front` (proximity penalty).
- Multi-class output: `objective="multi:softprob"` gives `p_low`, `p_med`, `p_high` per cell. Heatmap weight = `0.5 × p_med + 1.0 × p_high`.
- Output: `hazard_model.pkl`, `hazard_rf_model.pkl`, `hazard_features.pkl`.

### Real-Time Data Pipeline
**File:** [backend/routes/realtime_data.py](backend/routes/realtime_data.py)

Integrated three external data sources and built the `build_point_next_hour()` function that assembles a complete feature dict for any GPS coordinate:

- **CAL FIRE ArcGIS FeatureServer** (public, no key): queries fire perimeters within 100 km, finds the nearest one, extracts `dist_to_front_km`, `dist_to_center_km`, `r_boundary_km`.
- **Visual Crossing Timeline API** (key via `VISUAL_CROSSING_API_KEY`): fetches hourly wind forecast, converts km/h → m/s, flips "from" direction to "toward" convention, computes U/V wind components.
- **USGS Elevation Point Query** (public, no key): samples elevation at 4 neighbours (N/S/E/W, 90m apart) to compute a slope proxy in [0, 1], LRU-cached to avoid repeated calls.
- Computes `downwind_alignment` (dot product of wind direction and center→point bearing) so the model knows whether the user is upwind or downwind of the fire.

### Feature Engineering (Shared)
**File:** [backend/routes/feature_utils.py](backend/routes/feature_utils.py)

Centralized `build_fire_features()` and `build_hazard_features()` so training scripts and the inference path use identical transformations. Also includes `align_features()` which safely reorders and zero-fills columns to match the saved feature list, preventing silent shape mismatches at inference time.

### Model Inference Wrapper
**File:** [backend/routes/inference.py](backend/routes/inference.py)

`FireHazardService` loads both `.pkl` models once at startup and exposes `predict_one(point_dict)` — runs Model 1, injects `burn_probability` into the point dict, then runs Model 2. Returns `burn_probability`, `hazard_pred_class`, `p_low/p_med/p_high`, and `heat_weight`.

### Backend Integration & Endpoint
**File:** [backend/routes/predict.py](backend/routes/predict.py)

Flask Blueprint (`POST /api/predict`): orchestrates the full pipeline — CAL FIRE lookup → real-time feature build → ML inference → AI guidance → A\* routing. Appends every request to a JSONL log file for offline analysis. Validates input, returns structured JSON.

### Validation
**Commits:** `validation proof`, `model1 and model2 use all dataset for training`

Generated validation screenshots after switching from train/test splits to training on the full 30-fire dataset (screenshots in `backend/model_1/` and `backend/model_2/`). Tracked large `.pkl` files via Git LFS.

### Mock Test Suite
**File:** [backend/test_mock2.py](backend/test_mock2.py)

Full offline test suite (no live APIs or `.pkl` files required). Patches all external calls with realistic CA wildfire scenario data (Azusa foothills, 6.2 m/s westerly wind). Tests cover:
- `EvacuationRouter`: key structure, reachability under normal conditions, fire-cell blocking, disability flag effect on path, coordinate validity, grid stats sum.
- `/api/predict` endpoint: 200 responses, full response key structure, ML block validity, evacuation block validity, guidance string, all-checkbox payload, error handling (missing/invalid lat-lon, empty body), non-fatal routing failure.

---

## Team

| Name | Role |
|---|---|
| **Lillian Le** | ML models (fire spread + hazard), real-time data pipeline, feature engineering, inference layer, backend integration, mock test suite |
| **Pedro Aguirre** | A\* pathfinding algorithm, hazard grid implementation ([pathfinder.py](backend/pathfinder.py)) |
| **Franco Garcia** | AI prompt engineering ([ai.py](backend/ai.py)), ElevenLabs TTS integration |
| **Collin Fiske** | Project scaffolding, initial commit, frontend skeleton |

---

## Setup

### Prerequisites
- Python >= 3.9
- Node.js >= 18

### Environment Variables (`.env` in project root)
```
OPENAI_API_KEY=...
VISUAL_CROSSING_API_KEY=...
```

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                   # Flask on http://localhost:5000
```

Run offline tests (no APIs needed):
```bash
cd backend
python test_mock2.py
```

### Frontend
```bash
cd frontend
npm install
npm start                       # React on http://localhost:3000
```

---

## Data

Training data is in `backend/datasets_train/` — 5 CSVs derived from 30 California wildfires, each covering 48-hour windows:
- `FINAL_fire_spread_ml_30fires_48h.csv` — Model 1 training set
- `FINAL_hazard_ml_30fires_48h.csv` — Model 2 training set
- `FINAL_fire_metadata_30fires.csv`, `FINAL_fire_perimeter_polygons_*`, `FINAL_smoke_polygons_*`

---

## Known Gaps (Hackathon Scope)

- Google Maps panel is wired but incomplete — route polyline not rendering.
- Safe zone goals are heuristic grid-edge cells; production would use Google Places API (shelters, hospitals).
- Model training used full dataset (no held-out test set) due to time constraints — validation was done via hardcoded scenario checks.
- No authentication, no persistent user profiles.
