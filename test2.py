import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
# backend/              ← test_mock2.py, pathfinder.py, ai.py, app.py live here
# backend/routes/       ← inference.py, predict.py, realtime_data.py, feature_utils.py

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTES_DIR  = os.path.join(BACKEND_DIR, "routes")

for p in (BACKEND_DIR, ROUTES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# ─────────────────────────────────────────────────────────────────────────────
# Shared mock data — grounded in realistic CA wildfire scenario
# Location: foothills east of Los Angeles (Azusa area)
# Fake fire: "MOCK FIRE" 2023, moderate wind from the west
# ─────────────────────────────────────────────────────────────────────────────

MOCK_LAT = 34.1336
MOCK_LON = -117.9070

MOCK_PERIMETER_METRICS = {
    "fire_name":         "MOCK FIRE",
    "year":              2023,
    "irwinid":           "MOCK-IRWIN-001",
    "inc_num":           "CA-ANF-000001",
    "center_lat":        34.12,
    "center_lon":        -117.88,
    "r_boundary_km":     4.2,
    "dist_to_front_km":  1.8,
    "dist_to_center_km": 2.6,
}

MOCK_WIND = {
    "time_utc":        datetime(2023, 8, 14, 15, 0, tzinfo=timezone.utc),
    "wind_speed_ms":   6.2,    # ~22 km/h from west — typical CA offshore wind
    "wind_dir_deg_to": 90.0,   # blowing east toward the hills
}

MOCK_ML_OUTPUT = {
    "burn_probability":  0.61,
    "hazard_pred_class": 2,
    "p_low":             0.08,
    "p_med":             0.31,
    "p_high":            0.61,
    "heat_weight":       0.765,
}

MOCK_SLOPE = 0.18


def _make_mock_fire_svc(ml_output=None):
    svc = MagicMock()
    svc.predict_one.return_value = ml_output or MOCK_ML_OUTPUT
    return svc


# ─────────────────────────────────────────────────────────────────────────────
# Patch targets — keyed to actual module locations
#
#   routes/realtime_data.py  → "realtime_data.<fn>"
#   backend/pathfinder.py    → "pathfinder.<fn>"
#   routes/inference.py      → "inference.<fn>"
#   routes/predict.py        → "predict.<fn>"
# ─────────────────────────────────────────────────────────────────────────────

def _all_patches():
    return [
        # realtime_data external calls (used by build_point_next_hour)
        patch("realtime_data.get_nearest_calfire_perimeter_metrics", return_value=MOCK_PERIMETER_METRICS),
        patch("realtime_data.vc_hourly_wind",                        return_value=MOCK_WIND),
        patch("realtime_data.slope_proxy_from_elevation",            return_value=MOCK_SLOPE),
        patch("realtime_data._epqs_elevation_m",                     return_value=320.0),

        # pathfinder external calls (used by build_hazard_grid)
        patch("pathfinder.get_nearest_calfire_perimeter_metrics", return_value=MOCK_PERIMETER_METRICS),
        patch("pathfinder.vc_hourly_wind",                        return_value=MOCK_WIND),
        patch("pathfinder.slope_proxy_from_elevation",            return_value=MOCK_SLOPE),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Visualization (adapted from test.py)
#
# Uses the REAL GridCell grid captured during _run_router (via side-effect
# patches on build_hazard_grid and _find_safe_zone_candidates) so every cell
# type, all 3 candidate safe zones, and the true path are shown accurately.
#
# Cell encoding for imshow:
#   0  free       (white)
#   1  fire       (red)
#   2  smoke/air  (orange)
#   3  high haz   (dark red / maroon)
#
# Start / goals / path are scatter+line overlays (never obscure cell colour).
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE   = 25   # must match pathfinder GRID_RADIUS * 2 + 1
GRID_RADIUS = 12   # centre cell index (user location)

# Maps GridCell.cell_type → int value for imshow
_CELL_TO_INT = {".": 0, "S": 0, "G": 0, "A": 2, "F": 1, "H": 3, "#": 1}


def _grid_to_array(grid) -> list[list[int]]:
    """Convert a list[list[GridCell]] to a GRID_SIZE×GRID_SIZE int array."""
    size = len(grid)
    return [
        [_CELL_TO_INT.get(grid[r][c].cell_type, 0) for c in range(size)]
        for r in range(size)
    ]


def _overlay_markers(
    ax,
    start_cr: tuple,
    all_goal_crs: list,
    chosen_goal_cr: tuple | None = None,
    path_cr: list | None = None,
):
    """
    Draw overlays on a hazard-grid axes.

    Parameters
    ----------
    start_cr        : (col, row) of the user start cell
    all_goal_crs    : list of (col, row) for ALL candidate safe zones
    chosen_goal_cr  : (col, row) of the A*-reached goal (highlighted differently)
    path_cr         : list of {"col":, "row":} dicts for the evacuation path
    """
    # Start — purple circle
    ax.scatter([start_cr[0]], [start_cr[1]],
               s=180, c="purple", marker="o", edgecolors="black", linewidths=1.2,
               zorder=6, label="Start (you)")

    # Unchosen candidate safe zones — hollow green stars
    unchosen = [g for g in all_goal_crs if g != chosen_goal_cr]
    if unchosen:
        ax.scatter([g[0] for g in unchosen], [g[1] for g in unchosen],
                   s=200, facecolors="none", edgecolors="lime", linewidths=2.0,
                   marker="*", zorder=6, label=f"Alt shelter ({len(unchosen)})")

    # Chosen / reached safe zone — solid green star
    if chosen_goal_cr:
        ax.scatter([chosen_goal_cr[0]], [chosen_goal_cr[1]],
                   s=260, c="lime", marker="*", edgecolors="black", linewidths=1.2,
                   zorder=7, label="Chosen shelter")

    # Evacuation path — blue line
    if path_cr:
        px = [p["col"] for p in path_cr]
        py = [p["row"] for p in path_cr]
        ax.plot(px, py, color="deepskyblue", linewidth=2.5, zorder=5,
                label=f"Path ({len(path_cr)} steps)")


def visualize_router_result(
    result: dict,
    real_grid=None,           # list[list[GridCell]] captured from pathfinder
    all_goal_cells: list | None = None,  # all candidate (col,row) tuples
    has_disability: bool = False,
    title_prefix: str = "",
) -> None:
    """
    Render a side-by-side hazard-grid plot.  Mirrors show_side_by_side() in test.py.

    Parameters
    ----------
    result          : dict from EvacuationRouter.run()
    real_grid       : the actual list[list[GridCell]] from build_hazard_grid
                      (captured via side-effect patch in _run_router_with_grid).
                      If None, falls back to all-white grid with a warning.
    all_goal_cells  : all candidate safe-zone (col, row) tuples — shows the
                      unchosen shelters as hollow stars.
    has_disability  : used for the info-bar label
    title_prefix    : prepended to the figure suptitle
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError:
        print("  (matplotlib not installed — skipping visualisation)")
        return

    CMAP = ListedColormap(["white", "red", "orange", "darkred"])

    # ── Build base array ───────────────────────────────────────────────────
    if real_grid is not None:
        base_np = np.array(_grid_to_array(real_grid), dtype=int)
    else:
        print("  WARNING: no real grid available — grid will appear blank")
        base_np = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    path_cr        = result.get("path_col_row", [])
    safe_zone      = result.get("safe_zone")
    reachable      = result.get("reachable", False)
    cost           = result.get("route_cost")
    stats          = result.get("grid_stats", {})

    # Start = first path cell, or grid centre
    start_cr = (path_cr[0]["col"], path_cr[0]["row"]) if path_cr else (GRID_RADIUS, GRID_RADIUS)

    # Chosen goal = last path cell
    chosen_goal_cr = (path_cr[-1]["col"], path_cr[-1]["row"]) if (path_cr and reachable) else None

    # All candidate safe zones (as list of (col, row) tuples)
    goals_list = list(all_goal_cells) if all_goal_cells else (
        [chosen_goal_cr] if chosen_goal_cr else []
    )

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    suptitle = (
        f"{title_prefix}"
        f"Hazard Grid — {MOCK_PERIMETER_METRICS['fire_name']} {MOCK_PERIMETER_METRICS['year']}  "
        f"| ({MOCK_LAT}, {MOCK_LON})"
    )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    extent = [-0.5, GRID_SIZE - 0.5, GRID_SIZE - 0.5, -0.5]

    scenarios = [
        (axes[0], False, "Hazard Grid  (all shelters, no path)"),
        (axes[1], True,  _right_title(reachable, safe_zone, cost)),
    ]

    for i, (ax, show_path, subtitle) in enumerate(scenarios):
        ax.imshow(base_np, cmap=CMAP, vmin=0, vmax=3,
                  interpolation="nearest", extent=extent)
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel("col  (west ← → east)", fontsize=9)
        if i == 0:
            ax.set_ylabel("row  (north ↑ ↓ south)", fontsize=9)
        ax.grid(True, linewidth=0.4, color="gray", alpha=0.5)
        _overlay_markers(
            ax,
            start_cr=start_cr,
            all_goal_crs=goals_list,
            chosen_goal_cr=chosen_goal_cr,
            path_cr=(path_cr if (show_path and reachable) else None),
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # ── Colour legend ─────────────────────────────────────────────────────
    legend_patches = [
        Patch(facecolor="white",   edgecolor="gray", label="Free / passable"),
        Patch(facecolor="red",                       label="Fire (blocked)"),
        Patch(facecolor="orange",                    label="Smoke / Air (costly)"),
        Patch(facecolor="darkred",                   label="High Hazard (blocked)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    # ── Info bar ──────────────────────────────────────────────────────────
    air_rule = "blocked" if has_disability else "allowed (+cost)"
    info = "   |   ".join([
        f"Disabled: {'YES' if has_disability else 'NO'}",
        f"Smoke rule: {air_rule}",
        f"Fire={stats.get('fire',0)}  Smoke={stats.get('smoke',0)}  "
        f"HighHaz={stats.get('high_hazard',0)}  Free={stats.get('free',0)}",
        f"Path waypoints: {len(path_cr)}   Shelters shown: {len(goals_list)}",
        f"Wind: {MOCK_WIND['wind_speed_ms']} m/s → {MOCK_WIND['wind_dir_deg_to']}°",
    ])
    fig.text(0.5, -0.08, info, ha="center", va="bottom", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.85))

    plt.show()


def _right_title(reachable: bool, safe_zone: dict | None, cost: float | None) -> str:
    if not reachable:
        return "No Reachable Route"
    label = safe_zone.get("label", "Safe Zone") if safe_zone else "Safe Zone"
    title = f"Evacuation Path → {label}"
    if cost is not None:
        title += f"  (cost={cost:.1f})"
    return title


# ─────────────────────────────────────────────────────────────────────────────
# Module-level store: (result_dict, real_grid, all_goal_cells, has_disability)
# ─────────────────────────────────────────────────────────────────────────────
_VIZ_RESULTS: dict[str, tuple] = {}


# ─────────────────────────────────────────────────────────────────────────────
# EvacuationRouter isolation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvacuationRouter(unittest.TestCase):

    def _run_router(self, has_disability=False, ml_output=None):
        """
        Run the router with all external calls mocked.

        Also intercepts build_hazard_grid and _find_safe_zone_candidates via
        wrapping patches so we capture:
          • the real GridCell grid   → for accurate per-cell visualisation
          • all candidate goal cells → to show unchosen shelters as hollow stars
        """
        import pathfinder as pf
        from pathfinder import EvacuationRouter

        svc = _make_mock_fire_svc(ml_output)

        # Storage for captured internals
        _captured = {"grid": None, "goals": None}

        # Wrap build_hazard_grid to intercept its return value
        _orig_build = pf.build_hazard_grid
        def _capturing_build(*args, **kwargs):
            g = _orig_build(*args, **kwargs)
            _captured["grid"] = g
            return g

        # Wrap _find_safe_zone_candidates to intercept the chosen goals
        _orig_goals = pf._find_safe_zone_candidates
        def _capturing_goals(*args, **kwargs):
            goals = _orig_goals(*args, **kwargs)
            _captured["goals"] = goals
            return goals

        patches = _all_patches() + [
            patch.object(pf, "build_hazard_grid",          _capturing_build),
            patch.object(pf, "_find_safe_zone_candidates", _capturing_goals),
        ]
        for p in patches:
            p.start()
        try:
            router = EvacuationRouter(
                center_lat=MOCK_LAT,
                center_lon=MOCK_LON,
                has_disability=has_disability,
                fire_svc=svc,
            )
            result = router.run()
        finally:
            for p in patches:
                p.stop()

        return result, _captured["grid"], _captured["goals"]

    def test_router_returns_expected_keys(self):
        result, _, _ = self._run_router()
        for key in ("reachable", "route_cost", "safe_zone", "path_latlon", "grid_stats"):
            self.assertIn(key, result, f"Missing key: {key}")
        print("✓ Router returns all expected keys")

    def test_router_reachable_under_normal_conditions(self):
        """Mixed terrain: fire core (east), smoke band, free outer ring — reachable route expected.

        _varied_predict reads the actual lat/lon from each point dict and converts
        back to grid (col, row) using the same formula as pathfinder, so the zone
        geometry is exact regardless of iteration order.

        Zone layout:
          F  fire:       eastern half (col ≥ 14), dist ≤ 5   burn_prob=0.75 → "F"
          H  high-haz:   central core,            dist ≤ 3   hazard_class=2 → "H"
          A  smoke:      middle ring,         3 < dist ≤ 7   hazard_class=1 → "A"
          .  free:       outer ring,              dist > 7   hazard_class=0 → "."
        """
        import math

        # Replicate pathfinder's latlon→col/row conversion so we can
        # identify each cell from its lat/lon without relying on call_count.
        _M_LAT = 111_320.0
        _M_LON = 111_320.0 * math.cos(math.radians(MOCK_LAT))
        CELL_M = 90.0

        def _latlon_to_cr(lat, lon):
            dy = (lat - MOCK_LAT) * _M_LAT
            dx = (lon - MOCK_LON) * _M_LON
            col = int(round(dx / CELL_M)) + GRID_RADIUS
            row = int(round(-dy / CELL_M)) + GRID_RADIUS
            return col, row

        def _varied_predict(point):
            col, row = _latlon_to_cr(float(point["lat"]), float(point["lon"]))
            dist = math.hypot(col - GRID_RADIUS, row - GRID_RADIUS)

            if col >= GRID_RADIUS + 2 and dist <= 5:       # fire: east, close
                return {**MOCK_ML_OUTPUT, "burn_probability": 0.75, "hazard_pred_class": 2, "heat_weight": 0.90}
            elif dist <= 3:                                 # high-hazard core
                return {**MOCK_ML_OUTPUT, "burn_probability": 0.30, "hazard_pred_class": 2, "heat_weight": 0.80}
            elif dist <= 7:                                 # smoke band
                return {**MOCK_ML_OUTPUT, "burn_probability": 0.20, "hazard_pred_class": 1, "heat_weight": 0.30}
            else:                                           # free outer ring
                return {**MOCK_ML_OUTPUT, "burn_probability": 0.04, "hazard_pred_class": 0, "heat_weight": 0.02}

        svc = MagicMock()
        svc.predict_one.side_effect = _varied_predict

        import pathfinder as pf_mod
        _captured = {"grid": None, "goals": None}
        _orig_build = pf_mod.build_hazard_grid
        _orig_goals = pf_mod._find_safe_zone_candidates
        def _cap_build(*a, **kw):
            g = _orig_build(*a, **kw); _captured["grid"] = g; return g
        def _cap_goals(*a, **kw):
            g = _orig_goals(*a, **kw); _captured["goals"] = g; return g

        from pathfinder import EvacuationRouter
        patches = _all_patches() + [
            patch.object(pf_mod, "build_hazard_grid",          _cap_build),
            patch.object(pf_mod, "_find_safe_zone_candidates", _cap_goals),
        ]
        for p in patches: p.start()
        try:
            result = EvacuationRouter(
                center_lat=MOCK_LAT, center_lon=MOCK_LON,
                has_disability=False, fire_svc=svc,
            ).run()
        finally:
            for p in patches: p.stop()

        self.assertTrue(result["reachable"], "Expected a reachable route")
        self.assertIsNotNone(result["safe_zone"])
        self.assertGreater(len(result["path_latlon"]), 0)
        _VIZ_RESULTS["normal"] = (result, _captured["grid"], _captured["goals"], False)
        stats = result["grid_stats"]
        print(f"✓ Router found route — {len(result['path_latlon'])} waypoints, "
              f"cost={result['route_cost']:.1f}, "
              f"grid: F={stats.get('fire',0)} H={stats.get('high_hazard',0)} "
              f"A={stats.get('smoke',0)} free={stats.get('free',0)}, "
              f"safe zone=({result['safe_zone']['lat']:.4f}, {result['safe_zone']['lon']:.4f})")

    def test_router_blocks_fire_cells(self):
        """High burn probability should fill the grid with blocked cells."""
        result, grid, goals = self._run_router(ml_output={
            **MOCK_ML_OUTPUT,
            "burn_probability":  0.90,
            "hazard_pred_class": 2,
            "heat_weight":       0.95,
        })
        stats   = result["grid_stats"]
        blocked = stats.get("fire", 0) + stats.get("high_hazard", 0)
        self.assertGreater(blocked, 100, "Expected majority of grid to be blocked")
        _VIZ_RESULTS["blocked"] = (result, grid, goals, False)
        print(f"✓ High burn_probability → {blocked} blocked cells in grid")

    def test_router_disability_flag_affects_path(self):
        """Smoke cells are blocked entirely for disabled users."""
        smoke_output = {
            **MOCK_ML_OUTPUT,
            "burn_probability":  0.2,
            "hazard_pred_class": 1,
            "heat_weight":       0.3,
        }
        result_normal,   grid_n, goals_n = self._run_router(has_disability=False, ml_output=smoke_output)
        result_disabled, grid_d, goals_d = self._run_router(has_disability=True,  ml_output=smoke_output)
        _VIZ_RESULTS["disability_normal"]   = (result_normal,   grid_n, goals_n, False)
        _VIZ_RESULTS["disability_disabled"] = (result_disabled, grid_d, goals_d, True)
        print(f"✓ Disability flag test:")
        print(f"    Normal   — reachable={result_normal['reachable']},   "
              f"cost={result_normal.get('route_cost')}")
        print(f"    Disabled — reachable={result_disabled['reachable']}, "
              f"cost={result_disabled.get('route_cost')}")

    def test_path_latlon_are_valid_coordinates(self):
        """Every waypoint should be a valid California lat/lon."""
        result, _, _ = self._run_router(ml_output={
            **MOCK_ML_OUTPUT,
            "burn_probability":  0.2,
            "hazard_pred_class": 0,
        })
        if result["reachable"]:
            for pt in result["path_latlon"]:
                self.assertIn("lat", pt)
                self.assertIn("lon", pt)
                self.assertGreater(pt["lat"],  30.0)
                self.assertLess(pt["lat"],     42.0)
                self.assertGreater(pt["lon"], -125.0)
                self.assertLess(pt["lon"],    -114.0)
            print(f"✓ All {len(result['path_latlon'])} waypoints are valid CA coordinates")
        else:
            print("  (skipped coordinate check — no reachable route)")

    def test_grid_stats_sum_to_grid_size(self):
        """Grid stats cell counts should sum to 25×25 = 625."""
        result, _, _ = self._run_router()
        stats  = result["grid_stats"]
        total  = sum(stats.values())
        self.assertEqual(total, 625, f"Expected 625 cells, got {total}: {stats}")
        print(f"✓ Grid stats sum to 625: {stats}")


# ─────────────────────────────────────────────────────────────────────────────
# /api/predict pipeline tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint(unittest.TestCase):

    def setUp(self):
        self._pkl_patch = patch("inference.joblib.load", return_value=MagicMock())
        self._pkl_patch.start()

        self._svc_patch = patch(
            "predict._get_fire_svc",
            return_value=_make_mock_fire_svc(),
        )
        self._svc_patch.start()

        self._ext_patches = _all_patches()
        for p in self._ext_patches:
            p.start()

        self._ai_patch = patch(
            "predict.call_ai_model",
            return_value=(
                "Due to high fire risk in your area, evacuate immediately heading east. "
                "Take your pets and medications. Avoid smoke-filled roads near the hills."
            ),
        )
        self._ai_patch.start()

        self._router_patch = patch(
            "predict.EvacuationRouter",
            return_value=MagicMock(run=MagicMock(return_value={
                "reachable":   True,
                "route_cost":  14.5,
                "safe_zone":   {"lat": 34.15, "lon": -117.94, "label": "Emergency Shelter"},
                "path_latlon": [
                    {"lat": 34.1336, "lon": -117.9070},
                    {"lat": 34.1380, "lon": -117.9120},
                    {"lat": 34.1420, "lon": -117.9200},
                    {"lat": 34.1500, "lon": -117.9400},
                ],
                "path_col_row": [
                    {"col": 12, "row": 12},
                    {"col": 12, "row": 11},
                    {"col": 11, "row": 10},
                    {"col": 10, "row":  9},
                ],
                "grid_stats": {"free": 480, "fire": 60, "smoke": 70, "high_hazard": 15},
            })),
        )
        self._router_patch.start()

        from flask import Flask
        from routes import predict as predict_module
        app = Flask(__name__)
        app.register_blueprint(predict_module.predict_bp, url_prefix="/api")
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        self._pkl_patch.stop()
        self._svc_patch.stop()
        self._ai_patch.stop()
        self._router_patch.stop()
        for p in self._ext_patches:
            p.stop()

    def _post(self, payload):
        return self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_basic_request_returns_200(self):
        resp = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
        self.assertEqual(resp.status_code, 200)
        print("✓ POST /api/predict → 200 OK")

    def test_response_contains_all_top_level_keys(self):
        resp = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
        body = resp.get_json()
        for key in ("selected_fire", "ml", "evacuation", "guidance"):
            self.assertIn(key, body, f"Missing top-level key: {key}")
        print("✓ Response contains: selected_fire, ml, evacuation, guidance")

    def test_ml_block_structure(self):
        resp = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
        ml   = resp.get_json()["ml"]
        for key in ("burn_probability", "hazard_pred_class", "p_low", "p_med", "p_high", "heat_weight"):
            self.assertIn(key, ml)
        self.assertGreaterEqual(ml["burn_probability"], 0.0)
        self.assertLessEqual(ml["burn_probability"],    1.0)
        self.assertIn(ml["hazard_pred_class"], (0, 1, 2))
        print(f"✓ ML block valid — burn_prob={ml['burn_probability']:.2f}, "
              f"hazard_class={ml['hazard_pred_class']}")

    def test_evacuation_block_structure(self):
        resp = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
        evac = resp.get_json()["evacuation"]
        self.assertIn("reachable",   evac)
        self.assertIn("path_latlon", evac)
        self.assertIn("safe_zone",   evac)
        self.assertIn("grid_stats",  evac)
        self.assertTrue(evac["reachable"])
        self.assertGreater(len(evac["path_latlon"]), 0)
        self.assertIn("label", evac["safe_zone"])
        _VIZ_RESULTS["endpoint"] = (evac, None, None, False)
        print(f"✓ Evacuation block valid — {len(evac['path_latlon'])} waypoints, "
              f"safe zone='{evac['safe_zone']['label']}'")

    def test_guidance_is_non_empty_string(self):
        resp     = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
        guidance = resp.get_json()["guidance"]
        self.assertIsInstance(guidance, str)
        self.assertGreater(len(guidance), 20)
        print(f"✓ Guidance returned: \"{guidance[:80]}...\"")

    def test_full_request_with_all_checkboxes(self):
        payload = {
            "latitude":        MOCK_LAT,
            "longitude":       MOCK_LON,
            "has_disability":  True,
            "has_pets":        True,
            "has_kids":        True,
            "has_medications": True,
            "other_concerns":  "Elderly parent, limited mobility",
        }
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 200)
        print("✓ Full checkbox payload → 200 OK")

    def test_missing_lat_lon_returns_400(self):
        resp = self._post({"has_pets": True})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("latitude", resp.get_json()["error"])
        print("✓ Missing lat/lon → 400 with helpful error message")

    def test_invalid_lat_lon_type_returns_400(self):
        resp = self._post({"latitude": "not_a_number", "longitude": MOCK_LON})
        self.assertEqual(resp.status_code, 400)
        print("✓ Non-numeric lat/lon → 400")

    def test_empty_body_returns_400(self):
        resp = self.client.post("/api/predict", data="", content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        print("✓ Empty body → 400")

    def test_routing_failure_is_non_fatal(self):
        """If routing crashes, ML results should still come back."""
        self._router_patch.stop()
        broken_router = patch(
            "predict.EvacuationRouter",
            return_value=MagicMock(run=MagicMock(side_effect=RuntimeError("Grid build failed")))
        )
        broken_router.start()
        try:
            resp = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
            body = resp.get_json()
            self.assertEqual(resp.status_code, 200)
            self.assertIn("ml",       body)
            self.assertIn("guidance", body)
            self.assertFalse(body["evacuation"]["reachable"])
            self.assertIn("error", body["evacuation"])
            print("✓ Routing failure is non-fatal — ML + guidance still returned")
        finally:
            broken_router.stop()
            self._router_patch.start()


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MOCK PIPELINE TESTS  (no live APIs or fire needed)")
    print(f"Simulated location : ({MOCK_LAT}, {MOCK_LON})  — Azusa foothills, CA")
    print(f"Simulated fire     : {MOCK_PERIMETER_METRICS['fire_name']} "
          f"({MOCK_PERIMETER_METRICS['year']}), "
          f"{MOCK_PERIMETER_METRICS['dist_to_front_km']} km away")
    print(f"Simulated wind     : {MOCK_WIND['wind_speed_ms']} m/s → {MOCK_WIND['wind_dir_deg_to']}°")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    print("\n── EvacuationRouter / pathfinder (isolation) ────────────")
    suite.addTests(loader.loadTestsFromTestCase(TestEvacuationRouter))

    print("\n── /api/predict (full pipeline) ─────────────────────────")
    suite.addTests(loader.loadTestsFromTestCase(TestPredictEndpoint))

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"Results: {passed}/{result.testsRun} passed")
    if result.failures or result.errors:
        print("FAILURES:")
        for f in result.failures + result.errors:
            print(f"  ✗ {f[0]}")
    print("=" * 60)

    # ── Visualisation ──────────────────────────────────────────────────────
    # Renders one figure per captured scenario using the REAL GridCell grid.
    VIZ_SCENARIOS = [
        # (dict_key,              title_prefix,                    description)
        ("normal",              "Normal conditions — ",          "mixed terrain, reachable route"),
        ("disability_normal",   "Smoke grid, able-bodied — ",    "all-smoke grid, normal user"),
        ("disability_disabled", "Smoke grid, disabled — ",       "all-smoke grid, disabled user"),
        ("blocked",             "High-fire scenario — ",         "near-fully blocked grid"),
        ("endpoint",            "/api/predict response — ",      "mock endpoint evacuation block"),
    ]

    if not _VIZ_RESULTS:
        print("\n(No visualisation data captured — tests may have failed.)")
    else:
        print("\n── Visualisations ───────────────────────────────────────")
        for key, prefix, desc in VIZ_SCENARIOS:
            if key in _VIZ_RESULTS:
                router_result, real_grid, goal_cells, disability = _VIZ_RESULTS[key]
                print(f"  Showing: {desc}")
                visualize_router_result(
                    router_result,
                    real_grid=real_grid,
                    all_goal_cells=goal_cells,
                    has_disability=disability,
                    title_prefix=prefix,
                )