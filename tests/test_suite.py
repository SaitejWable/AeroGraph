"""
tests/test_suite.py
Verification & Acceptance Test Suite for AeroGraph.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.diagnostic import evaluate_diagnostic_matrix
from graph.workflow import (
    apply_safety_clamps,
    load_fallback_cache,
    run_aerograph_orchestration,
)
from ml.online_model import train_and_forecast_online
from tools.data_engine import (
    PRESET_CITIES,
    calculate_cpcb_composite,
    fetch_ground_truth_validation,
    fetch_live_data,
    fetch_monthly_history,
)


def run_tests():
    print("=== 1. TEST PRESET 10 CITIES CONFIGURATION ===")
    assert len(PRESET_CITIES) >= 10, f"Expected 10 cities, found {len(PRESET_CITIES)}"
    expected_cities = [
        "Delhi", "Mumbai", "Pune", "Bengaluru", "Kolkata",
        "Chennai", "Hyderabad", "Ahmedabad", "Lucknow", "Jaipur"
    ]
    for c in expected_cities:
        assert c in PRESET_CITIES, f"{c} missing from PRESET_CITIES"
    print(f"Verified all 10 cities in PRESET_CITIES: {list(PRESET_CITIES.keys())}")

    print("\n=== 2. TEST LIVE INGESTION FOR TARGET METROPOLITANS ===")
    for city in ["Delhi", "Mumbai", "Bengaluru"]:
        df_c = fetch_live_data(city)
        aqi_val = df_c["aqi"].iloc[-1]
        dom_val = df_c["dominant_pollutant"].iloc[-1]
        print(
            f"{city}: Rows={len(df_c)}, Current AQI={aqi_val:.1f},"
            f" Dominant={dom_val}"
        )
        assert len(df_c) >= 24, f"Expected at least 24 rows for {city}"

    print("\n=== 3. TEST CROSS-SOURCE GROUND-TRUTH VALIDATION ===")
    coords = PRESET_CITIES["Delhi"]
    val_res = fetch_ground_truth_validation(coords["lat"], coords["lon"], "Delhi", 150.0)
    print(f"Validation result: Ground AQI={val_res['ground_aqi']}, Delta={val_res['delta_pct']}%, Status={val_res['status']}")
    assert "ground_aqi" in val_res and "delta_pct" in val_res, "Validation dict must include ground_aqi and delta_pct"

    print("\n=== 4. TEST 30-DAY MONTHLY HISTORY RETRIEVAL ===")
    df_mon = fetch_monthly_history("Delhi")
    print(f"Monthly history: Rows={len(df_mon)}, Latest AQI={df_mon['aqi'].iloc[-1]:.1f}")
    assert len(df_mon) >= 24, "Monthly history should have at least 24 observations"

    print("\n=== 5. TEST RIVER MODEL DATASET PRE-TRAINING & REAL-TIME PROGRESSIVE FIT ===")
    df_live = fetch_live_data("Delhi")
    ml_res = train_and_forecast_online(df_live, city="Delhi")
    print(
        f"Pretrained Samples={ml_res.pretrain_samples:,} from {ml_res.pretrain_source}\n"
        f"River MAE={ml_res.model_mae:.2f}, Persistence MAE={ml_res.persistence_mae:.2f}, "
        f"Improvement={ml_res.mae_improvement_pct:.1f}%\n"
        f"3h Forecast Trajectory: {[p.predicted_aqi for p in ml_res.forecast_trajectory]}"
    )
    assert ml_res.pretrain_samples > 0, "River model must pre-train from dataset/"
    assert len(ml_res.forecast_trajectory) == 3, "Must generate 3-step forecast"

    print("\n=== 6. TEST POST-LLM SAFETY CLAMPS ===")
    dummy_plan = {
        "alert_level": "Normal Airflow",
        "root_cause_summary": "test",
        "commuter": {
            "mask_recommendation": "None Required",
            "departure_window_advice": "",
            "transit_route_advice": "",
        },
        "school": {
            "recess_protocol": "Normal Schedule",
            "ventilation_window_timing": "",
        },
        "facilities": {
            "recirculation_mode": "100% Outdoor Air",
            "pre_scrub_minutes": 30,
        },
        "municipal": {
            "source_action": "Deploy Mechanical Sweepers Upwind",
            "target_corridor": "",
        },
    }
    clamped = apply_safety_clamps(
        dummy_plan, current_aqi=380.0, forecast_t1_aqi=390.0
    )
    print("Clamped mask (AQI 380):", clamped["commuter"]["mask_recommendation"])
    print("Clamped school recess:", clamped["school"]["recess_protocol"])
    print("Clamped HVAC mode:", clamped["facilities"]["recirculation_mode"])
    assert (
        clamped["commuter"]["mask_recommendation"] == "N95 / FFP2 Respirator"
    )
    assert clamped["school"]["recess_protocol"] == "Suspend Outdoor Activities"
    assert clamped["facilities"]["recirculation_mode"] == "Isolated Indoor Loop"

    print("\n=== 7. TEST INVALID KEY CACHE FALLBACK ===")
    cache_state = run_aerograph_orchestration(
        city="Test City",
        current_aqi=200.0,
        current_pm25=80.0,
        current_pm10=140.0,
        current_no2=45.0,
        current_so2=15.0,
        wind_speed=4.0,
        wind_direction=315.0,
        humidity=60.0,
        dominant_pollutant="PM2.5",
        forecast_trajectory=[],
        is_breach=False,
        breach_deltas=(0.0, 0.0),
        model_mae=10.0,
        persistence_mae=12.0,
        api_key_override="INVALID_KEY_12345",
    )
    print("Used cache with invalid key:", cache_state["used_cache"])
    print("Fallback reason:", cache_state.get("fallback_reason"))
    assert (
        cache_state["used_cache"] is True
    ), "Must fallback to cache on invalid key"

    print("\n[SUCCESS] ALL ACCEPTANCE CRITERIA PASSED!")


if __name__ == "__main__":
    run_tests()
