"""
graph/diagnostic.py
Scientific Diagnostic Matrix for AeroGraph.
Evaluates atmospheric pollutant ratios (PM2.5/PM10), precursor concentrations (NO2, SO2),
and meteorological dispersion dynamics (wind speed, wind bearing, relative humidity)
to classify the situation into a defensible, scientifically-framed "Probable Emission Profile".
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class DiagnosticProfile:
    probable_emission_profile: str
    confidence_rationale: str
    dominant_pollutant: str
    pm25_pm10_ratio: float
    wind_vector_summary: str
    upwind_bearing_label: str
    upwind_degrees: float
    meteorological_context: str


def degrees_to_cardinal(deg: float) -> str:
    """Converts wind degrees (0-360) into 16-point cardinal compass abbreviations."""
    val = int((deg / 22.5) + 0.5) % 16
    compass_points = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    return compass_points[val]


def evaluate_diagnostic_matrix(
    pm25: float,
    pm10: float,
    no2: float,
    so2: float,
    wind_speed: float,
    wind_direction: float,
    humidity: float,
    dominant_pollutant: str = "PM2.5",
) -> DiagnosticProfile:
    """Evaluates atmospheric stoichiometry and micro-meteorological variables against

    the scientific diagnostic matrix, returning an evidence-backed hypothesis.
    """
    ratio = round(pm25 / max(1.0, pm10), 2)
    cardinal = degrees_to_cardinal(wind_direction)
    wind_summary = f"{wind_speed:.1f} km/h from {cardinal} ({int(wind_direction)}°)"

    # Upwind direction is where the air mass is originating from
    upwind_label = f"{cardinal} ({int(wind_direction)}°)"
    upwind_deg = float(wind_direction)

    # Condition 4: Industrial Point-Source Plume
    if so2 > 50.0:
        profile = "Point-Source Industrial Emissions from upwind cluster"
        rationale = (
            f"Elevated SO2 concentration ({so2:.1f} µg/m³ > 50 µg/m³ threshold) "
            f"indicates distinct fossil-fuel or metallurgical combustion upwind from {upwind_label}."
        )

    # Condition 1: Vehicular & Urban Combustion Plume
    elif ratio > 0.65 and no2 > 40.0 and wind_speed < 6.0:
        profile = (
            "Vehicular & Urban Combustion Plume trapped under low boundary"
            " layer"
        )
        rationale = (
            f"Fine aerosol fraction (PM2.5/PM10 = {ratio:.2f} > 0.65) paired with elevated "
            f"NO2 ({no2:.1f} µg/m³ > 40 µg/m³) and stagnant surface wind ({wind_speed:.1f} km/h < 6 km/h) "
            "demonstrates direct tailpipe combustion entrapment under a compressed thermal inversion layer."
        )

    # Condition 2: Biomass / Regional Agricultural Burning
    elif ratio > 0.65 and no2 < 25.0 and wind_speed > 8.0:
        profile = "Biomass / Regional Agricultural Burning drifting downwind"
        rationale = (
            f"High fine-mode fraction (PM2.5/PM10 = {ratio:.2f} > 0.65) with low primary NO2 "
            f"({no2:.1f} µg/m³ < 25 µg/m³) accompanied by directional advection ({wind_speed:.1f} km/h > 8 km/h) "
            f"is consistent with long-range biomass/crop residue smoke transport originating along the {upwind_label} corridor."
        )

    # Condition 3: Mechanical Surface Dust / Construction suspension
    elif ratio < 0.45 and no2 < 35.0 and wind_speed > 12.0:
        profile = "Mechanical Surface Dust / Construction suspension"
        rationale = (
            f"Coarse fraction dominance (PM2.5/PM10 = {ratio:.2f} < 0.45) under high surface wind speeds "
            f"({wind_speed:.1f} km/h > 12 km/h) points to non-exhaust mechanical resuspension, roadway dust, "
            "and unpaved construction aggregate uplift."
        )

    # Condition 1b: Urban combustion with moderate wind / inversion
    elif ratio > 0.60 and no2 >= 35.0:
        profile = (
            "Vehicular & Urban Combustion Plume trapped under low boundary"
            " layer"
        )
        rationale = (
            f"PM2.5/PM10 ratio of {ratio:.2f} with NO2 at {no2:.1f} µg/m³ reflects dominant urban "
            f"secondary aerosol formation and vehicular combustion plume stagnation."
        )

    # Condition 2b: Biomass / Regional Burning with moderate wind
    elif ratio > 0.65:
        profile = "Biomass / Regional Agricultural Burning drifting downwind"
        rationale = (
            f"Strong fine aerosol dominance (PM2.5/PM10 = {ratio:.2f} > 0.65) under regional wind vectors "
            f"from {upwind_label} indicates agricultural burning or open biomass combustion."
        )

    # Fallback / Mixed urban background
    else:
        profile = (
            "Mixed Urban Secondary Aerosol / Stagnant Background Stagnation"
        )
        rationale = (
            f"Intermediate PM2.5/PM10 ratio ({ratio:.2f}) and moderate precursor concentrations "
            f"(NO2: {no2:.1f} µg/m³, SO2: {so2:.1f} µg/m³) reflect a heterogeneous urban background equilibrium."
        )

    meteo_ctx = (
        f"Wind: {wind_summary} | Humidity: {humidity:.1f}% | Dominant Pollutant:"
        f" {dominant_pollutant}"
    )

    return DiagnosticProfile(
        probable_emission_profile=profile,
        confidence_rationale=rationale,
        dominant_pollutant=dominant_pollutant,
        pm25_pm10_ratio=ratio,
        wind_vector_summary=wind_summary,
        upwind_bearing_label=upwind_label,
        upwind_degrees=upwind_deg,
        meteorological_context=meteo_ctx,
    )
