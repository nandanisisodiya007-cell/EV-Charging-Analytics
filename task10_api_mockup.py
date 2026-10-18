"""
Task 10 API mockup.

Demonstrates the input and output structure for a future
EV charging prediction API. This is not a deployed API.
"""

def predict_charging_session(
    charger_type,
    station_region,
    user_type,
    price_per_kwh,
    year
):
    # Example response structure only.
    # Replace placeholder values with predictions from
    # the trained model before using this as a real API.
    return {
        "status": "mockup",
        "input": {
            "charger_type": charger_type,
            "station_region": station_region,
            "user_type": user_type,
            "price_per_kwh": price_per_kwh,
            "year": year
        },
        "predictions": {
            "session_duration_minutes": None,
            "energy_kwh": None
        },
        "message": (
            "Connect the trained models to return real predictions."
        )
    }


if __name__ == "__main__":
    response = predict_charging_session(
        charger_type="DC",
        station_region="Example Region",
        user_type="Example User",
        price_per_kwh=10.0,
        year=2026
    )

    print(response)

