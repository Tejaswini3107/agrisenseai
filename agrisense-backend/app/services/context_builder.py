def build_weather_context(weather_response: dict) -> str:
    """
    Build a plain English description of the current agricultural situation.

    Args:
        weather_response (dict): Full JSON response from /api/weather/current endpoint
                                 Keys: temperature, humidity, rainfall_mm, wind_speed, condition,
                                       location_name, soil_moisture, recommended_crop, crop_confidence,
                                       disease_risk, disease_confidence, plant_stress, stress_confidence,
                                       irrigation_need_litres, expected_yield_tons, climate_risk, climate_confidence

    Returns:
        str: Plain English description of the agricultural situation
    """
    context_parts = []

    # Location
    location_name = weather_response.get("location_name")
    if location_name:
        context_parts.append(f"Location: {location_name}.")

    # Weather
    temperature = weather_response.get("temperature")
    humidity = weather_response.get("humidity")
    rainfall_mm = weather_response.get("rainfall_mm")
    wind_speed = weather_response.get("wind_speed")
    condition = weather_response.get("condition")

    if all(x is not None for x in [temperature, humidity, rainfall_mm, wind_speed, condition]):
        context_parts.append(
            f"Current weather: temperature {round(temperature, 1)}°C, humidity {round(humidity, 1)}%, "
            f"rainfall today {round(rainfall_mm, 1)}mm, wind speed {round(wind_speed, 1)} km/h, "
            f"conditions: {condition}."
        )

    # Soil moisture
    soil_moisture = weather_response.get("soil_moisture")
    if soil_moisture is not None:
        soil_moisture = round(soil_moisture, 1)
        if soil_moisture < 25:
            level = "dry"
        elif soil_moisture <= 55:
            level = "moderate"
        else:
            level = "wet"
        context_parts.append(f"Soil moisture: {soil_moisture}% ({level}).")

    # Climate risk
    climate_risk = weather_response.get("climate_risk")
    if climate_risk:
        context_parts.append(f"Climate risk level: {climate_risk}.")

    # Recommended crop
    recommended_crop = weather_response.get("recommended_crop")
    if recommended_crop:
        context_parts.append(f"Recommended crop for these conditions: {recommended_crop}.")

    # Disease and stress
    disease_risk = weather_response.get("disease_risk")
    plant_stress = weather_response.get("plant_stress")
    if disease_risk or plant_stress:
        disease_str = f"Disease risk: {disease_risk}." if disease_risk else ""
        stress_str = f"Plant stress level: {plant_stress}." if plant_stress else ""
        context_parts.append(f"{disease_str} {stress_str}".strip() + ".")

    # Irrigation
    irrigation_need_litres = weather_response.get("irrigation_need_litres")
    if irrigation_need_litres is not None:
        context_parts.append(
            f"Irrigation recommendation: {round(irrigation_need_litres, 1)} litres per hectare needed today."
        )

    # Expected yield
    expected_yield_tons = weather_response.get("expected_yield_tons")
    if expected_yield_tons is not None:
        context_parts.append(
            f"Expected yield under current conditions: {round(expected_yield_tons, 1)} tons per hectare."
        )

    return " ".join(context_parts)


def build_irrigation_context(irrigate: bool, weather_response: dict, reason: str) -> str:
    """
    Build an irrigation decision context by combining weather context with irrigation recommendation.

    Args:
        irrigate (bool): Whether to irrigate today
        weather_response (dict): Full JSON response from /api/weather/current endpoint
        reason (str): Reason for the irrigation decision

    Returns:
        str: Combined weather context with irrigation decision
    """
    base_context = build_weather_context(weather_response)

    irrigation_decision = "Irrigate today" if irrigate else "Wait before irrigating"
    decision_sentence = f"Irrigation decision: {irrigation_decision}. Reason given: {reason}."

    if base_context:
        return f"{base_context} {decision_sentence}"
    else:
        return decision_sentence
