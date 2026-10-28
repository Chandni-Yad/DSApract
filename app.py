from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="FasalShield AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

STAGES = ["Emergence", "Vegetative", "Flowering", "Fruiting", "Maturity"]
STAGE_VULNERABILITY = {
    "Emergence": 0.48,
    "Vegetative": 0.58,
    "Flowering": 0.84,
    "Fruiting": 0.90,
    "Maturity": 0.55,
}
FEATURES = [
    "temperature_c",
    "humidity_pct",
    "rain_7d_mm",
    "stage_index",
    "pest_pressure",
    "leaf_wetness_hours",
    "forecast_rain_mm",
]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


@st.cache_resource
def train_demo_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train a deterministic prototype model on simulated, rule-generated examples."""
    rng = np.random.default_rng(42)
    sample_count = 6000
    data = pd.DataFrame(
        {
            "temperature_c": rng.uniform(10, 42, sample_count),
            "humidity_pct": rng.uniform(25, 100, sample_count),
            "rain_7d_mm": rng.uniform(0, 180, sample_count),
            "stage_index": rng.integers(0, len(STAGES), sample_count),
            "pest_pressure": rng.integers(0, 4, sample_count),
            "leaf_wetness_hours": rng.uniform(0, 18, sample_count),
            "forecast_rain_mm": rng.uniform(0, 100, sample_count),
        }
    )
    stage_values = np.array(list(STAGE_VULNERABILITY.values()))
    temp_suitability = np.clip(
        1 - np.abs(data["temperature_c"].to_numpy() - 27) / 18, 0, 1
    )
    humidity_signal = np.clip((data["humidity_pct"].to_numpy() - 55) / 45, 0, 1)
    rain_signal = np.clip(data["rain_7d_mm"].to_numpy() / 100, 0, 1)
    pest_signal = data["pest_pressure"].to_numpy() / 3
    wetness_signal = data["leaf_wetness_hours"].to_numpy() / 18
    forecast_signal = np.clip(data["forecast_rain_mm"].to_numpy() / 70, 0, 1)
    stage_signal = stage_values[data["stage_index"].to_numpy()]

    log_odds = (
        -3.1
        + 1.15 * temp_suitability
        + 1.45 * humidity_signal
        + 0.95 * rain_signal
        + 1.1 * stage_signal
        + 1.55 * pest_signal
        + 1.0 * wetness_signal
        + 0.75 * forecast_signal
    )
    labels = rng.binomial(1, _sigmoid(log_odds))
    features = data[FEATURES].to_numpy()
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales == 0] = 1
    scaled_features = (features - means) / scales
    design = np.column_stack((np.ones(sample_count), scaled_features))
    weights = np.zeros(design.shape[1])
    for _ in range(500):
        error = _sigmoid(design @ weights) - labels
        gradient = design.T @ error / sample_count
        gradient[1:] += 0.01 * weights[1:]
        weights -= 0.1 * gradient
    return weights, means, scales


def fetch_weather(place: str) -> dict[str, object]:
    geocode_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place, "count": 1, "language": "en", "format": "json"},
        timeout=12,
    )
    geocode_response.raise_for_status()
    locations = geocode_response.json().get("results", [])
    if not locations:
        raise ValueError(f"No location found for “{place}”. Try a nearby city or district.")

    location = locations[0]
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,relative_humidity_2m",
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "forecast_days": 7,
            "timezone": "auto",
        },
        timeout=12,
    )
    weather_response.raise_for_status()
    weather = weather_response.json()
    current = weather["current"]
    daily = weather["daily"]
    return {
        "temperature_c": float(current["temperature_2m"]),
        "humidity_pct": float(current["relative_humidity_2m"]),
        "forecast_rain_mm": float(sum(daily["precipitation_sum"])),
        "location_name": str(location["name"]),
        "forecast_daily": [
            {
                "Date": day,
                "Rain (mm)": float(rain),
                "High (°C)": float(high),
                "Low (°C)": float(low),
            }
            for day, rain, high, low in zip(
                daily["time"],
                daily["precipitation_sum"],
                daily["temperature_2m_max"],
                daily["temperature_2m_min"],
                strict=True,
            )
        ],
    }


def refresh_weather_inputs() -> None:
    try:
        live = fetch_weather(st.session_state.location.strip())
    except (requests.RequestException, ValueError, KeyError) as error:
        st.session_state.weather_error = f"Could not update weather: {error}"
        st.session_state.weather_message = ""
        return

    st.session_state.weather_values = live
    st.session_state.temperature_input = float(live["temperature_c"])
    st.session_state.humidity_input = round(float(live["humidity_pct"]))
    st.session_state.forecast_rain_input = float(live["forecast_rain_mm"])
    st.session_state.weather_error = ""
    st.session_state.weather_message = f"Weather updated for {live['location_name']}."


def predict_risk(
    *,
    temperature_c: float,
    humidity_pct: float,
    rain_7d_mm: float,
    stage: str,
    pest_pressure: int,
    leaf_wetness_hours: float,
    forecast_rain_mm: float,
) -> tuple[float, dict[str, float]]:
    weights, means, scales = train_demo_model()
    stage_index = STAGES.index(stage)
    row = pd.DataFrame(
        [
            {
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "rain_7d_mm": rain_7d_mm,
                "stage_index": stage_index,
                "pest_pressure": pest_pressure,
                "leaf_wetness_hours": leaf_wetness_hours,
                "forecast_rain_mm": forecast_rain_mm,
            }
        ],
        columns=FEATURES,
    )
    scaled_row = (row.to_numpy()[0] - means) / scales
    probability = float(_sigmoid(np.array([1.0, *scaled_row]) @ weights))
    temperature_signal = float(
        np.clip(1 - abs(temperature_c - 27) / 18, 0, 1) * 100
    )
    humidity_signal = float(np.clip((humidity_pct - 55) / 45, 0, 1) * 100)
    wet_weather_signal = float(
        np.clip((rain_7d_mm / 100 + leaf_wetness_hours / 18) / 2, 0, 1) * 100
    )
    stage_signal = STAGE_VULNERABILITY[stage] * 100
    pest_signal = float(pest_pressure / 3 * 100)
    forecast_signal = float(np.clip(forecast_rain_mm / 70, 0, 1) * 100)
    signals = {
        "Temperature-conducive conditions": temperature_signal,
        "Humidity pressure": humidity_signal,
        "Recent wetness": wet_weather_signal,
        "Crop-stage sensitivity": stage_signal,
        "Pest observations": pest_signal,
        "7-day rain outlook": forecast_signal,
    }
    return probability, signals


def risk_band(probability: float) -> tuple[str, str, str]:
    if probability >= 0.70:
        return "High", "#ef6b62", "Take preventive action today"
    if probability >= 0.40:
        return "Moderate", "#e7a847", "Monitor closely over the next 48 hours"
    return "Low", "#46a98b", "Continue routine crop monitoring"


def suggested_actions(
    stage: str,
    humidity_pct: float,
    leaf_wetness_hours: float,
    pest_pressure: int,
    forecast_rain_mm: float,
) -> list[str]:
    actions = []
    if humidity_pct >= 75 or leaf_wetness_hours >= 8:
        actions.append(
            "Inspect lower leaves for early fungal symptoms; avoid overhead irrigation "
            "and improve airflow where practical."
        )
    if pest_pressure >= 2:
        actions.append(
            "Scout 5–10 plants in each part of the field and check leaf undersides "
            "before choosing a control measure."
        )
    if forecast_rain_mm >= 45:
        actions.append(
            "Check field drainage and postpone spray applications until conditions "
            "are suitable and the product label allows."
        )
    if stage in ("Flowering", "Fruiting"):
        actions.append(
            f"{stage} is a sensitive stage: scout more frequently and protect "
            "pollinators when planning any treatment."
        )
    if not actions:
        actions.append(
            "Keep a regular scouting schedule and record any new symptoms or pest "
            "sightings for the next assessment."
        )
    return actions


def estimate_pest_risk(
    pest_pressure: int, temperature_c: float, humidity_pct: float, stage: str
) -> int:
    temperature_signal = np.clip(1 - abs(temperature_c - 28) / 18, 0, 1)
    humidity_signal = np.clip((humidity_pct - 45) / 55, 0, 1)
    pressure_signal = pest_pressure / 3
    stage_signal = STAGE_VULNERABILITY[stage]
    score = 100 * (
        0.50 * pressure_signal
        + 0.20 * temperature_signal
        + 0.15 * humidity_signal
        + 0.15 * stage_signal
    )
    return round(float(score))


def irrigation_guidance(
    crop: str,
    stage: str,
    soil_status: str,
    temperature_c: float,
    forecast_rain_mm: float,
) -> tuple[str, str]:
    if soil_status == "Wet":
        return (
            "Hold irrigation",
            "Soil is reported wet. Check drainage and root-zone moisture again before watering.",
        )
    if forecast_rain_mm >= 35:
        return (
            "Wait and reassess",
            f"Rain is forecast this week. Recheck {crop} root-zone moisture after rainfall "
            "before irrigating.",
        )
    if soil_status == "Dry":
        if stage in ("Flowering", "Fruiting") or temperature_c >= 32:
            return (
                "Prioritize a root-zone check",
                f"Dry soil during {stage.lower()} or hot weather can stress {crop}. "
                "If the root zone is dry, plan irrigation using your local soil and crop guidance.",
            )
        return (
            "Check before watering",
            f"The soil is reported dry. Verify moisture near the roots of your {crop} "
            "before scheduling irrigation.",
        )
    return (
        "No immediate irrigation signal",
        "Soil is reported moist. Recheck the root zone and local rainfall before watering.",
    )


def answer_hindi_question(question: str, crop: str, level: str, stage: str) -> str:
    text = question.casefold()
    if any(word in text for word in ("पानी", "सिंचाई", "water", "irrigat")):
        return (
            f"{crop} की {stage.lower()} अवस्था में जड़ के पास मिट्टी की नमी जाँचें। "
            "मिट्टी पहले से गीली हो या बारिश आने वाली हो तो सिंचाई रोककर दोबारा जाँचें। "
            "पानी की मात्रा मिट्टी और स्थानीय कृषि सलाह के अनुसार तय करें।"
        )
    if any(word in text for word in ("कीट", "pest", "insect", "इल्ली", "माहू")):
        return (
            "पत्तियों के नीचे सहित खेत के अलग-अलग हिस्सों में पौधों की जाँच करें और "
            "कीटों की संख्या लिखें। केवल इस अनुमान के आधार पर दवा न छिड़कें; स्थानीय "
            "कृषि विशेषज्ञ से पहचान और नियंत्रण की पुष्टि करें।"
        )
    if any(word in text for word in ("बीमारी", "रोग", "दाग", "पीला", "फोटो", "disease", "leaf")):
        return (
            "पत्तियों के दाग, पीलेपन या मुरझाने की तस्वीर लें और देखें कि लक्षण कितने "
            "पौधों में हैं। यह डेमो फोटो से बीमारी की पुष्टि नहीं करता। सही पहचान के "
            "लिए स्थानीय कृषि अधिकारी या पौधा रोग विशेषज्ञ को नमूना दिखाएँ।"
        )
    if any(word in text for word in ("जोखिम", "risk", "मौसम", "weather", "बारिश")):
        return (
            f"अभी डैशबोर्ड का संयुक्त जोखिम स्तर {level} है। यह मौसम, फसल की अवस्था "
            "और खेत में देखे गए कीटों के संकेतों पर आधारित डेमो अनुमान है, बीमारी की "
            "पुष्टि नहीं। खेत में नियमित निरीक्षण करें।"
        )
    return (
        f"मैं {crop} के मौसम, कीट, पत्तियों के लक्षण और सिंचाई पर सामान्य मार्गदर्शन "
        "दे सकता हूँ। अपना सवाल इनमें से किसी विषय पर पूछें। यह ऑफलाइन डेमो सहायक है; "
        "महत्वपूर्ण निर्णय स्थानीय कृषि विशेषज्ञ से जाँचें।"
    )


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink: #1d3028; --muted: #78877e; --green: #245c43; --line: #e7ece7; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f7f3; color: var(--ink); }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { font-family: 'Manrope', sans-serif; }
    .block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
    .brand { display:flex; align-items:center; gap:12px; margin:4px 0 27px; }
    .brand-icon { display:flex; align-items:center; justify-content:center; width:42px;
      height:42px; border-radius:14px; background:#e9f2e9; font-size:23px; }
    .brand-name { font-family:'Manrope',sans-serif; font-size:19px; font-weight:800;
      letter-spacing:-.7px; color:#1e3329; line-height:1.15; }
    .brand-caption { color:#88938b; font-size:11px; margin-top:3px; }
    .eyebrow { color:#62816e; text-transform:uppercase; font-weight:700;
      letter-spacing:1.3px; font-size:10px; }
    .page-title { font-family:'Manrope',sans-serif; font-size:30px; font-weight:800;
      letter-spacing:-1.1px; margin:2px 0 3px; color:#1e3329; }
    .subtitle { color:#7c8980; font-size:13px; margin-bottom:22px; }
    .panel { background:#fff; border:1px solid #e8ede8; border-radius:17px;
      padding:20px 21px; box-shadow:0 3px 14px rgba(37,61,44,.025); }
    .panel-title { font-family:'Manrope',sans-serif; font-size:15px; font-weight:700;
      color:#273a30; margin:0 0 3px; }
    .panel-subtitle { color:#8a958d; font-size:11px; margin:0 0 16px; }
    .metric-label { color:#859188; font-size:11px; font-weight:600; }
    .metric-value { font-family:'Manrope',sans-serif; font-size:24px;
      letter-spacing:-.5px; font-weight:800; color:#20382c; margin-top:3px; }
    .metric-foot { color:#87928a; font-size:10px; margin-top:2px; }
    .risk-card { background:#fff; border:1px solid #e8ede8; border-radius:17px;
      padding:19px 21px; min-height:176px; }
    .risk-number { font-family:'Manrope',sans-serif; font-size:43px; font-weight:800;
      letter-spacing:-2px; line-height:1.1; }
    .risk-pill { display:inline-block; border-radius:99px; padding:5px 10px;
      font-weight:700; font-size:11px; margin:9px 0 6px; }
    .risk-caption { color:#7d8981; font-size:11px; }
    .alert-banner { border-radius:12px; padding:13px 15px; border:1px solid #e8ede8;
      background:#fff; font-size:12px; color:#415349; margin:12px 0 18px; }
    .action { border-top:1px solid #edf0ed; padding:11px 0; color:#536259; font-size:12px; }
    .action:first-of-type { border-top:0; }
    .source-note { color:#8d978f; font-size:10px; line-height:1.5; margin-top:16px; }
    div.stButton > button { border-radius:10px; min-height:42px; font-weight:700; }
    div.stButton > button[kind="primary"] { background:#245c43; border-color:#245c43; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">🌱</div><div>'
        '<div class="brand-name">FasalShield AI</div>'
        '<div class="brand-caption">Early crop risk warning</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">Your field</div>', unsafe_allow_html=True)
    crop = st.selectbox("Crop", ["Tomato", "Cotton", "Rice", "Wheat", "Chilli", "Other"])
    stage = st.selectbox("Current crop stage", STAGES, index=2)
    st.text_input("Location", key="location", value="Pune")
    st.button(
        "↻  Fetch live weather",
        use_container_width=True,
        on_click=refresh_weather_inputs,
    )
    if st.session_state.get("weather_error"):
        st.error(st.session_state.weather_error)
    elif st.session_state.get("weather_message"):
        st.success(st.session_state.weather_message)
    st.caption("Live lookup uses Open-Meteo. You can also enter local readings below.")

    live_values = st.session_state.get("weather_values", {})
    st.session_state.setdefault(
        "temperature_input", float(live_values.get("temperature_c", 28.0))
    )
    st.session_state.setdefault(
        "humidity_input", round(float(live_values.get("humidity_pct", 72)))
    )
    st.session_state.setdefault(
        "forecast_rain_input", float(live_values.get("forecast_rain_mm", 18.0))
    )
    st.markdown('<div class="eyebrow" style="margin-top:22px">Weather signals</div>', unsafe_allow_html=True)
    temperature_c = st.slider(
        "Temperature · °C",
        5.0,
        45.0,
        step=0.5,
        key="temperature_input",
    )
    humidity_pct = st.slider(
        "Relative humidity · %",
        10,
        100,
        step=1,
        key="humidity_input",
    )
    rain_7d_mm = st.number_input("Rain in last 7 days · mm", 0.0, 500.0, 24.0, 1.0)
    leaf_wetness_hours = st.slider("Leaf wetness · hrs/day", 0.0, 18.0, 6.0, 0.5)
    forecast_rain_mm = st.number_input(
        "Forecast rain · next 7 days · mm",
        0.0,
        500.0,
        step=1.0,
        key="forecast_rain_input",
    )

    st.markdown('<div class="eyebrow" style="margin-top:22px">Pest observations</div>', unsafe_allow_html=True)
    pest_pressure = st.select_slider(
        "What have you noticed?",
        options=[0, 1, 2, 3],
        value=1,
        format_func=lambda value: ["None", "A few", "Several", "Many"][value],
    )
    st.caption("Use field scouting, not a photo diagnosis, to report pest pressure.")
    soil_status = st.selectbox("Soil near roots", ["Dry", "Moist", "Wet"], index=1)

st.markdown('<div class="eyebrow">FIELD OVERVIEW · EARLY WARNING</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Good morning, farmer 👋</div>', unsafe_allow_html=True)
location = st.session_state.get("weather_values", {}).get(
    "location_name", st.session_state.location
)
location_label = escape(str(location))
st.markdown(
    f'<div class="subtitle">Here’s the risk outlook for your <b>{crop}</b> field in '
    f'<b>{location_label}</b>. Update the signals in the sidebar for a fresh assessment.</div>',
    unsafe_allow_html=True,
)

probability, signals = predict_risk(
    temperature_c=temperature_c,
    humidity_pct=humidity_pct,
    rain_7d_mm=rain_7d_mm,
    stage=stage,
    pest_pressure=pest_pressure,
    leaf_wetness_hours=leaf_wetness_hours,
    forecast_rain_mm=forecast_rain_mm,
)
level, accent, guidance = risk_band(probability)
score = round(probability * 100)
pest_score = estimate_pest_risk(pest_pressure, temperature_c, humidity_pct, stage)
uncertainty_points = 22
risk_low = max(0, score - uncertainty_points)
risk_high = min(100, score + uncertainty_points)
irrigation_title, irrigation_detail = irrigation_guidance(
    crop, stage, soil_status, temperature_c, forecast_rain_mm
)

top = st.columns([1.1, 1, 1, 1, 1], gap="medium")
with top[0]:
    st.markdown(
        f'<div class="risk-card"><div class="metric-label">COMBINED CROP RISK</div>'
        f'<div class="risk-number" style="color:{accent}">{score}<span style="font-size:20px">%</span></div>'
        f'<span class="risk-pill" style="background:{accent}18;color:{accent}">{level} risk</span>'
        f'<div class="risk-caption">{guidance}</div></div>',
        unsafe_allow_html=True,
    )
for column, label, value, foot in [
    (top[1], "Current temperature", f"{temperature_c:g}°C", "Field / nearest weather station"),
    (top[2], "Relative humidity", f"{humidity_pct}%", "Higher humidity can favor disease"),
    (top[3], "Crop stage", stage, f"{crop} · stage sensitivity included"),
    (top[4], "Pest risk indicator", f"{pest_score}%", "Scouting + weather heuristic"),
]:
    with column:
        st.markdown(
            f'<div class="risk-card"><div class="metric-label">{label.upper()}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-foot">{foot}</div></div>',
            unsafe_allow_html=True,
        )

if level == "High":
    st.error("High crop-risk alert: scout the field today and confirm symptoms with a local expert.")
elif level == "Moderate":
    st.warning("Moderate crop-risk alert: increase field scouting over the next 48 hours.")
st.caption(
    f"Indicative risk range: **{risk_low}–{risk_high}%** "
    "(demo uncertainty band: ±22 percentage points, not a statistically validated confidence interval). "
    "The prototype model is trained on simulated data."
)

st.markdown(
    f'<div class="alert-banner"><b style="color:{accent}">◉ &nbsp;{level} risk outlook</b>'
    f' &nbsp;·&nbsp; {guidance}. This estimate combines weather, recent wetness, crop stage, '
    f'pest observations, and the 7-day rain outlook.</div>',
    unsafe_allow_html=True,
)

st.subheader("Smart irrigation guidance")
st.markdown(f"**{irrigation_title}** — {irrigation_detail}")
st.caption(
    "Guidance uses your soil-status selection, crop stage, temperature, and forecast rain. "
    "It does not calculate water volume; check root-zone moisture and local crop guidance."
)

st.subheader("Field observations")
with st.form("observation_form", clear_on_submit=True):
    observation_text = st.text_input(
        "Record a symptom, pest sighting, or field note",
        placeholder="e.g. Small spots on lower leaves in the east plot",
    )
    add_observation = st.form_submit_button("Add observation")
if add_observation:
    if observation_text.strip():
        observations = st.session_state.setdefault("field_observations", [])
        observations.insert(
            0,
            {
                "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "note": observation_text.strip(),
            },
        )
    else:
        st.warning("Enter an observation before adding it.")
for observation in st.session_state.get("field_observations", []):
    st.markdown(f"- **{observation['time']}** — {observation['note']}")

with st.expander("7-day weather forecast"):
    live_forecast = st.session_state.get("weather_values", {}).get("forecast_daily")
    if live_forecast:
        forecast_frame = pd.DataFrame(live_forecast)
        st.dataframe(forecast_frame, hide_index=True, width="stretch")
        forecast_chart = go.Figure()
        forecast_chart.add_bar(
            x=forecast_frame["Date"],
            y=forecast_frame["Rain (mm)"],
            name="Rain (mm)",
            marker_color="#83b79a",
        )
        forecast_chart.add_scatter(
            x=forecast_frame["Date"],
            y=forecast_frame["High (°C)"],
            name="High (°C)",
            yaxis="y2",
            line=dict(color="#e7a847", width=2),
        )
        forecast_chart.update_layout(
            height=270,
            margin=dict(l=10, r=10, t=25, b=10),
            yaxis=dict(title="Rain (mm)"),
            yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(forecast_chart, width="stretch")
    else:
        st.info("Fetch live weather in the sidebar to load the daily 7-day forecast.")

left, right = st.columns([1.05, 0.95], gap="medium")
with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">What is driving the risk?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtitle">Relative strength of each signal · not a disease diagnosis</div>',
            unsafe_allow_html=True,
        )
        signal_frame = pd.DataFrame(
            {"Signal": list(signals.keys()), "Relative signal": list(signals.values())}
        ).sort_values("Relative signal", ascending=True)
        chart = go.Figure(
            go.Bar(
                x=signal_frame["Relative signal"],
                y=signal_frame["Signal"],
                orientation="h",
                marker_color=[
                    "#a7c7ad" if value < 60 else "#e7a847" if value < 80 else "#ef8177"
                    for value in signal_frame["Relative signal"]
                ],
                text=[f"{value:.0f}" for value in signal_frame["Relative signal"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:.0f}/100<extra></extra>",
            )
        )
        chart.update_layout(
            height=300,
            margin=dict(l=0, r=30, t=5, b=5),
            xaxis=dict(range=[0, 110], showgrid=True, gridcolor="#edf1ed", title=None, ticksuffix=""),
            yaxis=dict(title=None, tickfont=dict(size=11, color="#536259")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color="#45544b"),
            showlegend=False,
        )
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Recommended next steps</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtitle">Practical checks based on the signals you entered</div>',
            unsafe_allow_html=True,
        )
        for action in suggested_actions(
            stage, humidity_pct, leaf_wetness_hours, pest_pressure, forecast_rain_mm
        ):
            st.markdown(f'<div class="action">↗ &nbsp;{action}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="source-note">Always confirm with field scouting or a local agricultural '
            'extension officer before applying crop-protection products. Follow product labels '
            'and local regulations.</div>',
            unsafe_allow_html=True,
        )

st.markdown("")
st.markdown(
    '<div class="source-note">Prototype warning only · The built-in classifier is trained on '
    'simulated data and is not calibrated for any specific crop, location, or disease. '
    'Weather inputs are user-entered unless refreshed using Open-Meteo. Not a substitute '
    'for agronomic advice.</div>',
    unsafe_allow_html=True,
)

disease_tab, assistant_tab = st.tabs(["📸 Photo screening", "🗣️ Hindi assistant"])

with disease_tab:
    st.subheader("Crop photo and symptom screening")
    st.write(
        "Upload a crop photo and select what you can see. This prototype displays the photo "
        "and combines your symptom selection with field conditions for a preliminary review."
    )
    st.info(
        "No crop-trained computer-vision model is bundled. The app does not automatically "
        "detect disease in image pixels, and this screening is not a diagnosis."
    )
    uploaded_photo = st.file_uploader(
        "Upload a leaf or whole-plant photo",
        type=["jpg", "jpeg", "png", "webp"],
        key="disease_photo",
    )
    if uploaded_photo:
        st.image(uploaded_photo, caption="Uploaded crop photo", width=420)
    visible_symptoms = st.multiselect(
        "Select visible signs",
        [
            "Spots or lesions",
            "White powdery coating",
            "Yellowing",
            "Wilting",
            "Chewed leaves or holes",
        ],
        key="visible_symptoms",
    )
    if visible_symptoms:
        if "White powdery coating" in visible_symptoms:
            st.warning(
                "A white coating can have several causes, including fungal growth. "
                "Check whether it wipes off and ask a local plant-health expert to confirm."
            )
        if "Spots or lesions" in visible_symptoms:
            if humidity_pct >= 75 or leaf_wetness_hours >= 8:
                st.warning(
                    "Leaf spots plus humid or wet conditions merit a closer field check. "
                    "Several fungal, bacterial, and non-infectious causes can look alike."
                )
            else:
                st.info(
                    "Leaf spots have many possible causes. Note their shape and spread "
                    "and check nearby plants for similar symptoms."
                )
        if "Yellowing" in visible_symptoms or "Wilting" in visible_symptoms:
            st.info(
                "Yellowing or wilting can result from water stress, nutrient issues, "
                "root damage, or disease. Check soil moisture and compare affected with "
                "healthy plants."
            )
        if "Chewed leaves or holes" in visible_symptoms:
            st.info(
                f"The current field scouting input gives a {pest_score}% heuristic pest-risk "
                "indicator. Inspect leaf undersides and nearby plants before taking action."
            )
    elif uploaded_photo:
        st.caption("Select visible signs to get cautious, symptom-guided checks.")
    st.caption(
        "For a reliable diagnosis, share clear photos of both sides of the leaf and the whole "
        "plant with a local extension officer. Do not apply treatments from this screening alone."
    )

with assistant_tab:
    st.subheader("Hindi AI assistant · offline demo")
    st.caption(
        "यह बिना इंटरनेट वाला नियम-आधारित सहायक है, जनरेटिव AI नहीं। यह सामान्य मार्गदर्शन "
        "देता है; रोग की पुष्टि या दवा की सिफारिश नहीं करता।"
    )
    if "hindi_chat" not in st.session_state:
        st.session_state.hindi_chat = []
    for chat_message in st.session_state.hindi_chat:
        with st.chat_message(chat_message["role"]):
            st.write(chat_message["content"])

    question = st.chat_input("फसल, मौसम, कीट या सिंचाई के बारे में पूछें")
    if question:
        reply = answer_hindi_question(question, crop, level, stage)
        st.session_state.hindi_chat.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reply},
            ]
        )
        st.rerun()
