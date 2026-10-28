# FasalShield AI — Early Crop Risk Warning

A Streamlit prototype that combines weather, crop stage, field-observed pest pressure,
and forecast signals into an early crop-risk estimate. It is designed as a farmer-facing
starting point, not as a disease diagnosis or a substitute for local agronomic advice.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use the sidebar to select a crop and growth stage, then enter local weather and scouting
observations. **Fetch live weather** optionally looks up current temperature, humidity,
and a seven-day daily forecast through the Open-Meteo geocoding and forecast APIs.
No API key is required; the dashboard remains usable with manually entered readings.

## Feature coverage

- **Crop health dashboard:** combined risk, crop stage, weather signals, recommendations,
  and a session-only field observation log.
- **Weather API and alerts:** current conditions, seven-day forecast chart, and an in-app
  moderate/high risk alert.
- **Pest-risk indicator:** separate scouting-and-weather heuristic alongside the combined risk.
- **Smart irrigation:** cautious guidance based on selected soil moisture, crop stage,
  temperature, and forecast rain; it does not prescribe a water volume.
- **Explainability:** signal-strength chart plus a clearly labeled illustrative uncertainty
  range. The range is not a calibrated confidence interval.
- **Photo screening:** upload and preview a photo, then select visible symptoms for cautious
  symptom-guided checks.
- **Hindi assistant:** offline, rule-based answers about common crop-risk, pest, symptom,
  and irrigation questions.

## Prototype limitations

The included NumPy logistic-regression classifier is trained at startup on deterministic, simulated
examples generated from transparent risk assumptions. Its probability is for demonstrating
the end-to-end interface only; it is **not field-calibrated**, crop- or disease-specific,
or suitable for making treatment decisions. A production model needs locally validated
crop/disease observations, agronomist-reviewed labels, evaluation, and ongoing monitoring.
The signal chart shows relative signal strength, not disease probability or causal effect.
No trained computer-vision model is included: the photo screen does not automatically identify
disease from image pixels. The Hindi assistant is a small offline ruleset, not a generative AI
system. Weather, risk, irrigation, and screening outputs are prototypes and must not be used
alone to make treatment decisions.
