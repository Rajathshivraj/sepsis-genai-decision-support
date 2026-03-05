import pandas as pd

from src.forecasting.risk_forecast import forecast_future_risk

df = pd.read_csv(
    "data/raw/training_setA/p000001.psv",
    sep="|"
)

result = forecast_future_risk(df)

print("\nForecast Result\n")
print(result)
