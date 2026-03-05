import pandas as pd

from src.digital_twin.risk_twin import get_digital_twin_results

df = pd.read_csv(
    "data/raw/training_setA/p000001.psv",
    sep="|"
)

result = get_digital_twin_results(df)

print("\nDigital Twin Result\n")
print(result)
