"""Generate a deliberately messy churn-style dataset for testing.

Injected problems: missing values, duplicate rows, numbers stored as text
(with blanks), outliers, a constant column, and inconsistent category labels.

Usage:  python scripts/make_synthetic_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "datasets" / "sample" / "synthetic_dirty_churn.csv"


def main(n: int = 1000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n)
    monthly = np.round(rng.normal(65, 20, n).clip(18, 120), 2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    churn_prob = 0.45 - 0.004 * tenure + 0.002 * (monthly - 65)
    churn_prob = churn_prob + np.where(contract == "Month-to-month", 0.15, -0.1)
    churn = rng.random(n) < churn_prob.clip(0.02, 0.9)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": np.round(tenure * monthly, 2).astype(object),
            "contract": contract,
            "paperless_billing": rng.choice(["Yes", "No"], n),
            "country": "DE",  # constant column
            "churn": np.where(churn, "Yes", "No"),
        }
    )

    # numeric stored as text with blanks (like Telco's TotalCharges)
    blank_idx = rng.choice(n, 12, replace=False)
    df.loc[blank_idx, "total_charges"] = " "
    # missing values
    df.loc[rng.choice(n, 60, replace=False), "monthly_charges"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "contract"] = np.nan
    # outliers
    df.loc[rng.choice(n, 8, replace=False), "monthly_charges"] = 900.0
    # inconsistent labels
    messy = rng.choice(n, 40, replace=False)
    df.loc[messy[:20], "paperless_billing"] = "yes "
    df.loc[messy[20:], "paperless_billing"] = "YES"
    # duplicates
    df = pd.concat([df, df.sample(30, random_state=seed)], ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
