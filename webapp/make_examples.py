"""Write one golden ride per category into webapp/examples/, the example rides shipped with the web app.

make webapp-examples     (needs data/processed/golden.parquet, i.e. `make pipeline`)

Golden rides were never used to train the model, so the demo shows honest predictions. The files keep the true
category in a `label` column, which the page shows next to the prediction (the API ignores it).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "examples"
COLUMNS = ["time_index", "ax", "ay", "az", "rx", "ry", "rz", "label"]

golden = pd.read_parquet(ROOT / "data" / "processed" / "golden.parquet")
OUT.mkdir(exist_ok=True)
for label, rides in golden.groupby("label"):
    first_ride = rides[rides["ride_id"] == rides["ride_id"].iloc[0]]
    path = OUT / f"{label.lower().replace(' ', '_')}.parquet"
    first_ride[COLUMNS].to_parquet(path, index=False)
    print(f"{path.relative_to(ROOT)}: {len(first_ride)} readings")
