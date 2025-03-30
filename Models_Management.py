import os
from joblib import load, dump


class WTV_Models:

    def __init__(self, USAC_Path=None) -> None:
        if USAC_Path is None:
            # Construct a proper file path
            USAC_Path = os.path.join(os.getcwd(), "Models", "usa_cities.joblib")

        if not os.path.exists(USAC_Path):
            raise FileNotFoundError(f"Model file not found at: {USAC_Path}")

        self.model = load(USAC_Path)
        self.cities = list(
            self.model.keys()
        )  # Ensure the model has the expected structure

