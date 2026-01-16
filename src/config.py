import os
from pathlib import Path
import pandas as pd
import random

class StationDataFile:
    def __init__(self,station_id):
        self.station_id = station_id
        self.folder = Config.STATION_PATH / station_id
        self.point_data = self.folder / f"{station_id}_point_data.pkl"
        self.price_data = self.folder / f"{station_id}_price.pkl"
    
    def get_price_df(self):
        return pd.read_pickle(self.price_data)
    
    def get_point_data_df(self):
        return pd.read_pickle(self.point_data)  
    
    def get_day_df_random(self):
        point_data_df = pd.read_pickle(self.point_data)
        # Convert datetime strings like '2025-05-06 17:00:00' to datetime objects
        point_data_df['datetime'] = pd.to_datetime(point_data_df['datetime'], format='%Y-%m-%d %H:%M:%S')
        point_data_df['date'] = point_data_df['datetime'].dt.date
        unique_dates = point_data_df['date'].unique()
        random_date = random.choice(unique_dates)
        day_df = point_data_df[point_data_df['date'] == random_date]
        return day_df

class Config:
    ROOT_PATH = Path(__file__).parent.parent.resolve()
    DATA_PATH = ROOT_PATH.parent / "Interpretability_action_dataset"
    STATION_PATH = DATA_PATH / "station_data"
    PROCESSED_DATA_PATH = ROOT_PATH / "data"
    PROFIT_PATH = DATA_PATH / "profit_metrics_info_df_cleaned.pkl"
    INFO_PATH = DATA_PATH / "filtered_station_info.pkl"

    @staticmethod
    def get_df_from_pkl(pkl_path):
        import pandas as pd
        return pd.read_pickle(pkl_path)
    @staticmethod
    def get_random_station_id():
        station_dirs = [d for d in os.listdir(Config.STATION_PATH) if os.path.isdir(Config.STATION_PATH / d)]
        return random.choice(station_dirs)
    
if __name__ == "__main__":
    # df = Config.get_df_from_pkl(Config.PROFIT_PATH)
    # df.to_csv(f"./data/sample_profit_data.csv", index=False)
    # df = Config.get_df_from_pkl(Config.INFO_PATH)
    # df.to_csv(f"./data/sample_station_info.csv", index=False)
    # station_id = Config.get_random_station_id()
    # station_data_file = StationDataFile(station_id)
    # df = station_data_file.get_point_data_df()
    # df.to_csv(f"./data/sample_point_data_{station_id}.csv", index=False)

# 2026-01-16 17:15:56,656 - Preprocessor - INFO - Skipping station 5725606: telemetry file missing.
# 2026-01-16 17:15:56,833 - Preprocessor - INFO - Station 5742094, Day 2025-09-28: No telemetry points found, skipping.
    sid = "5725606"
    station_data_file = StationDataFile(sid)
    df = station_data_file.get_point_data_df()
    print(df.head())