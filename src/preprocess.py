import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import Config, StationDataFile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Preprocessor")

@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""
    data_root: Path = Config.DATA_PATH
    station_data_dir: Path = Config.STATION_PATH
    profit_path: Path = Config.PROFIT_PATH
    info_path: Path = Config.INFO_PATH
    output_dir: Path = Config.PROCESSED_DATA_PATH
    
    # Data Quality parameters
    required_points_per_day: int = 288  # Standard target (5-min intervals)
    price_series_len: int = 288
    fill_value: float = -999.0          # Value for missing/lost points
    
    # Feature selections base on ExpectedFormat in readme.md
    metadata_cols: List[str] = field(default_factory=lambda: [
        'battery_capacity', 'power_limit', 'charge_type', 'profit_ai', 'profit_self'
    ])

    series_cols: List[str] = field(default_factory=lambda: [
        'Daytime', 'Price_purchase', 'Price_sell', 'PV_forcast', 'PV_real', 
        'Load_forcast', 'Load_real', 'Battery_dis', 'Battery_cha', 'Battery_soc'
    ])

    # Execution settings
    use_multiprocessing: bool = True
    max_workers: int = 4

class EnergyDataPreprocessor:
    """
    Highly structured preprocessor for energy station telemetry and pricing data.
    Aggregates data into daily instances suitable for Deep Learning.
    """
    
    def __init__(self, config: PreprocessConfig = PreprocessConfig()):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.profit_df = None
        self.station_info = None
        self.station_meta_map = {}

    def load_master_tables(self):
        """Load and index the global metadata tables."""
        logger.info("Loading master metadata tables...")
                
        self.profit_df = Config.get_df_from_pkl(self.config.profit_path)
        self.info_df = Config.get_df_from_pkl(self.config.info_path)
        
        # Standardize IDs and types
        self.profit_df['pvs_site_id'] = self.profit_df['pvs_site_id'].astype(str)
        self.profit_df['date_str'] = pd.to_datetime(self.profit_df['date']).dt.date.astype(str)
        
        self.info_df['pvs_site_id'] = self.info_df['pvs_site_id'].astype(str)
        self.station_meta_map = self.info_df.set_index('pvs_site_id').to_dict('index')
        
        logger.info(f"Loaded {len(self.profit_df)} profit records and metadata for {len(self.info_df)} stations.")

    def _parse_price_series(self, price_row_subset: pd.DataFrame) -> Any:
        """
        Load price_array from the subset. 
        Supports price_type 1 (selling) and 2 (buying) with charge_type 0 (fixed) or 4 (dynamic).
        """
        if price_row_subset.empty:
            return None
         
        price_data = price_row_subset.iloc[0]['price_array']
        return np.array(price_data) if price_data is not None else None

    def process_station(self, sid: str) -> List[Dict[str, Any]]:
        """Processes all valid days for a single station."""
        results = []
        
        station_files = StationDataFile(sid)
        if not station_files.point_data.exists():
            logger.info(f"Skipping station {sid}: telemetry file missing.")
            return []
        if not station_files.price_data.exists():
            logger.info(f"Skipping station {sid}: price file missing.")
            return []
            
        try:
            df_points = station_files.get_point_data_df()
            df_prices = station_files.get_price_df()
        except Exception as e:
            logger.error(f"Error loading pkl for station {sid}: {e}")
            return []
            
        # Standardize time
        df_points['datetime'] = pd.to_datetime(df_points['datetime'])
        df_points['date_str'] = df_points['datetime'].dt.date.astype(str)
        df_prices['date_str'] = pd.to_datetime(df_prices['date']).dt.date.astype(str)
        
        # Get days that have profit statistics
        target_days = self.profit_df[self.profit_df['pvs_site_id'] == sid].copy()
        if target_days.empty:
            logger.info(f"Station {sid} has no records in profit master table.")
            return []

        meta = self.station_meta_map.get(sid, {})
        
        for _, p_row in target_days.iterrows():
            d_str = p_row['date_str']
            
            # 1. Filter telemetry for the day
            day_raw = df_points[df_points['date_str'] == d_str].copy()
            if day_raw.empty:
                logger.info(f"Station {sid}, Day {d_str}: No telemetry points found, skipping.")
                continue
            
            # Map each point to a fixed 5-min slot (0-287)
            day_raw['slot'] = (day_raw['datetime'].dt.hour * 12 + 
                              day_raw['datetime'].dt.minute // 5).astype(int)
            
            # Handle duplicates: keep the last telemetry point for a slot
            day_cleansed = day_raw.drop_duplicates('slot', keep='last')
            valid_slots = day_cleansed['slot'].values
            
            # 2. Standardize Daytime (Full 24h range)
            full_day_range = pd.date_range(
                start=pd.Timestamp(d_str), 
                periods=self.config.required_points_per_day, 
                freq='5min'
            ).values
            
            # 3. Filter and process prices
            day_price_info = df_prices[df_prices['date_str'] == d_str]
            buy_price = self._parse_price_series(day_price_info[day_price_info['price_type'] == 2])
            sell_price = self._parse_price_series(day_price_info[day_price_info['price_type'] == 1])
            
            # Robust length check using np.size to avoid TypeError on scalars
            def _validate_price(arr):
                if arr is None or np.size(arr) != self.config.required_points_per_day:
                    return np.full(self.config.required_points_per_day, self.config.fill_value)
                return arr

            buy_price = _validate_price(buy_price)
            sell_price = _validate_price(sell_price)
            
            # Construct Day Instance
            instance = {
                'station_id': sid,
                'date': d_str,
            }

            # Property Extraction
            meta_source = {
                'power_limit': meta.get('inverter_pv_capacity', 0) + meta.get('hybrid_inverter_pv_capacity', 0),
                'battery_capacity': meta.get('battery_capacity', 0),
                'charge_type': day_price_info['charge_type'].iloc[0] if not day_price_info.empty else 0,
                'profit_ai': p_row.get('station_day_profit_ai_mode', 0),
                'profit_self': p_row.get('station_day_profit_simulink_mode', 0),
            }
            for col in self.config.metadata_cols:
                instance[col] = meta_source.get(col, 0)

            # Slot-based filling: Each position corresponds strictly to the time of day
            def fill_aligned(source_df, col_name):
                arr = np.full(self.config.required_points_per_day, self.config.fill_value)
                if col_name in source_df.columns:
                    # Place values at their respective indices (0-287)
                    vals = source_df[col_name].fillna(self.config.fill_value).values
                    arr[valid_slots] = vals
                return arr

            instance['Daytime'] = full_day_range
            instance['Price_purchase'] = buy_price
            instance['Price_sell'] = sell_price
            instance['PV_forcast'] = fill_aligned(day_cleansed, 'pv_forecast')
            instance['PV_real'] = fill_aligned(day_cleansed, 'pv_power')
            instance['Load_forcast'] = fill_aligned(day_cleansed, 'load_forecast')
            instance['Load_real'] = fill_aligned(day_cleansed, 'load_power')
            instance['Battery_dis'] = fill_aligned(day_cleansed, 'battery_discharge_power')
            instance['Battery_cha'] = fill_aligned(day_cleansed, 'battery_charge_power')
            instance['Battery_soc'] = fill_aligned(day_cleansed, 'battery_soc')
            
            results.append(instance)
            
        return results

    def run(self, sample_n: Optional[int] = None):
        """Execute the full preprocessing pipeline."""
        if self.profit_df is None:
            self.load_master_tables()
            
        stations = self.profit_df['pvs_site_id'].unique()
        if sample_n:
            stations = stations[:sample_n]
            logger.info(f"Subsampling to {sample_n} stations for testing.")
            
        all_instances = []
        
        logger.info(f"Starting aggregation for {len(stations)} stations...")
        
        if self.config.use_multiprocessing:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(self.process_station, sid): sid for sid in stations}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Aggregation"):
                    all_instances.extend(future.result())
        else:
            for sid in tqdm(stations, desc="Aggregation"):
                all_instances.extend(self.process_station(sid))
                
        if not all_instances:
            logger.warning("No data instances were created. Check data paths and filters.")
            return
            
        # Convert to DataFrame
        final_df = pd.DataFrame(all_instances)
        logger.info(f"Aggregation complete. Created {len(final_df)} daily instances.")
        
        # Save results
        output_path = self.config.output_dir / "aggregated_data.pkl"
        final_df.to_pickle(output_path)
        logger.info(f"Saved aggregated data to {output_path}")
        
        return final_df

if __name__ == "__main__":
    config = PreprocessConfig(use_multiprocessing=False)
    preprocessor = EnergyDataPreprocessor(config)
    preprocessor.run()

