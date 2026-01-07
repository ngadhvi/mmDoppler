import os
from glob import glob
import pandas as pd
import polars as pl


def export_to_parquet(data_folder: str):
    """
    Reads pickle dataframe files from the data_folder, 
    converts them to polars dataframe 
    and stores them as parquet files.

    Args:
        data_folder: Path to the data folder
    """
    if not os.path.exists(data_folder):
        raise ValueError(f"Data folder {data_folder} does not exist")
    
    pickle_dfs = glob(os.path.join(data_folder, "*.pkl"))

    try:
        for df in pickle_dfs:
            df_name = df.split("/")[-1].split(".")[0]
            df = pd.read_pickle(df)
            
            if 'doppz' in df.columns:
                df['doppz'] = df['doppz'].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)
            
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            pl_dataframe = pl.from_pandas(df)
            pl_dataframe.write_parquet(os.path.join(data_folder, f"{df_name}.parquet"))
        
        print("[INFO ✅] Export to parquet completed!")
    
    except Exception as e:
        print(f"[ERROR ❌] Error exporting data as parquet: {e}")


def analyze_data_quality(df: pl.DataFrame, name: str = "Dataset"):
    """
    Analyzes the dataset for duplicates, missing values, and rows with zero objects.
    
    Args:
        df: Polars DataFrame
        name: Name of the dataset for display
    """
    print(f"Analyzing {name}")
    print(f"[Info] Shape: {df.shape}")
    
    # Duplicates
    n_duplicates = df.is_duplicated().sum()
    print(f"[Info] Duplicate Rows: {n_duplicates}")
    
    # Missing Values
    print("[Info] Missing Values per column:")
    print(df.null_count())
    
    # Rows with all nulls
    n_completely_null = df.filter(pl.all_horizontal(pl.all().is_null())).height
    if n_completely_null > 0:
        print(f"[Info] Found {n_completely_null} rows with ALL values missing")
    else:
        print("[Info] No empty rows found in the DataFrame.")

    # Zero objects
    zero_objects = df.filter(pl.col("numDetectedObj") == 0)
    if zero_objects.height > 0:
        print(f"[Info] Found {zero_objects.height} rows with zero objects in the DataFrame.")
    else:
        print("[Info] No rows with zero objects in the DataFrame.")