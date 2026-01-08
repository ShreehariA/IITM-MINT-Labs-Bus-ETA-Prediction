"""
Data loading utilities for IITM Bus GPS Data
Handles loading CSV files from date-organized folders with multiple IMEI files per day
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from tqdm import tqdm
import random


def load_single_file(file_path: Union[str, Path], drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Load a single GPS CSV file and add metadata
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    drop_duplicates : bool, default=True
        Whether to drop duplicate rows (based on all columns)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added date, and unique_id columns
    """
    file_path = Path(file_path)
    
    # Read CSV
    df = pd.read_csv(file_path, parse_dates=['DateTime'])
    
    # Extract metadata from filename: IMEI_866069068704723_20251103.csv
    parts = file_path.stem.split('_')
    df['date'] = parts[2]    # Date
    df['source_file'] = file_path.name
    
    # Create unique_id: IMEI_timestamp_lat_lon
    # This ensures each GPS point is uniquely identifiable
    df['unique_id'] = (
        df['IMEI'].astype(str) + '_' + 
        df['DateTime'].astype(str) + '_' + 
        df['Latitude'].astype(str) + '_' + 
        df['Longitude'].astype(str)
    )
    
    # Drop duplicates if requested (based on ALL columns)
    if drop_duplicates:
        original_len = len(df)
        df = df.drop_duplicates(subset=None, keep='first')  # subset=None means all columns
        if original_len > len(df):
            duplicates_removed = original_len - len(df)
            # Only print if significant duplicates found
            if duplicates_removed > 10:
                print(f"  ⚠ Removed {duplicates_removed} duplicates from {file_path.name}")
    
    return df


def load_single_day(date: str, base_paths: List[str] = None, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Load all buses for a single day
    
    Parameters:
    -----------
    date : str
        Date in format YYYYMMDD (e.g., '20251103')
    base_paths : list of str, optional
        Base directories to search (default: ['1to19nov', '20to24nov'])
    drop_duplicates : bool, default=True
        Whether to drop duplicate rows
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all buses for that day
        
    Example:
    --------
    >>> df = load_single_day('20251103')
    >>> print(f"Loaded {df['IMEI'].nunique()} IMEI with {len(df)} records")
    """
    if base_paths is None:
        base_paths = ['1to19nov', '20to24nov']
    
    dfs = []
    
    for base_path in base_paths:
        date_folder = Path(base_path) / date
        
        if not date_folder.exists():
            continue
        
        csv_files = list(date_folder.glob("IMEI_*.csv"))
        
        for csv_file in tqdm(csv_files, desc=f"Loading {date}", leave=False):
            df = load_single_file(csv_file, drop_duplicates=drop_duplicates)
            dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No data found for date {date}")
    
    original_len = sum(len(df) for df in dfs)
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates across files if requested (based on ALL columns)
    if drop_duplicates:
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=None, keep='first')  # Check all columns
        after_dedup = len(combined_df)
        duplicates_removed = before_dedup - after_dedup
    
    print(f"✓ Loaded {len(dfs)} buses for {date}")
    print(f"  Total records: {len(combined_df):,}")
    if drop_duplicates and duplicates_removed > 0:
        print(f"  Duplicates removed: {duplicates_removed:,} ({duplicates_removed/original_len*100:.1f}%)")
    print(f"  Unique IMEI: {combined_df['IMEI'].nunique()}")
    print(f"  Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return combined_df


def load_multiple_days(dates: List[str], base_paths: List[str] = None, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Load multiple days from multiple folders
    
    Parameters:
    -----------
    dates : list of str
        List of dates in format YYYYMMDD
    base_paths : list of str, optional
        Base directories to search (default: ['1to19nov', '20to24nov'])
    drop_duplicates : bool, default=True
        Whether to drop duplicate rows
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all data
        
    Example:
    --------
    >>> dates = ['20251103', '20251104', '20251106']
    >>> df = load_multiple_days(dates)
    """
    if base_paths is None:
        base_paths = ['1to19nov', '20to24nov']
    
    all_dfs = []
    
    for date in tqdm(dates, desc="Loading dates"):
        try:
            df = load_single_day(date, base_paths, drop_duplicates=drop_duplicates)
            all_dfs.append(df)
        except ValueError as e:
            print(f"⚠ Warning: {e}")
            continue
    
    if not all_dfs:
        raise ValueError("No data loaded for any of the specified dates")
    
    original_len = sum(len(df) for df in all_dfs)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Drop duplicates across days if requested (based on ALL columns)
    if drop_duplicates:
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=None, keep='first')  # Check all columns
        after_dedup = len(combined_df)
        duplicates_removed = before_dedup - after_dedup
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Loaded {len(dates)} days")
    print(f"  Total records: {len(combined_df):,}")
    if drop_duplicates and duplicates_removed > 0:
        print(f"  Duplicates removed: {duplicates_removed:,} ({duplicates_removed/original_len*100:.1f}%)")
    print(f"  Unique IMEI: {combined_df['IMEI'].nunique()}")
    print(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"  Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"{'='*60}\n")
    
    return combined_df


def smart_sample(
    n_days: int = 5,
    n_buses_per_day: int = 10,
    base_paths: List[str] = None,
    random_state: int = 42,
    drop_duplicates: bool = True
) -> pd.DataFrame:
    """
    Smart sampling strategy: randomly sample days and buses
    
    Parameters:
    -----------
    n_days : int
        Number of days to sample
    n_buses_per_day : int
        Number of buses to sample per day
    base_paths : list of str, optional
        Base directories to search
    random_state : int
        Random seed for reproducibility
    drop_duplicates : bool, default=True
        Whether to drop duplicate rows
        
    Returns:
    --------
    pd.DataFrame
        Sampled data
        
    Example:
    --------
    >>> df = smart_sample(n_days=5, n_buses_per_day=10)
    """
    if base_paths is None:
        base_paths = ['1to19nov', '20to24nov']
    
    random.seed(random_state)
    
    # Get all date folders
    all_dates = set()
    for base_path in base_paths:
        base = Path(base_path)
        if base.exists():
            all_dates.update([d.name for d in base.iterdir() if d.is_dir()])
    
    all_dates = sorted(all_dates)
    
    # Sample dates
    sample_dates = random.sample(all_dates, min(n_days, len(all_dates)))
    
    print(f"Sampling {len(sample_dates)} days: {sample_dates}")
    
    dfs = []
    
    for date in tqdm(sample_dates, desc="Sampling data"):
        # Get all files for this date
        files = []
        for base_path in base_paths:
            date_folder = Path(base_path) / date
            if date_folder.exists():
                files.extend(list(date_folder.glob("IMEI_*.csv")))
        
        # Sample buses
        sample_files = random.sample(files, min(n_buses_per_day, len(files)))
        
        for csv_file in sample_files:
            df = load_single_file(csv_file, drop_duplicates=drop_duplicates)
            dfs.append(df)
    
    original_len = sum(len(df) for df in dfs)
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates across sampled data if requested (based on ALL columns)
    if drop_duplicates:
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=None, keep='first')  # Check all columns
        after_dedup = len(combined_df)
        duplicates_removed = before_dedup - after_dedup
    
    print(f"\n{'='*60}")
    print(f"SMART SAMPLE: {n_days} days × {n_buses_per_day} buses")
    print(f"  Total records: {len(combined_df):,}")
    if drop_duplicates and duplicates_removed > 0:
        print(f"  Duplicates removed: {duplicates_removed:,} ({duplicates_removed/original_len*100:.1f}%)")
    print(f"  Unique IMEI: {combined_df['IMEI'].nunique()}")
    print(f"  Unique days: {combined_df['date'].nunique()}")
    print(f"  Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"{'='*60}\n")
    
    return combined_df


def get_available_dates(base_paths: List[str] = None) -> List[str]:
    """
    Get list of all available dates in the dataset
    
    Parameters:
    -----------
    base_paths : list of str, optional
        Base directories to search
        
    Returns:
    --------
    list of str
        Sorted list of available dates
    """
    if base_paths is None:
        base_paths = ['1to19nov', '20to24nov']
    
    all_dates = set()
    for base_path in base_paths:
        base = Path(base_path)
        if base.exists():
            all_dates.update([d.name for d in base.iterdir() if d.is_dir()])
    
    return sorted(all_dates)


def get_buses_for_date(date: str, base_paths: List[str] = None) -> List[str]:
    """
    Get list of all bus IDs (IMEIs) available for a specific date
    
    Parameters:
    -----------
    date : str
        Date in format YYYYMMDD
    base_paths : list of str, optional
        Base directories to search
        
    Returns:
    --------
    list of str
        List of IMEI numbers
    """
    if base_paths is None:
        base_paths = ['1to19nov', '20to24nov']
    
    buses = set()
    
    for base_path in base_paths:
        date_folder = Path(base_path) / date
        if date_folder.exists():
            for csv_file in date_folder.glob("IMEI_*.csv"):
                imei = csv_file.stem.split('_')[1]
                buses.add(imei)
    
    return sorted(buses)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting to appropriate dtypes
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Optimized DataFrame
    """
    df_optimized = df.copy()
    
    # Convert float64 to float32
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = df_optimized[col].astype('float32')
    
    # Convert object columns to category if they have few unique values
    for col in df_optimized.select_dtypes(include=['object']).columns:
        if col not in ['DateTime', 'Packet Received']:  # Don't convert datetime strings
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized)
            if num_unique / num_total < 0.5:  # If less than 50% unique
                df_optimized[col] = df_optimized[col].astype('category')
    
    # Convert IMEI to category
    if 'IMEI' in df_optimized.columns:
        df_optimized['IMEI'] = df_optimized['IMEI'].astype('category')
    
    if 'date' in df_optimized.columns:
        df_optimized['date'] = df_optimized['date'].astype('category')
    
    # Print memory savings
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
    savings = (1 - optimized_memory / original_memory) * 100
    
    print(f"Memory optimization:")
    print(f"  Original: {original_memory:.2f} MB")
    print(f"  Optimized: {optimized_memory:.2f} MB")
    print(f"  Savings: {savings:.1f}%")
    
    return df_optimized


def dataset_summary(base_paths: List[str] = None) -> pd.DataFrame:
    """
    Get a summary of the entire dataset without loading all data
    
    Parameters:
    -----------
    base_paths : list of str, optional
        Base directories to search
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics by date
    """
    if base_paths is None:
        base_paths = ['1to19nov', '20to24nov']
    
    dates = get_available_dates(base_paths)
    
    summary_data = []
    
    for date in tqdm(dates, desc="Scanning dataset"):
        buses = get_buses_for_date(date, base_paths)
        
        # Count total files and estimate size
        total_files = 0
        total_size = 0
        
        for base_path in base_paths:
            date_folder = Path(base_path) / date
            if date_folder.exists():
                files = list(date_folder.glob("IMEI_*.csv"))
                total_files += len(files)
                total_size += sum(f.stat().st_size for f in files)
        
        summary_data.append({
            'date': date,
            'num_buses': len(buses),
            'num_files': total_files,
            'size_mb': total_size / 1024**2
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"  Total dates: {len(summary_df)}")
    print(f"  Total files: {summary_df['num_files'].sum()}")
    print(f"  Total size: {summary_df['size_mb'].sum():.2f} MB")
    print(f"  Avg buses per day: {summary_df['num_buses'].mean():.1f}")
    print(f"  Date range: {summary_df['date'].min()} to {summary_df['date'].max()}")
    print(f"{'='*60}\n")
    
    return summary_df


# Example usage
if __name__ == "__main__":
    print("IITM Bus GPS Data Loader\n")
    
    # Show available dates
    print("Available dates:")
    dates = get_available_dates()
    print(f"  {len(dates)} dates from {dates[0]} to {dates[-1]}\n")
    
    # Get dataset summary
    summary = dataset_summary()
    print(summary.head(10))
    
    # Example: Load single day
    print("\n" + "="*60)
    print("Example 1: Load single day (Nov 3)")
    print("="*60)
    df_single = load_single_day('20251103')
    print(f"\nFirst few rows:")
    print(df_single.head())
    
    # Example: Smart sample
    print("\n" + "="*60)
    print("Example 2: Smart sample (3 days, 5 buses each)")
    print("="*60)
    df_sample = smart_sample(n_days=3, n_buses_per_day=5)
    print(f"\nSample info:")
    print(df_sample.info())
