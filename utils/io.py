
import pandas as pd
import os
import glob
from .prep import prepare_df


def list_data_files(data_dir='data'):
    """Return a sorted list of CSV files in data_dir matching DATAtourisme naming conventions.

    Tries datatourisme*.csv first, then falls back to any .csv in the folder.
    """
    patterns = [os.path.join(data_dir, 'datatourisme*.csv'), os.path.join(data_dir, '*.csv')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    # Ensure datatourisme-place and datatourisme-product are present and prioritized
    prioritized = []
    place_path = os.path.join(data_dir, 'datatourisme-place.csv')
    prod_path = os.path.join(data_dir, 'datatourisme-product.csv')
    if os.path.exists(place_path):
        prioritized.append(place_path)
    if os.path.exists(prod_path):
        prioritized.append(prod_path)
    # Remove prioritized from files if present and then put them at front
    files = [f for f in sorted(files) if f not in prioritized]
    files = prioritized + files
    # Keep order and uniqueness
    seen = set()
    result = []
    for f in sorted(files):
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def load_raw_csv(path, **kwargs):
    """Load a raw CSV with reasonable defaults for these datasets."""
    kwargs.setdefault('low_memory', False)
    # Try utf-8 then fallback to latin-1
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.read_csv(path, encoding='latin-1', **kwargs)


def load_and_prepare_all(data_dir='data', files=None):
    """Load all CSV files in data_dir, prepare them and concatenate into a single DataFrame.

    Caching behavior:
    - If `data/cleaned.parquet` exists it will be loaded and returned (fast).
    - Otherwise CSVs are loaded, prepared, concatenated and then written to that parquet file.
    """
    cache_path = os.path.join(data_dir, 'cleaned.parquet')
    # If cache exists, load it (fast). This requires pyarrow or fastparquet installed.
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            # If reading parquet fails for any reason, fall back to rebuilding and overwrite cache.
            pass

    files = files or list_data_files(data_dir)
    dfs = []
    for p in files:
        # Accept absolute paths or paths that already include the data_dir prefix
        if os.path.isabs(p) or str(p).startswith(str(data_dir) + os.sep) or str(p).startswith(str(data_dir) + '/'):
            full = p
        else:
            full = os.path.join(data_dir, p)
        if not os.path.exists(full):
            continue
        df = load_raw_csv(full)
        from .prep import prepare_df
        pdf = prepare_df(df, source_file=p)
        pdf['source_file'] = p
        dfs.append(pdf)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)

    # Try to save parquet cache for faster subsequent runs.
    try:
        combined.to_parquet(cache_path, index=False)
    except Exception:
        # Don't fail if parquet write isn't available; just continue without cache.
        pass

    return combined


def save_cleaned(df, path):
    """Save cleaned DataFrame to CSV (no index)."""
    df.to_csv(path, index=False)

