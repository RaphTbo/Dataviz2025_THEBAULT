
import pandas as pd
import numpy as np
import re


def prepare_df(df, source_file: str = None):
    """Normalize and clean a raw datatourisme dataframe.

    - Normalize column names (lowercase, strip, underscores)
    - Coerce latitude/longitude/altitude to numeric where possible
    - Ensure common textual fields exist (nom, type, theme, region, departement, description)
    - Add basic deduplication
    - Optionally infer region from source_file name when region is missing

    Returns cleaned DataFrame.
    """
    # Normalize column names
    def _norm_col(c):
        if c is None:
            return c
        c = str(c).strip()
        c = c.lstrip('\ufeff')  # remove BOM if present
        c = c.lower().replace(' ', '_')
        return c

    df = df.rename(columns=lambda c: _norm_col(c))

    # Helper to map possible source columns to a canonical output
    def _map_first(existing, candidates):
        for cand in candidates:
            if cand in existing:
                return cand
        return None

    cols = set(df.columns)

    # Latitude / longitude / altitude: try common names
    lat_col = _map_first(cols, ['latitude', 'lat'])
    lon_col = _map_first(cols, ['longitude', 'lon', 'lng'])
    alt_col = _map_first(cols, ['altitude', 'alt'])

    if lat_col:
        df['latitude'] = pd.to_numeric(df[lat_col], errors='coerce')
    else:
        df['latitude'] = np.nan

    if lon_col:
        df['longitude'] = pd.to_numeric(df[lon_col], errors='coerce')
    else:
        df['longitude'] = np.nan

    if alt_col:
        df['altitude'] = pd.to_numeric(df[alt_col], errors='coerce')
    else:
        df['altitude'] = np.nan

    # Textual fields: try some common alternatives
    # name candidates for POI/Product files
    name_col = _map_first(cols, ['nom', 'name', 'title', 'nom_du_poi', 'nom_produit', 'nom_du_produit'])
    if name_col:
        df['nom'] = df[name_col].astype(str)
    else:
        df['nom'] = ''

    # try to derive a simple type/category from datatourisme columns
    type_col = _map_first(cols, ['type', 'categories_de_poi', 'categories', 'classements_du_poi'])
    if type_col:
        # extract a readable label if the column contains URIs (take fragment after '#')
        def _simplify_type(v):
            s = str(v)
            if '#' in s:
                return s.split('#')[-1]
            if '/' in s:
                return s.split('/')[-1]
            return s
        df['type'] = df[type_col].fillna('').astype(str).apply(_simplify_type)
    else:
        df['type'] = ''

    theme_col = _map_first(cols, ['theme'])
    if theme_col:
        df['theme'] = df[theme_col].astype(str)
    else:
        df['theme'] = ''

    dept_col = _map_first(cols, ['departement', 'department'])
    if dept_col:
        df['departement'] = df[dept_col].astype(str)
    else:
        df['departement'] = ''

    desc_col = _map_first(cols, ['description', 'desc'])
    if desc_col:
        df['description'] = df[desc_col].astype(str)
    else:
        df['description'] = ''

    # Parse common date columns (e.g., date_de_mise_a_jour) into datetime
    date_col = _map_first(cols, ['date_de_mise_a_jour', 'date_maj', 'date'])
    if date_col:
        try:
            df['date_maj'] = pd.to_datetime(df[date_col], errors='coerce')
        except Exception:
            df['date_maj'] = pd.NaT
    else:
        df['date_maj'] = pd.NaT

    # Region: keep if present and non-empty, otherwise try to infer from source filename
    if 'region' not in df.columns:
        df['region'] = ''
    else:
        df['region'] = df['region'].fillna('').astype(str)

    if source_file:
        # If all region values are empty or whitespace/NaN, infer from filename.
        # Use fillna + str.strip to avoid pandas' replace downcasting FutureWarning.
        region_s = df['region'].fillna('').astype(str).str.strip()
        if region_s.eq('').all():
            m = re.search(r'reg-([a-z]{3})', source_file, re.I)
            if m:
                code = m.group(1).upper()
                df['region'] = code

    # Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def filter_mountain(df):
    """Return subset of rows matching mountain-related keywords (heuristic)."""
    kw = ['montagne', 'randonnée', 'refuge', 'alpin', 'alpine', 'ski', 'pic', 'sommet', 'piste']
    pattern = '|'.join(kw)
    mask = pd.Series(False, index=df.index)
    for col in ['theme', 'type', 'nom', 'description']:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(pattern, na=False)
    return df[mask]
