import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from utils.io import load_and_prepare_all
from utils.departments import dept_name
from sections import story_mountain
from sections.intro import show_intro
from sections.overview import show_overview

st.set_page_config(page_title="Tourisme montagne", layout="wide")

# Map rendering limits to avoid Streamlit message size errors
MAX_MAP_POINTS = 10000  # max number of points to send to the browser for the pydeck map (scatter)
MAX_MAP_POINTS_HEAT = 150000  # larger cap for heatmap/hexagon layers (they use minimal payload)


@st.cache_data
def get_data():
    """Charge et retourne le DataFrame combiné nettoyé."""
    return load_and_prepare_all('data')


def extract_city_and_postal(df):
    """Try to extract city and postal code from common columns."""
    # Prefer a column `code_postal_et_commune` containing 'XXXXX#City'
    if 'code_postal_et_commune' in df.columns:
        parts = df['code_postal_et_commune'].fillna('').astype(str).str.split('#', n=1)
        df['postal_code'] = parts.str[0].replace('', pd.NA)
        df['city'] = parts.str[1].replace('', pd.NA)
    else:
        # fallback to adresse_postale: take last token after comma as city
        if 'adresse_postale' in df.columns:
            addr = df['adresse_postale'].fillna('').astype(str)
            df['city'] = addr.str.split(',').str[-1].str.strip().replace('', pd.NA)
        else:
            df['city'] = pd.NA
        df['postal_code'] = pd.NA
    # Derive department code from postal_code when departement is missing or empty
    if 'departement' not in df.columns:
        df['departement'] = pd.NA

    # Normalize existing departement column to strings (empty if missing)
    df['departement'] = df['departement'].fillna('').astype(str).str.strip()

    # If postal_code looks like a 5-digit code, take first two characters as department
    try:
        pc = df['postal_code'].fillna('').astype(str).str.strip()
        derived = pc.str[:2].where(pc.str.len() >= 2, '')
        mask_empty_dept = df['departement'].eq('') & derived.ne('')
        df.loc[mask_empty_dept, 'departement'] = derived[mask_empty_dept]
    except Exception:
        pass

    # Replace remaining empty strings with NA for clarity
    df['departement'] = df['departement'].replace('', pd.NA)

    return df



def make_type_simple(df):
    if 'type' in df.columns:
        def _simple_type(s):
            s = str(s)
            if '|' in s:
                s = s.split('|')[0]
            if '#' in s:
                s = s.split('#')[-1]
            if '/' in s:
                s = s.split('/')[-1]
            return s
        df['type_simple'] = df['type'].fillna('').astype(str).apply(_simple_type)
    else:
        df['type_simple'] = ''
    return df


# --- Load data and prepare columns ---
df = get_data()
df = extract_city_and_postal(df)
df = make_type_simple(df)

# Map detailed types into broader, user-friendly categories for filtering
def _map_type_category(s: str) -> str:
    s2 = str(s or '').lower()
    if any(k in s2 for k in ['restaur', 'bar', 'café', 'cafe', 'bistrot', 'brasserie']):
        return 'Restaurant'
    if any(k in s2 for k in ['sport', 'rand', 'velo', 'vélo', 'piscine', 'escalade', 'kayak', 'canoe', 'canoë', 'ski', 'plongée', 'plongee']):
        return 'Activité sportive'
    if any(k in s2 for k in ['musée', 'musee', 'château', 'chateau', 'église', 'eglise', 'monument', 'site', 'parc', 'jardin', 'aquarium', 'abbaye', 'tour']):
        return 'Monument / Site'
    if any(k in s2 for k in ['visite', 'visites', "à visiter", 'a visiter', "à voir", 'a voir', 'balade', 'promenade', 'parcours', 'point d']):
        return 'Places à visiter'
    if any(k in s2 for k in ['hébergement', 'hebergement', 'hotel', 'gite', 'gîte', 'chambre']):
        return 'Hébergement'
    if s2.strip() == '':
        return 'Inconnu'
    return 'Autre'

# create a broader category column
df['type_category'] = df['type_simple'].fillna('').apply(_map_type_category)

# Create a human-readable department name column using our mapping helper
if 'departement' in df.columns:
    try:
        df['departement_name'] = df['departement'].fillna('').astype(str).apply(lambda x: dept_name(x))
    except Exception:
        df['departement_name'] = df['departement']
else:
    df['departement_name'] = pd.NA

# --- Region selection (default to PAC/OCC/ARA but allow others) ---
if 'region' in df.columns:
    region_choices = sorted([r for r in df['region'].fillna('').astype(str).str.upper().unique().tolist() if r])
else:
    region_choices = []
default_regions = ['PAC (Provence-Alpes-Côte d\'Azur)', 'OCC (Occitanie)', 'ARA (Auvergne-Rhône-Alpes)']
# Sidebar region multiselect will be created further down with other filters; store choices here
available_region_choices = region_choices


# Ensure a usable date column
if 'date_maj' in df.columns:
    df['date_maj'] = pd.to_datetime(df['date_maj'], errors='coerce')
elif 'date_de_mise_a_jour' in df.columns:
    df['date_maj'] = pd.to_datetime(df['date_de_mise_a_jour'], errors='coerce')
else:
    df['date_maj'] = pd.NaT


# --- SIDEBAR: filters focused on type / departement / city / date ---
st.sidebar.header('Filtres')

# Quick guide for new users: short explanation placed at top of the sidebar
st.sidebar.info(
    "Utilisez les filtres ci-dessous (Catégorie, Région, Département, Ville, Période) \n"
    "pour restreindre les objets affichés. \n"
    "Choisissez la section 'Story — Tourisme montagne' pour une narration saisonnière ciblée."
)

# Helper: render a list of checkboxes in the sidebar and return selected options
def sidebar_checklist(label: str, options: list, default: list | None = None, key: str | None = None) -> list:
    """Render a vertical list of checkboxes in the sidebar.
    - label: shown above the list
    - options: iterable of option labels (strings)
    - default: list of options that should be checked initially (or None)
    - key: stable prefix to avoid Streamlit key collisions
    Returns: list of selected options (strings)
    """
    if key is None:
        key = label.replace(' ', '_')
    st.sidebar.markdown(f"**{label}**")
    default_set = set(default) if default else set()
    select_all_key = f"{key}__select_all"
    select_all_default = (set(options) == default_set) if default is not None else False
    select_all = st.sidebar.checkbox('Tout sélectionner', value=select_all_default, key=select_all_key)
    selected = []
    for i, opt in enumerate(options):
        opt_key = f"{key}__opt_{i}"
        checked = select_all or (opt in default_set)
        if st.sidebar.checkbox(opt, value=checked, key=opt_key):
            selected.append(opt)
    return selected


def container_checklist(container, label: str, options: list, default: list | None = None, key: str | None = None) -> list:
    """Render the same checklist but inside a given Streamlit container (expander or sidebar).
    container: Streamlit container object supporting markdown() and checkbox()
    """
    if key is None:
        key = label.replace(' ', '_')
    container.markdown(f"**{label}**")
    default_set = set(default) if default else set()
    select_all_key = f"{key}__select_all"
    select_all_default = (set(options) == default_set) if default is not None else False
    select_all = container.checkbox('Tout sélectionner', value=select_all_default, key=select_all_key)
    selected = []
    for i, opt in enumerate(options):
        opt_key = f"{key}__opt_{i}"
        checked = select_all or (opt in default_set)
        if container.checkbox(opt, value=checked, key=opt_key):
            selected.append(opt)
    return selected

# Category filter (more general groups) rendered inside a collapsible expander like Département
type_choices = sorted([t for t in df['type_category'].fillna('').astype(str).str.strip().unique().tolist() if t])
if type_choices:
    type_expander = st.sidebar.expander("Catégorie d'activité", expanded=False)
    with type_expander:
        sel_types = container_checklist(st, "Catégorie d'activité", type_choices, default=type_choices, key='type_category')
else:
    sel_types = []

# Region filter (default to PAC/OCC/ARA if present)
if available_region_choices:
    region_expander = st.sidebar.expander('Région', expanded=False)
    default_sel = [r for r in default_regions if r in available_region_choices]
    with region_expander:
        sel_regions = container_checklist(region_expander, 'Région', available_region_choices, default=default_sel, key='region')
else:
    sel_regions = []

# Departement filter (collapsible)
dept_choices = []
if 'departement' in df.columns:
    dept_choices = sorted([d for d in df['departement'].fillna('').astype(str).str.strip().unique().tolist() if d])

# show readable labels like '04 — Alpes-de-Haute-Provence' but keep underlying value as code
dept_display = [f"{c} — {dept_name(c)}" for c in dept_choices]

# Render the department checklist inside a collapsible expander in the sidebar
sel_dept_display = []
if dept_display:
    dept_expander = st.sidebar.expander('Département', expanded=False)
    with dept_expander:
        st.markdown("**Département**")
        # maintain same keying & "select all" behavior as sidebar_checklist
        default_set = set(dept_display)
        select_all_key = f"departement__select_all"
        select_all_default = True if dept_display else False
        select_all = st.checkbox('Tout sélectionner', value=select_all_default, key=select_all_key)
        for i, opt in enumerate(dept_display):
            opt_key = f"departement__opt_{i}"
            checked = select_all or True  # default: departments listed are shown as checked by default
            if st.checkbox(opt, value=checked, key=opt_key):
                sel_dept_display.append(opt)
else:
    sel_dept_display = []

# Map back to codes for filtering
sel_depts = [s.split(' — ')[0] for s in sel_dept_display]

# City filter (optionnel) — saisie avec suggestions
# We provide a text input where the user types letters and a selectbox
# shows matching city suggestions (substring match). If none selected,
# no city filter is applied.
city_choices = sorted([c for c in df['city'].dropna().astype(str).str.strip().unique().tolist() if c])
city_query = st.sidebar.text_input('Ville (optionnel) — tapez pour chercher', value='', key='city_query')
matched_cities = []
if city_query:
    q = city_query.strip().lower()
    # substring match; show up to 200 suggestions to avoid huge widgets
    matched_cities = [c for c in city_choices if q in c.lower()][:200]
    if not matched_cities:
        st.sidebar.write('Aucune suggestion pour cette saisie.')

# Present a selectbox of matches (or allow selecting 'Toutes' when query empty)
if city_query:
    options_for_select = ['--Choisir une ville--'] + matched_cities
    chosen = st.sidebar.selectbox('Suggestions', options_for_select, index=0, key='city_select')
    sel_cities = [] if chosen == '--Choisir une ville--' else [chosen]
else:
    # no query -> don't filter by city
    sel_cities = []

# Date range filter using date_maj: default to year 2025 when possible
date_min = df['date_maj'].min()
date_max = df['date_maj'].max()
if pd.isna(date_min) or pd.isna(date_max):
    st.sidebar.info('Aucune date disponible dans les données pour filtrer.')
    date_range = (None, None)
else:
    # prefer 2025 window if the data contains 2025
    try:
        year2025_start = pd.to_datetime('2025-01-01')
        year2025_end = pd.to_datetime('2025-12-31')
        # compute intersection between available dates and year 2025
        intersect_start = max(date_min, year2025_start)
        intersect_end = min(date_max, year2025_end)
        if (df['date_maj'] >= year2025_start).any() and intersect_start <= intersect_end:
            # use the intersection as the selectable window and default
            min_val = intersect_start.date()
            max_val = intersect_end.date()
            default_start = min_val
            default_end = max_val
        else:
            # fallback to global min/max
            min_val = date_min.date()
            max_val = date_max.date()
            default_start = min_val
            default_end = max_val
    except Exception:
        default_start = date_min.date()
        default_end = date_max.date()
        min_val = date_min.date()
        max_val = date_max.date()

    date_range = st.sidebar.date_input('Période', value=(default_start, default_end),
                                       min_value=min_val, max_value=max_val)


# --- Apply filters ---
mask = pd.Series(True, index=df.index)
if sel_regions:
    mask &= df['region'].fillna('').astype(str).str.upper().isin([r.upper() for r in sel_regions])
if sel_types:
    mask &= df['type_category'].isin(sel_types)
if sel_depts:
    mask &= df['departement'].isin(sel_depts)
if sel_cities:
    mask &= df['city'].isin(sel_cities)
if date_range[0] is not None and date_range[1] is not None:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1])
    mask &= df['date_maj'].between(start, end)

df_filtered = df[mask]


tabs = st.tabs(['Accueil', 'Dashboard', 'Tourisme montagne'])


def show_dashboard(df_filtered):
    st.title('Dashboard - Le tourisme en montagne')

    if df_filtered.empty:
        st.warning('Aucun objet trouvé avec les filtres actuels. Vérifie Type / Département / Ville / Période.')
        with st.expander('Aperçu des données chargées (5 lignes)'):
            st.dataframe(df.head(5))
        return

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Nombre d'objets visibles", int(len(df_filtered)))
    c2.metric('Catégories visibles', int(df_filtered['type_category'].nunique()))
    c3.metric('Départements visibles', int(df_filtered['departement'].nunique() if 'departement' in df_filtered.columns else 0))

    # Time series by month (saisonnalité)
    if df_filtered['date_maj'].notna().sum() > 0:
        ts = df_filtered.dropna(subset=['date_maj']).groupby(pd.Grouper(key='date_maj', freq='ME')).size().reset_index(name='count')
        ts['month'] = ts['date_maj'].dt.to_period('M').astype(str)
        fig = px.line(
            ts,
            x='date_maj',
            y='count',
            labels={'date_maj': 'Mois', 'count': "Nombre d'objets"},
            title="Nombre d'objets par mois"
        )
        fig.update_layout(xaxis_title='Mois', yaxis_title="Nombre d'objets")
        fig.update_traces(mode='lines+markers', hovertemplate="%{y} objets<br>%{x}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Pas de dates valides pour afficher la série temporelle.')

    # Interactive map (pydeck)
    st.subheader('Carte des POI')
    if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
        coords = df_filtered.dropna(subset=['latitude', 'longitude'])
        if coords.shape[0] == 0:
            st.info('Aucune coordonnée disponible pour la carte.')
        else:
            # Select only necessary columns for the map to reduce payload
            map_cols = ['longitude', 'latitude', 'nom', 'type_category', 'departement']
            present = [c for c in map_cols if c in coords.columns]
            coords_small = coords[present].dropna(subset=['latitude', 'longitude'])

            total_points = len(coords_small)
            if total_points == 0:
                st.info('Aucune coordonnée valide après nettoyage pour la carte.')
            else:
                # Map mode: Heatmap / Scatter (scatter uses sampling to keep payload small)
                map_mode = st.sidebar.selectbox(
                    'Mode de la carte',
                    ['Heatmap', 'Scatter'],
                    index=0,
                    help='Heatmap: vue dense; Scatter: points individuels (peut être échantillonné).'
                )

                # Select only necessary columns for the chosen map mode to reduce payload
                if map_mode == 'Scatter':
                    map_cols = ['longitude', 'latitude', 'nom', 'type_category', 'departement_name']
                else:
                    # heatmap only need lat/lon
                    map_cols = ['longitude', 'latitude']

                present = [c for c in map_cols if c in coords.columns]
                coords_small = coords[present].dropna(subset=['latitude', 'longitude'])
                total_points = len(coords_small)
                if total_points == 0:
                    st.info('Aucune coordonnée valide après nettoyage pour la carte.')
                else:
                    if map_mode == 'Scatter':
                        if total_points > MAX_MAP_POINTS:
                            coords_small = coords_small.sample(n=MAX_MAP_POINTS, random_state=42)
                            st.info(f"Affichage limité à {MAX_MAP_POINTS} points sur {total_points} pour performance.")
                        layer = pdk.Layer(
                            'ScatterplotLayer',
                            data=coords_small,
                            get_position='[longitude, latitude]',
                            get_radius=100,
                            get_fill_color='[255, 140, 0, 160]',
                            pickable=True
                        )
                    elif map_mode == 'Heatmap':
                        if total_points > MAX_MAP_POINTS_HEAT:
                            coords_small = coords_small.sample(n=MAX_MAP_POINTS_HEAT, random_state=42)
                            st.info(f"Affichage heatmap limité à {MAX_MAP_POINTS_HEAT} points sur {total_points} pour performance.")
                        layer = pdk.Layer(
                            'HeatmapLayer',
                            data=coords_small,
                            get_position='[longitude, latitude]',
                            aggregation='MEAN',
                            pickable=False,
                        )

                    midpoint = (coords_small['latitude'].mean(), coords_small['longitude'].mean())
        view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=7)
        tooltip = {"text": "{nom}\n{type_category}\n{city}"} if 'nom' in coords_small.columns else None
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
        st.pydeck_chart(r)
    else:
        st.info('Colonnes latitude/longitude absentes pour la carte.')

    # Top cities table
    st.subheader('Top villes (par nombre d\'objets)')
    top_cities = df_filtered['city'].fillna('Inconnu').value_counts().head(10).reset_index()
    top_cities.columns = ['city', 'count']
    st.table(top_cities)

    # Bar by departement
    if 'departement_name' in df_filtered.columns:
        st.subheader('Répartition par département')
        by_dept = df_filtered.groupby('departement_name').size().reset_index(name='count').sort_values('count', ascending=False)
        fig2 = px.bar(by_dept.head(30), x='departement_name', y='count', title='Objets par département')
        st.plotly_chart(fig2, use_container_width=True)

    # Data quality section
    st.subheader('Qualité des données')
    dq_cols = ['nom', 'type_category', 'type_simple', 'city', 'departement', 'latitude', 'longitude', 'date_maj']
    present_dq = [c for c in dq_cols if c in df_filtered.columns]
    if present_dq:
        missing_pct = (df_filtered[present_dq].isna().mean() * 100).round(2).sort_values(ascending=False)
        st.markdown('Missingness (% of filtered rows):')
        st.table(missing_pct.reset_index().rename(columns={'index':'column', 0:'missing_pct'}).rename(columns={0:'missing_pct'}))

        dedup_subset = ['nom']
        if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
            dedup_subset = ['nom', 'latitude', 'longitude']
        dup_count = df_filtered.duplicated(subset=dedup_subset).sum()
        st.write(f'Duplicates (by {dedup_subset}): {int(dup_count)}')
    else:
        st.info('Pas de colonnes standard disponibles pour évaluer la qualité des données.')


# Fill tabs with the appropriate content; filters remain in the sidebar and apply to the sections
with tabs[0]:
    show_intro()

with tabs[1]:
    show_dashboard(df_filtered)

with tabs[2]:
    st.title('Tourisme en montagne')
    story_mountain.show_story_mountain(df_filtered)

