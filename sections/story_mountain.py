import streamlit as st
import pandas as pd
import plotly.express as px

from utils.prep import filter_mountain
from sections.conclusions import show_conclusions


def _month_to_season(m):
    if pd.isna(m):
        return 'Unknow'
    m = int(m)
    if m in (12, 1, 2):
        return 'Winter'
    if m in (3, 4, 5):
        return 'Spring'
    if m in (6, 7, 8):
        return 'Summer'
    return 'Automn'


def show_story_mountain(df: pd.DataFrame):
    st.header('Mountain tourism by season')

    st.markdown(
        """
        This chapter explores the seasonality of tourism offerings in mountain areas.
        We use heuristics to isolate objects related to mountains (name, type, theme, description),
        then compare seasons based on the date of update/publication.

        Questions addressed:
        - Which offerings are mainly winter-based (skiing, mountain huts) vs. summer-based (hiking, mountain peaks)?
        - Where (geographically) are these offerings concentrated according to the season?
        - What types of activities emerge as priorities for a tourist office according to the season?
        """
    )

    # Heuristic mountain filter
    m_df = filter_mountain(df)
    orig_count = len(m_df)
    st.write(f"Events detected as related to mountains : **{orig_count}**")

    if orig_count == 0:
        st.info("No events identified as 'mountain'")
        return

    allowed_regions = {'PAC', 'ARA', 'OCC'}
    m_df = m_df.copy()
    m_df['region_norm'] = m_df.get('region', '').fillna('').astype(str).str.strip().str.upper()
    m_df_filtered = m_df[m_df['region_norm'].isin(allowed_regions)]

    if m_df_filtered.empty and 'departement' in m_df.columns:
        import re

        def _extract_dept_code(x):
            if x is None:
                return ''
            s = str(x).strip()
            m = re.match(r'^(2A|2B|\d{1,3})', s, flags=re.I)
            if not m:
                return ''
            code = m.group(1).upper()
            if code.isdigit() and len(code) == 1:
                code = code.zfill(2)
            return code

        m_df['dept_code'] = m_df['departement'].apply(_extract_dept_code)
        region_to_depts = {
            'PAC': ['04', '05', '06', '13', '83', '84'],
            'ARA': ['01', '03', '07', '15', '26', '38', '42', '43', '63', '69', '73', '74'],
            'OCC': ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82']
        }
        depts_allowed = set(sum([region_to_depts[r] for r in allowed_regions], []))
        m_df_filtered = m_df[m_df['dept_code'].isin(depts_allowed)]

    st.write(f"After filtering by region (PAC/ARA/OCC) : **{len(m_df_filtered)}** (sur {orig_count})")
    if len(m_df_filtered) == 0:
        st.info("No mountainous events found in the PAC/ARA/OCC regions. Try widening the filters or checking the data.")
        return

    m_df = m_df_filtered

    if 'date_maj' in m_df.columns:
        m_df = m_df.copy()
        m_df['date_maj'] = pd.to_datetime(m_df['date_maj'], errors='coerce')
        m_df['month'] = m_df['date_maj'].dt.month
        m_df['season'] = m_df['month'].apply(_month_to_season)
    else:
        m_df = m_df.copy()
        m_df['season'] = 'Inconnue'

    # Distribution saisonnière
    season_order = ['Winter', 'Spring', 'Summer', 'Automn', 'Unknow']
    season_colors = {
        'Winter': '#4DA6FF',       # bleu clair
        'Spring': '#66BB6A',   # vert
        'Summer': '#E53935',         # rouge
        'Automn': '#FFA726',     # orange
        'Unknow': '#9E9E9E'     # gris
    }
    se_counts = (
        m_df['season'].value_counts().reindex(season_order).fillna(0).reset_index()
    )
    se_counts.columns = ['saison', 'count']

    fig_season = px.bar(
        se_counts,
        x='saison',
        y='count',
        color='saison',
        color_discrete_map=season_colors,
        title='Seasonal distribution of events',
        category_orders={"saison": season_order},
        labels={'saison':'Season', 'count':'Number of events'}
    )
    fig_season.update_layout(showlegend=False, yaxis_title="Number of events")
    st.plotly_chart(fig_season, use_container_width=True)

    # Top catégories 
    type_col = 'type_category' if 'type_category' in m_df.columns else ('type' if 'type' in m_df.columns else None)
    if type_col:
        overall = (
            m_df.groupby(type_col).size().reset_index(name='count').sort_values('count', ascending=False)
        )
        overall['pct'] = (overall['count'] / overall['count'].sum() * 100).round(1)

        st.subheader("Statistics by activity category (mountain)")
        top_overall = overall.head(10)
        if not top_overall.empty:
            top_n = int(top_overall.shape[0])
            title_top = f"Top {top_n} categories of mountain activities"
            fig_top = px.bar(
                top_overall,
                x='count',
                y=type_col,
                orientation='h',
                title=title_top,
                labels={'count':"Number of events", type_col:'Catégorie'}
            )
            fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
           
        # show small table with counts + pct
        st.markdown('**Overall distribution by category**')
        st.table(overall.head(20).rename(columns={type_col:'Category', 'count':'Number', 'pct':'Part (%)'}).reset_index(drop=True))

    # Top catégories par saison (top 5)
    if type_col and m_df['season'].nunique() > 1:
        grp = m_df.groupby(['season', type_col]).size().reset_index(name='count')
        top_season = grp.sort_values(['season', 'count'], ascending=[True, False]).groupby('season').head(5)
        if not top_season.empty:
            season_totals = m_df.groupby('season').size().rename('season_total')
            top_season = top_season.merge(season_totals, on='season')
            top_season['pct_within_season'] = (top_season['count'] / top_season['season_total'] * 100).round(1)

            fig_top_season = px.bar(
                top_season,
                x='count',
                y=type_col,
                color='season',
                color_discrete_map=season_colors,
                facet_col='season',
                orientation='h',
                title='Top categories by season (top 5)',
                labels={'count':"Number of events", type_col:'Category', 'season':'Season'}
            )
            fig_top_season.update_layout(showlegend=False)
            st.plotly_chart(fig_top_season, use_container_width=True)

    # Carte des POI montagneux colorée par saison 
    if {'latitude', 'longitude'}.issubset(m_df.columns) and m_df[['latitude', 'longitude']].dropna().shape[0] > 0:
        st.subheader('Map of mountain POIs (by season)')
        m_map = m_df.dropna(subset=['latitude', 'longitude']).copy()
        try:
            fig_map = px.scatter_mapbox(
                m_map,
                lat='latitude',
                lon='longitude',
                hover_name='nom' if 'nom' in m_map.columns else None,
                    hover_data={'type':True, 'city':True, 'date_maj':True},
                    color='season',
                    color_discrete_map=season_colors,
                zoom=6,
                height=600,
            )
            fig_map.update_layout(mapbox_style='open-street-map', margin={"r":0,"t":0,"l":0,"b":0}, legend_title_text='Saison')
            fig_map.update_traces(marker={'size':8}, selector=dict(mode='markers'))
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception:
            st.map(m_map[['latitude', 'longitude']].rename(columns={'latitude':'lat','longitude':'lon'}))
    else:
        st.info("Not enough geographic coordinates to display the map of mountain POIs.")

    st.markdown('''
    Observations:
    - POIs related to skiing and mountain huts mainly appear in winter; routes and peaks appear in summer.
    - Seasonal mapping helps prioritize seasonal investments (e.g., trail maintenance vs. hiking signage).

    This dataset tells us about the tourist offers that have been posted. It does not tell us about the flow of tourism.
    ''')
    st.markdown('---')
    show_conclusions(m_df)
