import streamlit as st
import pandas as pd
import plotly.express as px

from utils.prep import filter_mountain
from sections.conclusions import show_conclusions


def _month_to_season(m):
    if pd.isna(m):
        return 'Inconnue'
    m = int(m)
    if m in (12, 1, 2):
        return 'Hiver'
    if m in (3, 4, 5):
        return 'Printemps'
    if m in (6, 7, 8):
        return 'Été'
    return 'Automne'


def show_story_mountain(df: pd.DataFrame):
    """Storytelling section: tourisme en montagne par saison.

    - Filtre heuristique pour POI/produits liés à la montagne
    - Construction d'une colonne "season" à partir de la date
    - Visualisations: distribution saisonnière, top types, top par saison, carte
    """
    st.header('Story — Tourisme en montagne par saison')

    st.markdown(
        """
        Ce chapitre explore la saisonnalité des offres touristiques en montagne.
        Nous utilisons une heuristique pour isoler les objets liés à la montagne (nom, type, thème, description),
        puis comparons les saisons sur la base de la date de mise à jour / publication.

        Questions traitées :
        - Quelles offres sont principalement hivernales (ski, refuges) vs estivales (randonnée, sommets) ?
        - Où (géographiquement) se concentrent ces offres selon la saison ?
        - Quels types d'activités émergent comme prioritaires pour un office de tourisme selon la saison ?
        """
    )

    # Heuristic mountain filter
    m_df = filter_mountain(df)
    orig_count = len(m_df)
    st.write(f"Objets détectés comme liés à la montagne : **{orig_count}**")

    if orig_count == 0:
        st.info("Aucun objet identifié comme 'montagne' — essayez d'élargir vos filtres ou vérifiez les données sources.")
        return

    # --- Apply region filter globally for the whole storytelling section ---
    allowed_regions = {'PAC', 'ARA', 'OCC'}
    m_df = m_df.copy()
    # normalize region column when present
    m_df['region_norm'] = m_df.get('region', '').fillna('').astype(str).str.strip().str.upper()
    m_df_filtered = m_df[m_df['region_norm'].isin(allowed_regions)]

    # fallback: try to infer region from department codes when region not present
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

    st.write(f"Après filtrage par région (PAC/ARA/OCC) : **{len(m_df_filtered)}** (sur {orig_count})")
    if len(m_df_filtered) == 0:
        st.info("Aucun objet montagneux trouvé dans les régions PAC/ARA/OCC. Essayez d'élargir les filtres ou vérifiez les données.")
        return

    # use the region-filtered DataFrame for all subsequent charts in this section
    m_df = m_df_filtered

    # Ensure date_maj exists and is datetime
    if 'date_maj' in m_df.columns:
        m_df = m_df.copy()
        m_df['date_maj'] = pd.to_datetime(m_df['date_maj'], errors='coerce')
        m_df['month'] = m_df['date_maj'].dt.month
        m_df['season'] = m_df['month'].apply(_month_to_season)
    else:
        m_df = m_df.copy()
        m_df['season'] = 'Inconnue'

    # Distribution saisonnière
    season_order = ['Hiver', 'Printemps', 'Été', 'Automne', 'Inconnue']
    # Palette couleur fixe pour les saisons — garantit la même couleur partout (Été = rouge)
    season_colors = {
        'Hiver': '#4DA6FF',       # bleu clair
        'Printemps': '#66BB6A',   # vert
        'Été': '#E53935',         # rouge (toujours rouge)
        'Automne': '#FFA726',     # orange
        'Inconnue': '#9E9E9E'     # gris
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
        title='Distribution saisonnière des objets montagne',
        category_orders={"saison": season_order},
        labels={'saison':'Saison', 'count':'Nombre d\'objets'}
    )
    fig_season.update_layout(showlegend=False, yaxis_title="Nombre d'objets")
    st.plotly_chart(fig_season, use_container_width=True)

    # Top catégories (utilise la colonne `type_category` si disponible, sinon `type`)
    type_col = 'type_category' if 'type_category' in m_df.columns else ('type' if 'type' in m_df.columns else None)
    if type_col:
        # counts and percentages overall
        overall = (
            m_df.groupby(type_col).size().reset_index(name='count').sort_values('count', ascending=False)
        )
        overall['pct'] = (overall['count'] / overall['count'].sum() * 100).round(1)

        st.subheader("Statistiques par catégorie d'activité (montagne)")
        # show top 10 as chart
        top_overall = overall.head(10)
        if not top_overall.empty:
            top_n = int(top_overall.shape[0])
            title_top = f"Top {top_n} des catégories d'activités en montagne"
            fig_top = px.bar(
                top_overall,
                x='count',
                y=type_col,
                orientation='h',
                title=title_top,
                labels={'count':"Nombre d'objets", type_col:'Catégorie'}
            )
            fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
           
        # show small table with counts + pct
        st.markdown('**Répartition globale par catégorie**')
        st.table(overall.head(20).rename(columns={type_col:'Catégorie', 'count':'Nombre', 'pct':'Part (%)'}).reset_index(drop=True))

    # Top catégories par saison (top 5)
    if type_col and m_df['season'].nunique() > 1:
        grp = m_df.groupby(['season', type_col]).size().reset_index(name='count')
        # keep top 5 categories per season
        top_season = grp.sort_values(['season', 'count'], ascending=[True, False]).groupby('season').head(5)
        if not top_season.empty:
            # Add percent within season
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
                title='Top catégories par saison (top 5)',
                labels={'count':"Nombre d'objets", type_col:'Catégorie', 'season':'Saison'}
            )
            fig_top_season.update_layout(showlegend=False)
            st.plotly_chart(fig_top_season, use_container_width=True)

    # Carte des POI montagneux colorée par saison (plotly mapbox open-street-map)
    if {'latitude', 'longitude'}.issubset(m_df.columns) and m_df[['latitude', 'longitude']].dropna().shape[0] > 0:
        st.subheader('Carte des POI montagneux (par saison)')
        # m_df has already been filtered to PAC/ARA/OCC above; just drop rows without coords
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
            # Fallback to simple st.map if plotly map fails
            st.map(m_map[['latitude', 'longitude']].rename(columns={'latitude':'lat','longitude':'lon'}))
    else:
        st.info("Pas assez de coordonnées géographiques pour afficher la carte des POI montagneux.")

    st.markdown('''
    Observations possibles :
    - Les POI liés au ski et aux refuges apparaissent principalement en hiver ; les itinéraires et sommets en été.
    - La cartographie par saison aide à prioriser les investissements saisonniers (ex. entretien des pistes vs signalétique randonnée).

    Suggestions pour approfondir : enrichir les données avec la fréquentation (si disponible), la météo saisonnière ou les avis pour prioriser les actions.
    ''')
    # Append the global conclusions at the end of the storytelling section
    st.markdown('---')
    # Show conclusions based on the mountain POI subset (and region-filtered) so conclusions
    # reflect 'tourisme à la montagne' dans PAC/ARA/OCC as requested by the user.
    show_conclusions(m_df)
