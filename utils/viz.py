
import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px

def render_kpis(df):
    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("Nombre d'objets visibles", int(len(df)))
    c2.metric('Types distincts visibles', int(df['type'].nunique() if 'type' in df.columns else 0))
    if 'altitude' in df.columns:
        try:
            avg_alt = int(df['altitude'].dropna().mean())
        except:
            avg_alt = 0
        c3.metric('Altitude moyenne (m)', avg_alt)
    st.caption("KPI calculés sur l'ensemble filtré passé à cette fonction.")

def render_time_series(df):
    st.subheader('Saisonnalité (événements / activités)')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        ts = df.groupby(df['date'].dt.to_period('M')).size().reset_index(name='count')
        ts['date'] = ts['date'].dt.to_timestamp()
        fig = px.line(ts, x='date', y='count', labels={'date':'Mois','count':'Nombre d\'objets'}, title='Nombre d\'objets par mois')
        fig.update_layout(xaxis_title='Mois', yaxis_title="Nombre d'objets")
        fig.update_traces(mode='lines+markers', hovertemplate='%{y} objets<br>%{x}')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Aucune colonne date trouvée dans le jeu de données.')

def render_bar_by_region(df):
    st.subheader('Répartition par département / région')
    grp = df.groupby('region').size().reset_index(name='count').sort_values('count', ascending=False)
    fig = px.bar(grp, x='region', y='count', title='Objets touristiques par région')
    st.plotly_chart(fig, use_container_width=True)

def render_map(df):
    st.subheader('Carte des POI (aperçu)')
    if df[['latitude', 'longitude']].dropna().shape[0] == 0:
        st.info('Aucune coordonnée géographique disponible dans le dataset utilisé.')
        return
    midpoint = (df['latitude'].mean(), df['longitude'].mean())
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=df.dropna(subset=['latitude','longitude']),
        get_position='[longitude, latitude]',
        get_radius=100,
        get_fill_color='[200, 30, 0, 160]',
        pickable=True
    )
    view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=7, pitch=0)
    # Tooltip: show name and type on separate lines when available
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={ "text": "{nom}\n{type}" })
    st.pydeck_chart(r)
