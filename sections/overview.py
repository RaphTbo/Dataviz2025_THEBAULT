
import streamlit as st
from utils.viz import render_kpis, render_bar_by_region, render_map, render_time_series

def show_overview(df):
    st.header('Overview')
    render_kpis(df)
    render_bar_by_region(df)
    render_map(df)
    render_time_series(df)
