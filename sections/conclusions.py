
import streamlit as st

def show_conclusions(df):
    st.header('Conclusions')
    st.markdown(
        (
            "This study shows that in the Provence-Alpes-Côte d'Azur (PAC), Auvergne-Rhône-Alpes (ARA), and Occitanie (OCC) regions,"
            "mountain-related offerings are mainly concentrated around hiking activities, mountain huts, and seasonal accommodation facilities, with a clear seasonal pattern between the winter months (skiing, mountain huts) and summer (trails, peaks). "
            "However, these conclusions should be interpreted with caution: DATAtourisme data provides descriptive and local information but often lacks attendance metrics and has uneven coverage depending on the offices and distributors, which may bias spatial and thematic representativeness. "
            "In practice, in order to prioritize operational actions, it is recommended to enrich this dataset with attendance indicators (accommodation capacity filled, counts, reservations) and, if possible, to cross-reference it with meteorological and road accessibility data to anticipate seasonal needs. "
            "Finally, in the short term, tourist offices could use these results to focus signage and maintenance efforts (trails, paths) according to identified seasonality, and in the medium term, launch targeted surveys of service providers to fill information gaps and refine investment priorities."
        )
    )
