
import streamlit as st

def show_intro():
    st.title('Tourisme de montagne — DATAtourisme')
    st.markdown('''
**Problématique** : Où se concentrent les offres touristiques en montagne (POI, activités, itinéraires), quelles zones sont sous‑dotées, et quelle est la saisonnalité des événements ?

**Audience** : Offices de tourisme, collectivités locales, opérateurs d'activités outdoor.

**Sources** : DATAtourisme (data.gouv.fr)
''')
