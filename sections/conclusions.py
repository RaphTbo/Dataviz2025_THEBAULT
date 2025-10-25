
import streamlit as st

def show_conclusions(df):
    st.header('Conclusions & prochaines étapes')
    # Single extended paragraph summarizing findings, limits and recommendations
    st.markdown(
        (
            "Ce travail montre que, dans les régions Provence‑Alpes‑Côte d'Azur (PAC), Auvergne‑Rhône‑Alpes (ARA) et Occitanie (OCC), "
            "les offres liées à la montagne se concentrent principalement autour des activités de randonnée, des refuges et des structures d'hébergement saisonnier, avec une saisonnalité nette entre les mois d'hiver (ski, refuges) et l'été (itinéraires, sommets). "
            "Toutefois, ces conclusions doivent être interprétées avec prudence : les données DATAtourisme fournissent des informations descriptives et locales mais manquent souvent de métriques de fréquentation et présentent une couverture hétérogène selon les offices et diffuseurs, ce qui peut biaiser la représentativité spatiale et thématique. "
            "En pratique, pour prioriser des actions opérationnelles, il est recommandé d'enrichir ce jeu de données par des indicateurs de fréquentation (capacité d'hébergement remplie, comptages, réservations) et, si possible, de croiser avec des données météorologiques et d'accessibilité routière pour anticiper les besoins saisonniers. "
            "Enfin, à court terme, les offices de tourisme pourraient utiliser ces résultats pour concentrer les efforts de signalétique et d'entretien (pistes, sentiers) selon la saisonnalité identifiée, et à moyen terme lancer des enquêtes ciblées auprès des prestataires pour combler les manques d'information et affiner les priorités d'investissement."
        )
    )
