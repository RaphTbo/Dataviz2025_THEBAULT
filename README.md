
# Data Storytelling Dashboard — DATAtourisme (Montagne)

**Projet**: Dashboard Streamlit axé sur le tourisme de montagne à partir des exports DATAtourisme (data.gouv.fr).

## Contenu du dépôt
- `app.py` — Entrée principale Streamlit
- `sections/` — Modules de pages : intro, overview, deep_dives, conclusions
- `utils/` — IO, préparation et fonctions de visualisation
- `data/sample_mountain.csv` — Petit jeu d'exemple (extrait) pour tests rapides
- `download_data.py` — Script pour télécharger les CSV officiels (à lancer localement)
- `requirements.txt` — Dépendances Python
- `README.md` — Ce fichier

## Objectif & Storyline
Un dashboard destiné aux acteurs du tourisme (OT, collectivités, office de guides) pour :
- Explorer l'offre touristique en zone de montagne (POI, activités, itinéraires, hébergements)
- Identifier zones sous‑dotées, saisonnalité et opportunités d'offre
- Faciliter la planification (proximité, itinéraires) et la communication

## Quickstart (local)
1. Cloner / décompresser le projet.
2. Installer un environnement virtuel et installer les dépendances :
   ```bash
   # DATAtourisme — Streamlit dashboard

   Student Project — Streamlit Dashboard

   This repository contains a Streamlit dashboard that explores DATAtourisme open data. The app loads multiple DATAtourisme CSV files, normalizes and cleans them, and presents an interactive data storytelling dashboard focused on activity type, location and seasonality.

   ## Features

   - Robust CSV loading (UTF-8 / latin-1 fallback).
   - Preprocessing: column normalization, `type_simple`, city/postal extraction, department inference.
   - Parquet cache: cleaned combined dataset is written to `data/cleaned.parquet` for faster subsequent runs.
   - Sidebar controls: Type, Région, Département, Ville, Date range, Map mode.
   - Map modes: Heatmap, Hexagon, Scatter (scatter is sampled to keep payload small).
   - Data quality panel: missingness and duplicate checks for the filtered dataset.

   ## How to run (PowerShell)

   1. Create a virtual environment and install dependencies (recommended):

   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   2. Run the Streamlit app:

   ```powershell
   streamlit run app.py
   ```

   ## Notes on caching

   - On first run the app may take longer since it reads all CSVs and writes `data/cleaned.parquet`.
   - To force a rebuild from CSVs, open the sidebar and click **Rebuild cache (rebuild cleaned.parquet)**.

   ## Deliverables checklist

   - Streamlit app (`app.py` + `sections/`, `utils/`)
   - Dataset (CSV files are included in `data/`)
   - README (this file)
   - Short demo video (include separately when packaging)

   ## Course alignment

   - Sidebar controls, KPIs, time series, bar chart and map are present.
   - Data quality checks included.
   - Performance: caching and map controls (heatmap/hexagon/scatter with sampling) reduce payload.

   If you want, I can:
   - Add a short `Makefile` or PowerShell script to automate setup.
   - Add unit tests for `utils/io.py` and `utils/prep.py`.
   - Implement server-side clustering (centroids) for scatter points instead of sampling.

