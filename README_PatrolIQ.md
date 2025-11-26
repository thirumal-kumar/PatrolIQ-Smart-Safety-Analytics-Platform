# PatrolIQ — Smart Safety Dashboard

PatrolIQ is a police-style crime analytics dashboard designed to visualize hotspot clusters, temporal trends, PCA/UMAP embeddings, and patrol recommendations using Chicago Crime Data (500k sampled records).

## Features
- Geographic hotspot visualization (KMeans, DBSCAN, Hierarchical)
- Temporal analysis (hour/day/month trends)
- PCA & UMAP dimensionality reduction
- Cluster comparison and insights
- Patrol unit allocation recommendations
- Police-style blue/grey UI
- Fully CPU-compatible pipeline (no GPU required)

## Project Structure
- **data/** — Parquet outputs from pipeline
- **src/** — Ingest, preprocess, features, clustering, dimred
- **streamlit_app.py** — Main dashboard UI
- **run_all.py** — Executes full pipeline

## Running the Dashboard
```bash
python -m streamlit run streamlit_app.py
```

## Running the Full Pipeline
```bash
python run_all.py
```

## Requirements
- Python 3.10+
- pandas, numpy, scikit-learn
- plotly, pydeck, streamlit
- umap-learn
- mlflow

## Author
PatrolIQ Crime Analytics System
