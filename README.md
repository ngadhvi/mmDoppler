# mmDoppler

Toolkit for processing and visualizing mmWave radar data, specifically for analyzing and modeling Micro-Doppler (e.g., typing) human activities. The dataset also contains Macro-Doppler (e.g., walking, jumping) human activities but that is not the focus of this project.

## Key Features

- **Pipeline**: End-to-end processing from raw logs to Polars DataFrames (Parquet).
- **Anomaly Detection**: Isolation Forest implementation to filter spatiotemporal noise.
- **Visualization**: Interactive tools for Range-Doppler heatmaps and activity signature grids.

## Project Structure

```bash
mmDoppler/
├── data/                   # Processed datasets
├── helper/                 # Utilities for analysis, anomalies, and preprocessing
└── notebooks/              # Interactive Analysis (01_preprocess, 02_analyze)
```

## Getting Started

This project uses `uv` for dependency management.

```bash
# Sync dependencies
uv sync

# Start Jupyter
uv run jupyter notebook
```

**Workflow:**
1.  Run `01_preprocess.ipynb` to clean data and detect outliers.
2.  Run `02_analyze.ipynb` to visualize heatmaps and check class balance.

## Data Source & Citation

The data collection methodology and original dataset are sourced from:
[https://github.com/arghasen10/mmDoppler/tree/main](https://github.com/arghasen10/mmDoppler/tree/main)
