import polars as pl
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict

class RadarOutlierDetector:
    def __init__(self, df: pl.DataFrame):
        """
        Initialize the detector with a Polars DataFrame containing radar point cloud data.
        Expected list columns: 'x_coord', 'y_coord', 'peakVal', 'dopplerIdx'.
        """
        self.df = df
        self.model = None
        self.results_df = None
        self.feature_cols = []

    def _extract_frame_features(self) -> pl.DataFrame:
        """
        Aggregates point-cloud features into frame-level signals for anomaly detection.
        Features created:
        - numDetectedObj: Number of points
        - mean_intensity: Average signal strength (peakVal)
        - spatial_area: Approximation of spatial spread (x_range * y_range)
        - movement_energy: Average absolute doppler velocity
        """
        # Ensure we handle possible nulls/empty lists by filling with 0
        features = self.df.select([
            pl.col("datetime"),
            pl.col("numDetectedObj"),
            
            # Signal Strength Stats
            pl.col("peakVal").list.mean().fill_null(0).alias("mean_intensity"),
            pl.col("peakVal").list.max().fill_null(0).alias("max_intensity"),
            
            # Spatial Stats
            ((pl.col("x_coord").list.max() - pl.col("x_coord").list.min()).fill_null(0) * 
             (pl.col("y_coord").list.max() - pl.col("y_coord").list.min()).fill_null(0)).alias("spatial_area"),
             
            # Doppler/Motion Stats (using simple mean for now, can be sophisticated later)
            # Note: Using eval for absolute calculation if negative doppler exists
            pl.col("dopplerIdx").list.eval(pl.element().abs()).list.mean().fill_null(0).alias("avg_motion")
        ])
        
        # Return only numeric feature columns for outlier detection
        return features

    def detect_frame_anomalies(self, contamination: float | str = "auto", random_state: int = 42) -> pl.DataFrame:
        """
        Trains Isolation Forest on frame statistics to identify anomalous timeframes.
        
        Args:
            contamination (float): Expected proportion of outliers in the dataset.
            random_state (int): Seed for reproducibility.
            
        Returns:
            pl.DataFrame: Original dataframe with 'anomaly_score' and 'is_anomaly' columns.
        """
        features_df = self._extract_frame_features()
        
        # Select numeric columns for training (exclude datetime)
        train_cols = [c for c in features_df.columns if c != "datetime"]
        self.feature_cols = train_cols
        
        X = features_df.select(train_cols).to_numpy()
        
        print(f"[Info] Training Isolation Forest (n={len(X)})")
        self.model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
        
        # Fit and predict
        # decision_function: lower is more anomalous (negative)
        # predict: -1 is anomaly, 1 is normal
        predictions = self.model.fit_predict(X)
        scores = self.model.decision_function(X)
        
        self.results_df = self.df.with_columns([
            pl.Series(name="anomaly_score", values=scores),
            pl.Series(name="is_anomaly", values=(predictions == -1))
        ])
        
        n_anomalies = (predictions == -1).sum()
        print(f"[Info] Detection Complete")
        print(f"[Info] Identified {n_anomalies} anomalous frames ({n_anomalies/len(X)*100:.2f}%)")
        
        return self.results_df

    def show_distribution_score(self):
        """
        Plots the distribution of anomaly scores.
        """
        if self.results_df is None:
            raise ValueError("Run detect_frame_anomalies() first.")
            
        scores = self.results_df["anomaly_score"].to_numpy()
        labels = self.results_df["is_anomaly"].to_numpy()
        
        fig = px.histogram(
            x=scores, 
            color=labels.astype(str),
            nbins=50,
            title="Distribution of Isolation Forest Anomaly Scores",
            labels={"x": "Anomaly Score (Lower = More Anomalous)", "color": "Is Anomaly"},
            color_discrete_map={"True": "red", "False": "blue"}
        )
        
        fig.add_vline(x=self.model.offset_, line_dash="dash", line_color="black", annotation_text="Threshold")
        return fig

    def show_anomalies_scatter(self, x_axis: str = "mean_intensity", y_axis: str = "numDetectedObj"):
        """
        Scatter plot of Anomalies vs Normal frames.
        """
        if self.results_df is None:
            raise ValueError("Run detect_frame_anomalies() first.")

        features_with_labels = self._extract_frame_features().with_columns([
            self.results_df["is_anomaly"],
            self.results_df["anomaly_score"]
        ])
        
        fig = px.scatter(
            features_with_labels.to_pandas(), 
            x=x_axis, 
            y=y_axis, 
            color="is_anomaly",
            hover_data=["datetime", "anomaly_score"],
            title=f"Anomaly Clustering: {y_axis} vs {x_axis}",
            color_discrete_map={True: "red", False: "blue"},
            opacity=0.7
        )
        return fig
