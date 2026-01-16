
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_heatmap(df: pl.DataFrame, frame_index: int = 0):
    """
    Visualizes the Range-Doppler heatmap for a specific frame.
    'doppz' contains the heatmap data.
    """
    row = df.row(frame_index, named=True)
    heatmap_data = row['doppz']
    
    heatmap_arr = np.array(heatmap_data)
    
    # Check shape and reshape if flat
    # 128 Doppler bins since Doppler indices are from -64 to 63
    DOPPLER_BINS = 128
    
    if heatmap_arr.ndim == 1:
        total_points = heatmap_arr.size
        # Range bins = Total / Doppler
        range_bins = total_points // DOPPLER_BINS
        
        try:
            heatmap_arr = heatmap_arr.reshape(range_bins, DOPPLER_BINS)
        except ValueError:
             print(f"Error: Could not reshape array of size {total_points} with {DOPPLER_BINS} Doppler bins.")
             return None

    # Plot using Plotly
    fig = px.imshow(
        heatmap_arr,
        labels=dict(x="Doppler Bin", y="Range Bin", color="Intensity"),
        title=f"Range-Doppler Heatmap (Frame {frame_index}, Activity: {row.get('activity', 'Unknown')})",
        origin='lower',  # Standard radar plotting convention
        aspect='auto'
    )
    return fig

def visualize_activity_grid_inline(df: pl.DataFrame, analyzer, n_frames: int = 5):
    """
    Visualizes the Range-Doppler heatmap in a 3x3 grid.
    Aggregates (averages) the first `n_frames` for each activity to show a cleaner signature.
    """
    # 1. Get first N frames for each activity
    # 2. Sort by activity to maintain grid order
    samples_df = (
        df
        .group_by("activity")
        .head(n_frames)
        .sort("activity")
    )
    
    unique_activities = samples_df["activity"].unique(maintain_order=True).head(9).to_list()
    activity_map = analyzer._get_activity_map()
    
    fig = make_subplots(
        rows=3, cols=3, 
        subplot_titles=[activity_map.get(act, f"ID {act}") for act in unique_activities],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    DOPPLER_BINS = 128

    for i, act_id in enumerate(unique_activities):
        row_idx = (i // 3) + 1
        col_idx = (i % 3) + 1
        
        # Get all N frames for this specific activity
        act_frames = samples_df.filter(pl.col("activity") == act_id)
        
        # Accumulate heatmaps
        accumulated_heatmap = None
        count = 0
        
        for frame_row in act_frames.iter_rows(named=True):
            heatmap_data = np.array(frame_row['doppz'])
            
            # Reshape if necessary
            if heatmap_data.ndim == 1:
                try:
                    range_bins = heatmap_data.size // DOPPLER_BINS
                    heatmap_data = heatmap_data.reshape(range_bins, DOPPLER_BINS)
                except ValueError:
                    continue # Skip malformed frames

            if accumulated_heatmap is None:
                accumulated_heatmap = heatmap_data.astype(np.float64)
            else:
                if accumulated_heatmap.shape == heatmap_data.shape:
                    accumulated_heatmap += heatmap_data
            
            count += 1
            
        if accumulated_heatmap is not None and count > 0:
            avg_heatmap = accumulated_heatmap / count
            
            fig.add_trace(
                go.Heatmap(z=avg_heatmap, coloraxis="coloraxis"),
                row=row_idx, col=col_idx
            )

    fig.update_layout(
        title_text=f"Average Activity Signatures (First {n_frames} frames)",
        height=900,
        width=900,
        coloraxis=dict(colorscale='Viridis'),
        showlegend=False
    )
    return fig


class ClassDistributionAnalyzer:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        
    def _get_activity_map(self) -> dict:
        """
        Returns a dictionary mapping Activity ID to Canonical Name.
        Derived from scripts/process_data.py
        """
        return {
            0: 'Clapping',
            1: 'Jumping', 
            2: 'Lunges',
            3: 'Walking',
            4: 'Squats',
            5: 'Waving',
            6: 'Folding Clothes',
            7: 'Changing Clothes',
            8: 'Vacuum Cleaning', 
            9: 'Running',
            10: 'Phone Typing',
            11: 'Laptop Typing',
            12: 'Sitting', 
            13: 'Eating',
            14: 'Phone Talking', 
            15: 'Playing Guitar',
            16: 'Brushing Teeth',
            17: 'Combing Hair',
            18: 'Drinking Water'
        }

    def compute_distribution(self) -> pl.DataFrame:
        """
        Calculates the count and percentage of each activity class.
        """
        activity_map = self._get_activity_map()

        counts = self.df.group_by("activity").count()
        
        total_frames = self.df.height
        
        dist_df = (
            counts
            .with_columns(
                pl.col("activity").map_elements(lambda x: activity_map.get(x, f"Unknown ({x})"), return_dtype=pl.Utf8).alias("Activity Name"),
                (pl.col("count") / total_frames * 100).alias("Percentage")
            )
            .sort("count", descending=True)
        )
        
        return dist_df

    def plot_distribution(self, distribution_df: pl.DataFrame):
        """
        Plots a bar chart of the activity distribution.
        """
        fig = px.bar(
            distribution_df,
            x="Activity Name",
            y="count",
            text="Percentage",
            color="Activity Name",
            title="Activity Class Distribution",
            labels={"count": "Number of Frames"},
            hover_data=["Percentage"]
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False)
        
        return fig

    def compute_duration_stats(self, fps: float = 2.0) -> pl.DataFrame:
        """
        Calculates duration statistics for each activity based on continuous segments.
        Handles time gaps to distinguish separate events even if activity ID matches.
        
        Args:
            fps (float): Frames per second (sampling rate). Default is 2.0 used for Micro-Activities.
            
        Returns:
            pl.DataFrame: DataFrame containing activity name and duration statistics (median, p10, min, max).
        """
        activity_map = self._get_activity_map()
        
        # Expected max gap between frames for a continuous segment
        max_gap_sec = (1.0 / fps) * 2.0
        
        if "datetime" not in self.df.columns:
            # Fallback if no datetime column: just use activity changes
            print("[Warning] No 'datetime' column found. Falling back to activity-change segmentation only.")
            segments = (
                self.df
                .with_columns(
                    (pl.col("activity") != pl.col("activity").shift(1).fill_null(pl.col("activity").first()))
                    .cum_sum()
                    .alias("segment_id")
                )
            )
        else:
            segments = (
                self.df
                # Do NOT sort by datetime to preserve file/user concatenation order
                .with_columns([
                    # Check activity change
                    (pl.col("activity") != pl.col("activity").shift(1)).fill_null(False).alias("act_changed"),
                    
                    # Check time gap (current - prev)
                    pl.col("datetime").diff().dt.total_seconds().fill_null(0).alias("time_diff")
                ])
                .with_columns(
                    # New segment trigger if:
                    # 1. Activity changed or,
                    # 2. Huge time gap (gap > 1s)
                    # 3. Time went backwards (diff < 0), implying new file/user
                    ((pl.col("act_changed")) | (pl.col("time_diff") > max_gap_sec) | (pl.col("time_diff") < 0))
                    .cum_sum()
                    .alias("segment_id")
                )
            )
        
        # Group by segment ID to get the length (number of frames)
        segment_durations = (
            segments
            .group_by(["segment_id", "activity"])
            .agg(
                (pl.count().cast(pl.Float64) / fps).alias("duration_sec")
            )
        )

        # Compute statistics per activity
        stats_df = (
            segment_durations
            .group_by("activity")
            .agg([
                pl.count().alias("num_segments"),
                pl.col("duration_sec").median().alias("median_duration"),
                pl.col("duration_sec").quantile(0.10).alias("p10_duration"),
                pl.col("duration_sec").quantile(0.90).alias("p90_duration"),
                pl.col("duration_sec").min().alias("min_duration"),
                pl.col("duration_sec").max().alias("max_duration"),
            ])
            .with_columns(
                pl.col("activity").map_elements(lambda x: activity_map.get(x, f"Unknown ({x})"), return_dtype=pl.Utf8).alias("Activity Name")
            )
            .sort("median_duration", descending=True)
            .select([
                "Activity Name", 
                "num_segments", 
                "median_duration", 
                "p10_duration", 
                "p90_duration", 
                "min_duration", 
                "max_duration"
            ])
        )
        
        return stats_df
