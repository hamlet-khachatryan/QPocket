"""Cross-docking results for the explorer."""
import pandas as pd


def rmsd_heatmap(pairs: pd.DataFrame, threshold: float):
    """Top-1 RMSD of every active in every representative.

    Args:
        pairs: crossdock_pairs.csv.
        threshold: Success threshold, Å, shown in the title.

    Returns:
        A plotly figure.
    """
    import plotly.express as px

    grid = pairs.pivot_table(index="representative", columns="active_id", values="top1_rmsd", aggfunc="min")
    fig = px.imshow(grid, color_continuous_scale="RdBu_r", zmin=0, zmax=2 * threshold, aspect="auto",
                    labels=dict(color="top-1 RMSD (Å)"))
    fig.update_layout(title=f"Top-1 pose RMSD to the crystal pose · success below {threshold} Å")
    return fig
