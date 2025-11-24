"""
Metadata loading, joins, and dataset-level EDA/visualization.
"""

from .visualization import (
    plot_category_histograms,
    crosstab_heatmap
)
from .loading import (
    load_metadata, 
    load_datasets
)
from .joins import inner_join
