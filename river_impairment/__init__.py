from river_impairment.model import MLPImpairment
from river_impairment.data import load_and_preprocess, spatiotemporal_split
from river_impairment.trainer import train_mlp, optimize_mlp
from river_impairment.metrics import compute_classification_metrics
from river_impairment.explainer import shap_summary, shap_waterfall

__version__ = "1.0.0"
