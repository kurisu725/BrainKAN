"""The fixed-grid KAN constructor used in the EN experiments."""
from legacy_kan import KAN

def build_model(kind, n, cfg):
    if kind != "kan_legacy":
        raise ValueError("This entry point reproduces the fixed-grid EN/KAN workflow")
    return KAN([n*n] + cfg["hidden"] + [2], grid_size=cfg.get("grid_size", 5),
               spline_order=cfg.get("spline_order", 3))
