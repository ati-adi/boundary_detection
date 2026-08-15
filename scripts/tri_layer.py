#!/usr/bin/env python3
"""Overlay: RGB scene + GT polygons (red) + predictions (cyan) -> PNG."""
import sys
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

tif, gt_path, pred_path, out = sys.argv[1:5]

with rasterio.open(tif) as src:
    scale = max(src.height, src.width) / 2500
    img = src.read(
        out_shape=(src.count, int(src.height / scale), int(src.width / scale)),
        resampling=Resampling.bilinear,
    )
    b, crs = src.bounds, src.crs

rgb = img[:3].astype("float32").transpose(1, 2, 0)
valid = rgb[rgb > 0]
if valid.size:
    lo, hi = np.percentile(valid, (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)

gt = gpd.read_file(gt_path).to_crs(crs)
pred = gpd.read_file(pred_path).to_crs(crs)

fig, ax = plt.subplots(figsize=(14, 14))
ax.imshow(rgb, extent=[b.left, b.right, b.bottom, b.top])
gt.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1.2, label="Ground Truth")
pred.plot(ax=ax, facecolor="none", edgecolor="cyan", linewidth=0.8, label="Prediction")
ax.legend(loc="lower right", fontsize=12)
ax.set_title(out.split("/")[-1].replace(".png", ""))
ax.set_axis_off()
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)
