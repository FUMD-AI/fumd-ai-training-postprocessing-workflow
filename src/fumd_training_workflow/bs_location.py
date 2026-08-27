"""
=============================================================================
FUMD-AI Training Workflow -- shared library: real-measurement BS location estimation
=============================================================================
Used by: notebooks/step_5_estimate_bs_locations.ipynb

Author(s):
  - Cristina Bernad (ORCID: 0000-0001-9537-415X)
  - Sonja Filiposka <sonja.filiposka@finki.ukim.mk> (ORCID: 0000-0003-0034-2855)
  - Katja Gilly (ORCID: 0000-0002-8985-0639)

Funding: This work has been funded by the FUMD-AI project, an EOSC GRAVITY -
Inter Project with Grant Number 25-EOSC-GRV-INTER-013.

SPDX-License-Identifier: MIT
-----------------------------------------------------------------------------

Estimates real-world physical base-station site coordinates from real
measured RSRP samples (cell id + lat/lon + signal strength readings taken
while driving/walking around a coverage area) -- NOT simulation output.
This is the part of the workflow that anchors the simulation's abstract
`servingCell` ids to actual locations, so `migration_map.py` can plot
predicted/actual handover destinations on a real street map.

Refactored from `estimacion_antenas.ipynb`'s working logic (its exploratory
cells -- a pyrosm Helsinki demo, a folium custom-icon experiment -- are not
carried over, they never fed into the notebook's own final output).

Method: for each cell id, take the top-quantile (strongest) RSRP readings
and use their median position as that cell's estimated location; then
cluster estimated per-cell locations that sit within a few meters of each
other (DBSCAN, haversine metric) into physical "sites", since one physical
site commonly serves more than one cell id (sector/carrier).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_M = 6_371_000.0


def load_rsrp_measurements(
    csv_path: str,
    *,
    plausible_rsrp_range: tuple[float, float] = (-160, -30),
) -> pd.DataFrame:
    """
    Load a real RSRP-measurement CSV. Expected columns: `lat`, `lon` (or
    `long`), `cid`, `tech` (containing "5g" for 5G rows, anything else
    treated as LTE), `rsrp_lte`, `rsrp_5g`. Picks the RSRP column matching
    each row's `tech`, drops rows with missing coordinates/cid/tech, and
    filters out physically-implausible RSRP values.
    """
    df = pd.read_csv(csv_path).rename(columns={"long": "lon"})
    for c in ["lat", "lon", "rsrp_lte", "rsrp_5g"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["cid"] = df["cid"].astype(str)
    df["tech"] = df["tech"].astype(str).str.lower()
    df = df.dropna(subset=["lat", "lon", "cid", "tech"]).copy()

    df["rsrp"] = np.where(df["tech"].str.contains("5g", na=False), df["rsrp_5g"], df["rsrp_lte"])
    df["rsrp"] = pd.to_numeric(df["rsrp"], errors="coerce")
    df = df.dropna(subset=["rsrp"]).copy()

    lo, hi = plausible_rsrp_range
    df = df[(df["rsrp"] >= lo) & (df["rsrp"] <= hi)].copy()
    return df


def estimate_bs_per_cid(df: pd.DataFrame, top_q: float = 0.20) -> pd.DataFrame:
    """
    One estimated location per cell id: the median lat/lon of that cell's
    strongest `top_q` fraction of readings (strong signal => likely close
    to the transmitter).
    """
    def _estimate(g):
        thr = g["rsrp"].quantile(1 - top_q)
        gg = g[g["rsrp"] >= thr]
        return pd.Series({
            "lat_bs": gg["lat"].median(),
            "lon_bs": gg["lon"].median(),
            "n_samples": len(g),
            "n_top": len(gg),
            "tech_mode": g["tech"].mode().iloc[0] if not g["tech"].mode().empty else "unk",
        })

    return df.groupby("cid", as_index=False).apply(_estimate, include_groups=False).reset_index(drop=True)


def cluster_sites(bs_df: pd.DataFrame, eps_m: float = 1.0) -> pd.DataFrame:
    """
    Group per-cid estimated locations that sit within `eps_m` meters of
    each other into physical sites (DBSCAN, haversine metric -- so `eps_m`
    is a real ground distance, not a naive lat/lon Euclidean one).
    """
    coords_rad = np.radians(bs_df[["lat_bs", "lon_bs"]].to_numpy())
    eps_rad = eps_m / EARTH_RADIUS_M
    clu = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine")
    site_ids = clu.fit_predict(coords_rad)

    bs_df = bs_df.assign(site_id=site_ids)
    sites = (
        bs_df.groupby("site_id")
        .agg(
            lat_site=("lat_bs", "median"),
            lon_site=("lon_bs", "median"),
            n_cids=("cid", "count"),
            total_samples=("n_samples", "sum"),
        )
        .reset_index()
    )
    return sites


def bounding_box(sites: pd.DataFrame) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) covering every estimated site."""
    return (
        float(sites["lon_site"].min()), float(sites["lat_site"].min()),
        float(sites["lon_site"].max()), float(sites["lat_site"].max()),
    )


def plot_coverage_and_sites(df: pd.DataFrame, sites: pd.DataFrame, out_path: str | None = None):
    """Scatter of raw measurement positions + estimated site markers + bounding box. No OSM data required."""
    import matplotlib.pyplot as plt

    lon_min, lat_min, lon_max, lat_max = bounding_box(sites)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df["lon"], df["lat"], s=1, alpha=0.05)
    sizes = 40 + 30 * np.log1p(sites["n_cids"].to_numpy())
    ax.scatter(sites["lon_site"], sites["lat_site"], s=sizes, marker="X", edgecolors="black")
    for r in sites.itertuples(index=False):
        ax.annotate(f"{r.n_cids}", (r.lon_site, r.lat_site), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min], [lat_min, lat_min, lat_max, lat_max, lat_min], linewidth=2)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Coverage samples + estimated BS sites (bounding box)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_sites_on_osm(
    sites: pd.DataFrame,
    osm_pbf_path: str,
    bbox: tuple[float, float, float, float] | None = None,
    out_path: str | None = None,
    title: str = "Base station locations",
):
    """
    Plot estimated sites over a real street network. Needs a local
    `.osm.pbf` extract (e.g. from Geofabrik) and the `pyrosm` package --
    both bring-your-own, not bundled with this workflow (see the parent
    notebook's parameters cell). Imports `pyrosm` lazily so the rest of
    this module works without it installed.
    """
    from pyrosm import OSM
    import matplotlib.pyplot as plt

    bbox = bbox or bounding_box(sites)
    osm = OSM(osm_pbf_path, bounding_box=list(bbox))
    _nodes, edges = osm.get_network(network_type="driving", nodes=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    edges.plot(ax=ax, linewidth=0.5, color="gray", alpha=0.7)
    ax.plot(sites["lon_site"], sites["lat_site"], "mx", markersize=6, markeredgewidth=3, label="Base Station Location")
    ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig
