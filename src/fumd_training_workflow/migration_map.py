"""
=============================================================================
FUMD-AI Training Workflow -- shared library: spatial migration-event maps
=============================================================================
Used by: notebooks/step_5_map_migration_events.ipynb

Author(s):
  - Cristina Bernad (ORCID: 0000-0001-9537-415X)
  - Sonja Filiposka <sonja.filiposka@finki.ukim.mk> (ORCID: 0000-0003-0034-2855)
  - Katja Gilly (ORCID: 0000-0002-8985-0639)

Funding: This work has been funded by the FUMD-AI project, an EOSC GRAVITY -
Inter Project with Grant Number 25-EOSC-GRV-INTER-013.

SPDX-License-Identifier: MIT
-----------------------------------------------------------------------------

Matches handover/migration events to the vehicle position at the moment
they occurred, then plots them grouped by event type and colored by
destination cell -- optionally over a real street map and real
base-station coordinates, supplied as a plain dict (see `BS_COORDS` below).

Refactored from `mapa_EB_K_mapAlacant.ipynb` / `mapa_EB_K-mapAveiro.ipynb`,
generalized so base-station coordinates are always a parameter (a plain
`{cell_id: (lon, lat)}` dict) instead of duplicated per-city notebook
constants -- a hand-curated dict of known site coordinates, the way the
original Alicante notebook worked.

Default data inputs are the FUMD-AI preprocessing workflow's own
`dataset_labeled_w<W>.csv` (positions) and `events_all_w<W>.csv` (handover
events) -- both already directly compatible, so unlike the original
notebooks this needs no bespoke intermediate files
(`dataset_con_migration_v5_*.csv` / `eventos_todoss_*.csv`).

Note: a real-measurement site id (e.g. clustered from actual RSRP data)
and a simulation's abstract `servingCell`/`destination` id are two
different numbering domains -- there is no automatic way to match "this
physical site" to "this simulation cell id". Building the `bs_coords` dict
passed to the plotting functions below is a manual, domain-knowledge step
(as the original Alicante notebook did by hand); this module does not
attempt to infer that mapping.
"""

from __future__ import annotations

import os
import re
import sqlite3
import struct
import tempfile
import zipfile

import numpy as np
import pandas as pd


# Default OSM `highway` tag values pyrosm's `network_type="driving"` covers --
# used as the default filter for `load_osm_lines_from_geopackage` so both
# background-map code paths (pyrosm+.osm.pbf, or this module's own
# GeoPackage reader) draw roughly the same road network by default.
DRIVING_HIGHWAY_TYPES = (
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "service", "track", "road",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
)


def _read_wkb_geometry(buf: bytes, offset: int = 0) -> tuple[list[list[tuple[float, float]]], int]:
    """Parse one standard ISO WKB geometry starting at `offset` in `buf`.
    Only LineString (type 2) and MultiLineString (type 5) are supported --
    all this module needs. Returns (list_of_linestrings, new_offset); a
    plain LineString yields a single-element list, a MultiLineString one
    element per member line."""
    byte_order = buf[offset]
    endian = "<" if byte_order == 1 else ">"
    offset += 1
    geom_type, = struct.unpack_from(endian + "I", buf, offset)
    offset += 4

    if geom_type == 2:  # LineString
        n, = struct.unpack_from(endian + "I", buf, offset)
        offset += 4
        pts = []
        for _ in range(n):
            x, y = struct.unpack_from(endian + "dd", buf, offset)
            pts.append((x, y))
            offset += 16
        return [pts], offset
    elif geom_type == 5:  # MultiLineString -- each member is a nested WKB blob
        n, = struct.unpack_from(endian + "I", buf, offset)
        offset += 4
        lines: list[list[tuple[float, float]]] = []
        for _ in range(n):
            sub_lines, offset = _read_wkb_geometry(buf, offset)
            lines.extend(sub_lines)
        return lines, offset
    else:
        raise ValueError(f"unsupported WKB geometry type {geom_type} (only LineString/MultiLineString are)")


def _read_gpkg_geometry(blob: bytes) -> list[list[tuple[float, float]]]:
    """Strip a GeoPackage binary geometry header (magic b"GP" + version +
    flags byte [byte order, envelope-contents code, ...] + srs_id + an
    optional envelope, whose length the flags byte's envelope code
    determines) and parse the standard WKB body that follows it."""
    if blob[0:2] != b"GP":
        raise ValueError('not a GeoPackage geometry blob (bad "GP" magic)')
    flags = blob[3]
    envelope_code = (flags >> 1) & 0x07
    envelope_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_code]
    offset = 8 + envelope_len  # 2 (magic) + 1 (version) + 1 (flags) + 4 (srs_id)
    lines, _ = _read_wkb_geometry(blob, offset)
    return lines


def load_osm_lines_from_geopackage(
    path: str,
    *,
    table: str = "lines",
    geom_col: str = "geom",
    highway_types: tuple[str, ...] | None = DRIVING_HIGHWAY_TYPES,
) -> list[list[tuple[float, float]]]:
    """
    Read road-network line geometries out of a GeoPackage exported by
    BBBike's extract service (https://extract.bbbike.org/, `format=
    geopackage.zip`) or `ogr2ogr`/osmium's own GeoPackage output -- both
    use the same schema: a "lines" table (LINESTRING geometry, SRS 4326 /
    WGS84) with an OSM `highway` tag column, alongside "points"/
    "multilinestrings"/"multipolygons"/"other_relations" tables this
    function ignores.

    An alternative to `pyrosm`'s `.osm.pbf` + `OSM.get_network(...)` path
    (see `step_5_map_migration_events.ipynb`'s `OSM_PBF_PATH` parameter):
    reads the GeoPackage directly via `sqlite3` + a minimal WKB parser --
    no `geopandas`/`fiona`/GDAL dependency, since a GeoPackage is just a
    SQLite database with geometries stored as (a small GeoPackage-specific
    header) + (standard ISO WKB), both of which the stdlib can decode
    directly. Useful when `pyrosm` itself is impractical to install (as it
    was for this project -- see README's "Notes on the source notebooks").

    `path` may be a `.gpkg` file directly, or the `.zip` BBBike actually
    delivers (auto-extracted into a temporary directory -- BBBike always
    names the `.gpkg` inside after the extract itself).

    `highway_types`, if given, filters to rows whose `highway` column is
    one of these values (default: `DRIVING_HIGHWAY_TYPES`, roughly
    matching pyrosm's own `network_type="driving"` filter) -- pass `None`
    for every line in the table regardless of `highway` value.

    Returns a list of linestrings, each a list of `(lon, lat)` tuples --
    directly usable as `plot_events_by_type`'s `edges` argument (which
    also accepts a geopandas GeoDataFrame, e.g. from pyrosm, and tells the
    two apart by duck-typing a `.plot` attribute).
    """
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            gpkg_names = [n for n in zf.namelist() if n.lower().endswith(".gpkg")]
            if not gpkg_names:
                raise ValueError(f"no .gpkg file found inside {path}")
            tmp_dir = tempfile.mkdtemp(prefix="fumd_osm_gpkg_")
            zf.extract(gpkg_names[0], tmp_dir)
            path = os.path.join(tmp_dir, gpkg_names[0])

    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cols = [row[1] for row in cur.execute(f"PRAGMA table_info({table})")]
        other_cols = [c for c in cols if c != geom_col]
        sql = f"SELECT {', '.join(other_cols)}, {geom_col} FROM {table}"
        if highway_types is not None and "highway" in other_cols:
            placeholders = ", ".join("?" for _ in highway_types)
            sql += f" WHERE highway IN ({placeholders})"
            rows = cur.execute(sql, highway_types)
        else:
            rows = cur.execute(sql)

        lines: list[list[tuple[float, float]]] = []
        for row in rows:
            blob = row[-1]
            if blob is not None:
                lines.extend(_read_gpkg_geometry(blob))
        return lines
    finally:
        conn.close()


def match_events_to_positions(
    positions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    veh_col: str = "veh_id",
    time_col: str = "t",
    event_time_col: str = "t_change",
    tolerance_s: float = 0.05,
) -> pd.DataFrame:
    """
    For each event row, find the closest-in-time position sample for that
    vehicle (within `tolerance_s`) and attach its (x, y) as
    (`x_event`, `y_event`). Events with no sample within tolerance are
    dropped (via the `matched` column).

    `positions` needs [veh_col, time_col, "x", "y"];
    `events` needs [veh_col, event_time_col] plus whatever event columns
    you want carried through (e.g. `to_cell`, `case`).
    """
    pos = positions.dropna(subset=[veh_col, time_col, "x", "y"]).copy()
    pos[veh_col] = pos[veh_col].round().astype(int)
    pos = pos.sort_values([veh_col, time_col]).reset_index(drop=True)

    ev = events.dropna(subset=[veh_col, event_time_col]).copy()
    ev[veh_col] = ev[veh_col].round().astype(int)

    index_by_vid = {}
    for vid, g in pos.groupby(veh_col, sort=False):
        index_by_vid[vid] = (g[time_col].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy())

    xs, ys, ok = [], [], []
    for row in ev.itertuples(index=False):
        vid = int(getattr(row, veh_col))
        te = float(getattr(row, event_time_col))
        if vid not in index_by_vid:
            xs.append(np.nan); ys.append(np.nan); ok.append(False)
            continue
        t_arr, x_arr, y_arr = index_by_vid[vid]
        pos_idx = np.searchsorted(t_arr, te)
        candidates = [i for i in (pos_idx, pos_idx - 1) if 0 <= i < len(t_arr)]
        best_i = min(candidates, key=lambda i: abs(t_arr[i] - te))
        if abs(t_arr[best_i] - te) <= tolerance_s:
            xs.append(x_arr[best_i]); ys.append(y_arr[best_i]); ok.append(True)
        else:
            xs.append(np.nan); ys.append(np.nan); ok.append(False)

    ev = ev.assign(x_event=xs, y_event=ys, matched=ok)
    return ev[ev["matched"] & ev["x_event"].notna() & ev["y_event"].notna()].reset_index(drop=True)


def add_case_short(events: pd.DataFrame, case_col: str = "case", exclude: tuple[str, ...] = ("C4_no_estable",)) -> pd.DataFrame:
    """Drop excluded case labels (default: the diagnostic-only "unstable, no
    handover" case) and extract a short case code (e.g. "C1", "C2a") into
    `case_short`, sorted for consistent plot ordering."""
    ev = events[~events[case_col].astype(str).isin(exclude)].copy()
    ev["case_short"] = ev[case_col].astype(str).str.extract(r"^(C\d+[a-z]?)", expand=False)
    return ev.dropna(subset=["case_short"]).reset_index(drop=True)


def group_by_top_destinations(events: pd.DataFrame, dest_col: str = "to_cell", top_k: int = 9) -> tuple[pd.DataFrame, list[str]]:
    """Add a `bs_group` column: the destination cell id as a string for the
    `top_k` most common destinations, `"Otras"` ("other") for the rest.
    Returns (events_with_group, ordered_group_labels) for consistent
    legend/color ordering across subplots."""
    ev = events.copy()
    top = ev[dest_col].value_counts().head(top_k).index.tolist()
    ev["bs_group"] = np.where(ev[dest_col].isin(top), ev[dest_col].astype(int).astype(str), "Otras")
    nums = sorted(int(x) for x in ev["bs_group"].unique() if x != "Otras")
    order = [str(x) for x in nums] + (["Otras"] if "Otras" in ev["bs_group"].unique() else [])
    return ev, order


def _sorted_case_types(ev: pd.DataFrame) -> list[str]:
    def key(s):
        m = re.match(r"^C(\d+)", s)
        return (int(m.group(1)) if m else 999, s)
    return sorted(ev["case_short"].dropna().unique(), key=key)


def plot_events_by_type(
    ev: pd.DataFrame,
    order_groups: list[str],
    *,
    bs_coords: dict[int, tuple[float, float]] | None = None,
    edges=None,
    cols: int = 2,
    title: str = "Event locations by type: destination BS",
    out_path: str | None = None,
):
    """
    Grid of subplots, one per event `case_short`, scattering matched event
    positions colored by `bs_group`. Optionally overlays `bs_coords`
    (`{cell_id: (lon, lat)}`, a hand-curated dict of known site
    coordinates) and a street network (`edges`), plotted first as a gray
    background if given -- both fully optional so this works with only
    simulation output. `edges` accepts either a geopandas GeoDataFrame
    (e.g. from `pyrosm`'s `OSM.get_network(...)`, duck-typed via a `.plot`
    attribute) or a plain list of linestrings -- each a list of `(x, y)`
    tuples -- such as `load_osm_lines_from_geopackage`'s return value.
    """
    import matplotlib.pyplot as plt

    types = _sorted_case_types(ev)
    rows = (len(types) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    colormap = plt.get_cmap("tab10")
    color_by_group = {g: colormap(i % 10) for i, g in enumerate(order_groups)}

    for i, case in enumerate(types):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        if edges is not None:
            if hasattr(edges, "plot"):
                edges.plot(ax=ax, linewidth=0.5, color="gray", alpha=0.7)
            else:
                from matplotlib.collections import LineCollection
                ax.add_collection(LineCollection(edges, linewidth=0.5, color="gray", alpha=0.7))
        sub = ev[ev["case_short"] == case]
        for g in order_groups:
            sub_g = sub[sub["bs_group"] == g]
            if not sub_g.empty:
                ax.scatter(sub_g["x_event"], sub_g["y_event"], s=14, alpha=0.65,
                           color=color_by_group[g], label=(f"BS {g}" if g != "Otras" else "Otras"))
        if bs_coords:
            for cell_id, (lon, lat) in bs_coords.items():
                color = color_by_group.get(str(cell_id), "black")
                ax.scatter(lon, lat, s=200, marker="o", facecolors="none", edgecolors=color, linewidths=2, zorder=10)
                ax.scatter(lon, lat, s=40, marker="o", color=color, zorder=11)
        ax.set_title(case); ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.axis("equal")
        # `axis("equal")` shrinks each subplot's plotted box to match the
        # data's aspect ratio, which narrows the space available for x tick
        # labels -- cap the tick count and rotate so long coordinate values
        # (e.g. SUMO meters, "123456.78") don't overlap each other.
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        h, l = ax.get_legend_handles_labels()
        by_label = dict(zip(l, h))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, frameon=True)

    for j in range(len(types), rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig
