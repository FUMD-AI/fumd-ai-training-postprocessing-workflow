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


# Known real-world base-station coordinates for the Alicante area every
# dataset bundled with/available to this repository was simulated over
# (confirmed against dataset_labeled_w<W>.csv's own x/y extent, ~-0.497 to
# -0.477 lon, ~38.336 to 38.352 lat) -- the same 9-site dict
# `mapa_EB_K_mapAlacant.ipynb` and `estimacion_antenas.ipynb` both hardcoded
# (comment there: "Coordinates in excel file Alicante_touristic_places").
# {cell_id: (lon, lat)}, ready to pass as `bs_coords`/`BS_COORDS` for any
# run against this project's own datasets. Not a real base-station-to-
# simulation-cell-id mapping for a *different* simulated area or city --
# build a new dict by hand for one of those (see this module's own
# docstring above).
ALICANTE_BS_COORDS: dict[int, tuple[float, float]] = {
    1: (-0.4902473, 38.3459893),  # Luceros
    2: (-0.4853830, 38.3435634),  # Gabriel Miro
    3: (-0.4948606, 38.3430170),  # Teatro Arniches
    4: (-0.4867156, 38.3490117),  # Plaza del Mercado
    5: (-0.4883843, 38.3394852),  # Paseo Canalejas
    6: (-0.4819092, 38.3484225),  # Parque de La Ereta
    7: (-0.4789899, 38.3479951),  # Castillo Santa Barbara
    8: (-0.4785041, 38.3437444),  # Playa Postiguet
    9: (-0.4802594, 38.3395432),  # Zona Volvo
}


def derive_events_path(dataset_path: str) -> str:
    """
    Given a `dataset_labeled_w<W>.csv` path, return the matching
    `events_all_w<W>.csv` path in the same directory -- the naming
    convention the FUMD-AI preprocessing workflow always pairs the two
    files under (same directory, same window size `W`).

    Used by `run_pipeline.ipynb`'s combined-dataset track to find each
    source run's own events file for its per-source migration map, without
    also having to list events paths by hand alongside
    `COMBINE_DATASET_PATHS` (positions and events for a single run are
    never spread across different directories in this project's data
    layout, so the derivation is unambiguous).

    Raises ValueError if `dataset_path`'s filename doesn't match the
    `dataset_labeled_w<W>.csv` pattern -- call sites that may see
    non-conforming filenames should catch this and fall back to asking the
    caller for an explicit events path instead of guessing one.
    """
    directory, filename = os.path.split(dataset_path)
    m = re.fullmatch(r"dataset_labeled_(w\d+)\.csv", filename)
    if not m:
        raise ValueError(
            f"derive_events_path: {filename!r} doesn't match the expected "
            "'dataset_labeled_w<W>.csv' naming convention, so the matching "
            "events_all_w<W>.csv path can't be derived automatically -- "
            "pass an explicit events path instead."
        )
    events_filename = f"events_all_{m.group(1)}.csv"
    return os.path.join(directory, events_filename) if directory else events_filename


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
    `case_short`, sorted for consistent plot ordering.

    Also corrects a known mislabeling in the upstream event data: the
    "C2b_handover_sin_historico" ("no prior history") and "C3_pingpong"
    ("ABA") cases have their numeric prefixes swapped -- rows the raw
    `case` column tags "C2b" are the ABA/ping-pong case and should read
    "C3", and rows it tags "C3" are the no-prior-history case and should
    read "C2b". The original `mapa_EB_K_mapAlacant.ipynb` notebook applied
    this same swap by hand at plot time (`if case == 'C2b': case = 'C3'
    ...`); this reproduces it here once instead, so every downstream
    consumer of `case_short` (plots, tables, future analysis) sees the
    corrected code without needing to know about the swap.
    """
    ev = events[~events[case_col].astype(str).isin(exclude)].copy()
    ev["case_short"] = ev[case_col].astype(str).str.extract(r"^(C\d+[a-z]?)", expand=False)
    ev["case_short"] = ev["case_short"].replace({"C2b": "C3", "C3": "C2b"})
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
    zoom_to_events: bool = True,
    zoom_padding_frac: float = 0.1,
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

    `edges` (a full street network) is typically far larger than the area
    events actually occurred in, so by default (`zoom_to_events=True`)
    every subplot is cropped to the bounding box of all matched events in
    `ev` (padded by `zoom_padding_frac` of that box's width/height on each
    side, plus any `bs_coords` markers so they aren't clipped) instead of
    the full extent of `edges` -- pass `zoom_to_events=False` to see the
    whole map (e.g. for spatial context) instead.
    """
    import matplotlib.pyplot as plt

    types = _sorted_case_types(ev)
    rows = (len(types) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    colormap = plt.get_cmap("tab10")
    color_by_group = {g: colormap(i % 10) for i, g in enumerate(order_groups)}

    xlim = ylim = None
    if zoom_to_events and not ev.empty:
        xs = list(ev["x_event"]); ys = list(ev["y_event"])
        if bs_coords:
            xs += [lon for lon, lat in bs_coords.values()]
            ys += [lat for lon, lat in bs_coords.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # A single point (or all-identical coordinates) has zero width/height
        # -- pad by a small absolute amount in that case instead of 10% of 0.
        x_pad = (x_max - x_min) * zoom_padding_frac or max(abs(x_max), 1.0) * 0.05
        y_pad = (y_max - y_min) * zoom_padding_frac or max(abs(y_max), 1.0) * 0.05
        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

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
        if xlim is not None:
            # Set the crop explicitly, then keep the aspect ratio square by
            # resizing the subplot's box rather than re-expanding the limits
            # back out -- ax.axis("equal") alone re-autoscales to the data
            # (here, the *un-cropped* edges/scatter extent), undoing the crop.
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
        else:
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
