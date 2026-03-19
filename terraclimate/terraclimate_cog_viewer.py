#!/usr/bin/env python3
"""
Annual global TerraClimate COG viewer (12-band year file).

- Scroll wheel: zoom
- Left-drag: pan
- Right-click: probe value under cursor
- [ / ]: prev/next month band
- 1..9: jump to month 1..9
- 0: jump to month 10
- -: month 11
- =: month 12
- n / p: next/prev file (if you opened a directory)
- a: toggle autoscale (per-view) vs fixed global percentile scale
- c: recompute global percentile scale for current band
- r: reset view (fit world)
- q / Esc: quit

Works best on COGs (uses internal tiling/overviews via GDAL/rasterio).
"""

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

if not os.environ.get("MPLBACKEND"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt  # noqa: E402

try:
    import rasterio  # noqa: E402
    from rasterio.enums import Resampling  # noqa: E402
    from rasterio.windows import Window  # noqa: E402
    from rasterio.windows import from_bounds  # noqa: E402
except Exception as e:
    print(
        "This viewer requires rasterio.\n"
        "Install with:\n"
        "  conda install -c conda-forge rasterio matplotlib\n"
        "or:\n"
        "  pip install rasterio matplotlib\n\n"
        f"Import error: {e}",
        file=sys.stderr,
    )
    raise SystemExit(2)


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def list_cogs(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted([*p.rglob("*.tif"), *p.rglob("*.tiff")])
        # Prefer obvious COG-ish names if present
        cogs = [f for f in files if ".cog." in f.name.lower()]
        return cogs if cogs else files
    return []


@dataclass
class ViewState:
    cx: float  # center x in dataset CRS units
    cy: float  # center y in dataset CRS units
    scale: float  # dataset CRS units per screen pixel
    dragging: bool = False
    drag_start: Optional[Tuple[float, float]] = None  # screen coords
    drag_start_center: Optional[Tuple[float, float]] = None  # (cx,cy)


class AnnualCogViewer:
    def __init__(self, paths: List[Path], band: int, max_dim: int, resampling: str):
        if not paths:
            raise ValueError("No .tif/.tiff files found")

        self.paths = paths
        self.idx = 0

        self.max_dim = max(256, int(max_dim))
        self.resampling = Resampling.bilinear if resampling == "bilinear" else Resampling.nearest

        self.ds = None
        self.path: Optional[Path] = None
        self.bounds = None
        self.nodata = None
        self.count = 1

        self.band = int(band)
        self.scale_cache: Dict[int, Tuple[float, float]] = {}
        self.autoscale = False

        plt.ioff()
        self.fig, self.ax = plt.subplots()
        self.ax.set_axis_off()
        self.im = None
        self.state: Optional[ViewState] = None

        self._connect_events()
        self._open_index(0)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _close_ds(self):
        if self.ds is not None:
            try:
                self.ds.close()
            except Exception:
                pass
        self.ds = None

    def _fit_view(self):
        # Fit whole dataset into max_dim, in dataset CRS units per screen pixel.
        assert self.ds is not None
        b = self.ds.bounds
        w_px, h_px = self._viewport_px()
        full_w = b.right - b.left
        full_h = b.top - b.bottom
        scale_fit = max(full_w / w_px, full_h / h_px, 1e-12)
        self.state = ViewState(
            cx=(b.left + b.right) * 0.5,
            cy=(b.bottom + b.top) * 0.5,
            scale=scale_fit,
        )

    def _open_index(self, i: int):
        self._close_ds()
        self.idx = int(clamp(i, 0, len(self.paths) - 1))
        self.path = self.paths[self.idx]

        self.ds = rasterio.open(self.path)
        self.bounds = self.ds.bounds
        self.nodata = self.ds.nodata
        self.count = int(self.ds.count)

        # keep band in range; default to 1 if weird
        self.band = int(clamp(self.band, 1, max(1, self.count)))

        self.scale_cache.clear()
        self._fit_view()
        self._ensure_scale_for_band(self.band)
        self._render()

    def _viewport_px(self) -> Tuple[int, int]:
        w_px, h_px = self.fig.canvas.get_width_height()
        w_px = max(256, int(w_px))
        h_px = max(256, int(h_px))

        s = max(w_px, h_px) / self.max_dim if max(w_px, h_px) > self.max_dim else 1.0
        out_w = int(max(256, w_px / s))
        out_h = int(max(256, h_px / s))
        out_w = min(out_w, self.max_dim)
        out_h = min(out_h, self.max_dim)
        return out_w, out_h

    def _window_from_state(self) -> Tuple[Window, Tuple[float, float, float, float], Tuple[int, int]]:
        assert self.ds is not None
        assert self.state is not None
        b = self.ds.bounds

        out_w, out_h = self._viewport_px()

        view_w = out_w * self.state.scale
        view_h = out_h * self.state.scale

        left = self.state.cx - view_w * 0.5
        right = self.state.cx + view_w * 0.5
        bottom = self.state.cy - view_h * 0.5
        top = self.state.cy + view_h * 0.5

        # clamp to dataset bounds
        left = clamp(left, b.left, b.right)
        right = clamp(right, b.left, b.right)
        bottom = clamp(bottom, b.bottom, b.top)
        top = clamp(top, b.bottom, b.top)

        # ensure non-zero
        if right <= left:
            right = min(b.right, left + (b.right - b.left) * 0.01)
        if top <= bottom:
            top = min(b.top, bottom + (b.top - b.bottom) * 0.01)

        win = from_bounds(left, bottom, right, top, transform=self.ds.transform)
        extent = (left, right, bottom, top)
        return win, extent, (out_w, out_h)

    def _read_view(self) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        assert self.ds is not None
        win, extent, out_shape = self._window_from_state()
        out_h, out_w = out_shape[1], out_shape[0]
        arr = self.ds.read(
            self.band,
            window=win,
            out_shape=(out_h, out_w),
            resampling=self.resampling,
            masked=False,
        )
        return arr, extent

    def _robust_scale(self, band: int) -> Tuple[float, float]:
        # Fast-ish global scale based on a downsampled read.
        assert self.ds is not None
        h = int(clamp(self.ds.height // 8, 256, 1024))
        w = int(clamp(self.ds.width // 8, 512, 2048))

        arr = self.ds.read(
            band,
            out_shape=(h, w),
            resampling=Resampling.average if self.resampling != Resampling.nearest else Resampling.nearest,
            masked=False,
        ).astype(np.float32, copy=False)

        if self.nodata is not None:
            arr = arr[arr != self.nodata]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return (0.0, 1.0)

        lo = float(np.percentile(arr, 2.0))
        hi = float(np.percentile(arr, 98.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            if hi <= lo:
                hi = lo + 1.0
        return (lo, hi)

    def _ensure_scale_for_band(self, band: int):
        if band not in self.scale_cache:
            self.scale_cache[band] = self._robust_scale(band)

    def _title(self, extent: Tuple[float, float, float, float], vmin: float, vmax: float):
        assert self.path is not None
        left, right, bottom, top = extent
        month = MONTH_NAMES[self.band - 1] if 1 <= self.band <= 12 else f"band {self.band}"

        zoom = 1.0 / self.state.scale if self.state and self.state.scale > 0 else 0.0
        self.ax.set_title(
            f"{self.path.name}  [{self.idx+1}/{len(self.paths)}]  "
            f"{month} ({self.band}/{self.count})  "
            f"view lon[{left:.2f},{right:.2f}] lat[{bottom:.2f},{top:.2f}]  "
            f"zoom≈{zoom:.2f}x  "
            f"scale={'auto' if self.autoscale else f'p2–p98 ({vmin:.2f}..{vmax:.2f})'}  "
            f"(scroll zoom, drag pan, right-click probe, [ ] month, n/p file, a autoscale, c rescale, r reset)"
        )

    def _render(self):
        assert self.state is not None
        assert self.ds is not None

        self._ensure_scale_for_band(self.band)
        arr, extent = self._read_view()

        arr = arr.astype(np.float32, copy=False)
        if self.nodata is not None:
            arr = np.where(arr == self.nodata, np.nan, arr)

        if self.autoscale:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                vmin = float(np.percentile(finite, 2.0))
                vmax = float(np.percentile(finite, 98.0))
                if vmax <= vmin:
                    vmin, vmax = float(np.min(finite)), float(np.max(finite))
                    if vmax <= vmin:
                        vmax = vmin + 1.0
            else:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = self.scale_cache[self.band]

        self._title(extent, vmin, vmax)

        if self.im is None:
            self.im = self.ax.imshow(
                arr,
                origin="upper",
                extent=extent,
                interpolation="nearest",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            self.im.set_data(arr)
            self.im.set_extent(extent)
            self.im.set_clim(vmin, vmax)

        self.fig.canvas.draw_idle()

    def _probe(self, x: float, y: float):
        # x,y are in dataset CRS units (lon/lat in EPSG:4326)
        assert self.ds is not None
        col, row = self.ds.index(x, y)
        if col < 0 or row < 0 or col >= self.ds.width or row >= self.ds.height:
            return
        win = Window(col_off=col, row_off=row, width=1, height=1)
        val = self.ds.read(self.band, window=win, out_shape=(1, 1), resampling=Resampling.nearest)[0, 0]
        try:
            val = float(val)
        except Exception:
            pass
        month = MONTH_NAMES[self.band - 1] if 1 <= self.band <= 12 else str(self.band)
        print(f"probe: file={self.path.name} band={self.band}({month}) lon={x:.6f} lat={y:.6f} value={val}")

    def _on_scroll(self, ev):
        if ev.inaxes != self.ax or self.state is None or self.ds is None:
            return
        if ev.xdata is None or ev.ydata is None:
            return

        zoom_factor = 1.25 if ev.button == "up" else (1.0 / 1.25)
        new_scale = self.state.scale / zoom_factor
        b = self.ds.bounds
        full_w = b.right - b.left
        full_h = b.top - b.bottom
        # clamp zoom
        new_scale = clamp(new_scale, min(full_w, full_h) / 8000.0, max(full_w, full_h))

        # zoom around cursor (keep cursor location stable)
        cx0, cy0, s0 = self.state.cx, self.state.cy, self.state.scale
        out_w, out_h = self._viewport_px()
        view_w0 = out_w * s0
        view_h0 = out_h * s0
        left0 = cx0 - view_w0 * 0.5
        bottom0 = cy0 - view_h0 * 0.5

        fx = (ev.xdata - left0) / view_w0 if view_w0 > 0 else 0.5
        fy = (ev.ydata - bottom0) / view_h0 if view_h0 > 0 else 0.5
        fx = clamp(fx, 0.0, 1.0)
        fy = clamp(fy, 0.0, 1.0)

        view_w1 = out_w * new_scale
        view_h1 = out_h * new_scale
        left1 = ev.xdata - fx * view_w1
        bottom1 = ev.ydata - fy * view_h1

        self.state.scale = new_scale
        self.state.cx = clamp(left1 + view_w1 * 0.5, b.left, b.right)
        self.state.cy = clamp(bottom1 + view_h1 * 0.5, b.bottom, b.top)

        self._render()

    def _on_press(self, ev):
        if ev.inaxes != self.ax or self.state is None:
            return
        if ev.button == 1:
            self.state.dragging = True
            self.state.drag_start = (ev.x, ev.y)
            self.state.drag_start_center = (self.state.cx, self.state.cy)
        elif ev.button == 3:
            if ev.xdata is not None and ev.ydata is not None:
                self._probe(ev.xdata, ev.ydata)

    def _on_release(self, ev):
        if self.state is None:
            return
        self.state.dragging = False
        self.state.drag_start = None
        self.state.drag_start_center = None

    def _on_motion(self, ev):
        if self.ds is None or self.state is None:
            return
        if not self.state.dragging or self.state.drag_start is None or self.state.drag_start_center is None:
            return

        dx = ev.x - self.state.drag_start[0]
        dy = ev.y - self.state.drag_start[1]

        b = self.ds.bounds
        cx0, cy0 = self.state.drag_start_center

        # screen px -> dataset units via scale
        self.state.cx = clamp(cx0 - dx * self.state.scale, b.left, b.right)
        self.state.cy = clamp(cy0 + dy * self.state.scale, b.bottom, b.top)  # y grows upward in CRS

        self._render()

    def _set_band(self, b: int):
        b = int(clamp(b, 1, max(1, self.count)))
        if b != self.band:
            self.band = b
            self._ensure_scale_for_band(self.band)
            self._render()

    def _on_key(self, ev):
        k = (ev.key or "").lower()

        if k in ("q", "escape"):
            plt.close(self.fig)
            return

        if k == "r":
            self._fit_view()
            self._render()
            return

        if k == "a":
            self.autoscale = not self.autoscale
            self._render()
            return

        if k == "c":
            # recompute global robust scale for current band
            if self.ds is not None:
                self.scale_cache[self.band] = self._robust_scale(self.band)
            self._render()
            return

        # month band navigation
        if k == "[":
            self._set_band(self.band - 1)
            return
        if k == "]":
            self._set_band(self.band + 1)
            return

        if k in ("1","2","3","4","5","6","7","8","9"):
            self._set_band(int(k))
            return
        if k == "0":
            self._set_band(10)
            return
        if k == "-":
            self._set_band(11)
            return
        if k == "=":
            self._set_band(12)
            return

        # file navigation if viewing a directory
        if k in ("n",):
            self._open_index(self.idx + 1)
            return
        if k in ("p",):
            self._open_index(self.idx - 1)
            return


def main() -> int:
    ap = argparse.ArgumentParser(description="Annual global COG viewer (toggle month bands + pan/zoom).")
    ap.add_argument("path", help="Annual global 12-band COG file OR directory containing many COGs")
    ap.add_argument("--band", type=int, default=1, help="Initial band (1-12 for months)")
    ap.add_argument("--max-dim", type=int, default=1600, help="Max render dimension (smaller = faster)")
    ap.add_argument("--resampling", choices=["nearest", "bilinear"], default="bilinear")
    ap.add_argument("--pause-on-exit", action="store_true")
    args = ap.parse_args()

    paths = list_cogs(Path(args.path).expanduser())
    if not paths:
        print("No .tif/.tiff files found at path.", file=sys.stderr)
        return 2

    viewer = AnnualCogViewer(paths=paths, band=args.band, max_dim=args.max_dim, resampling=args.resampling)
    plt.show(block=True)
    return 0


if __name__ == "__main__":
    pause = "--pause-on-exit" in sys.argv
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        if pause:
            input("Press Enter to exit...")
        raise