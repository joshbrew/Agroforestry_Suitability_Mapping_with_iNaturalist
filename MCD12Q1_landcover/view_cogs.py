#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
    from rasterio.windows import Window  # noqa: E402
    from rasterio.enums import Resampling  # noqa: E402
except Exception as e:
    print(
        "This viewer requires rasterio.\n"
        "Install with:\n"
        "  conda install -c conda-forge rasterio\n"
        "or:\n"
        "  pip install rasterio\n\n"
        f"Import error: {e}",
        file=sys.stderr,
    )
    raise SystemExit(2)

import subprocess  # noqa: E402


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + p.stdout
            + "\n\nSTDERR:\n"
            + p.stderr
        )


def list_cogs(dir_path: Path) -> List[Path]:
    files = sorted(
        [
            *dir_path.glob("*.cog.tif"),
            *dir_path.glob("*.cog.tiff"),
            *dir_path.glob("*.tif"),
            *dir_path.glob("*.tiff"),
        ]
    )
    return [p for p in files if p.is_file()]


def build_vrt_from_dir(
    cog_dir: Path,
    vrt_path: Path,
    rebuild: bool,
) -> Path:
    if vrt_path.exists() and not rebuild:
        return vrt_path

    cogs = list_cogs(cog_dir)
    if not cogs:
        raise RuntimeError(f"No .tif files found in: {cog_dir}")

    vrt_path.parent.mkdir(parents=True, exist_ok=True)

    # Use -input_file_list to avoid Windows command-line length limits.
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as f:
        list_path = Path(f.name)
        for p in cogs:
            f.write(str(p.resolve()) + "\n")

    try:
        cmd = [
            "gdalbuildvrt",
            "-overwrite",
            "-input_file_list",
            str(list_path),
            str(vrt_path),
        ]
        run_cmd(cmd)
    finally:
        try:
            list_path.unlink()
        except OSError:
            pass

    return vrt_path


@dataclass
class ViewState:
    cx: float
    cy: float
    scale: float  # raster pixels per screen pixel
    dragging: bool = False
    drag_start: Optional[Tuple[float, float]] = None
    drag_start_center: Optional[Tuple[float, float]] = None


class RasterViewer:
    def __init__(
        self,
        raster_path: Path,
        band: int = 1,
        max_dim: int = 1600,
        resampling: str = "nearest",
    ):
        self.raster_path = raster_path
        self.band = int(band)
        self.max_dim = max(256, int(max_dim))
        self.resampling = Resampling.nearest if resampling == "nearest" else Resampling.bilinear

        plt.ioff()
        self.fig, self.ax = plt.subplots()
        self.ax.set_axis_off()
        self.im = None

        self.ds = rasterio.open(self.raster_path)
        self.w = int(self.ds.width)
        self.h = int(self.ds.height)
        self.nodata = self.ds.nodata

        scale_fit = max(self.w / self.max_dim, self.h / self.max_dim, 1.0)
        self.state = ViewState(cx=self.w * 0.5, cy=self.h * 0.5, scale=scale_fit)

        self._connect_events()
        self._render()

    def _connect_events(self):
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

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

    def _read_window(self, cx: float, cy: float, scale: float) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        out_w, out_h = self._viewport_px()

        view_w = out_w * scale
        view_h = out_h * scale

        left = cx - view_w * 0.5
        top = cy - view_h * 0.5

        left_c = clamp(left, 0.0, max(0.0, self.w - 1.0))
        top_c = clamp(top, 0.0, max(0.0, self.h - 1.0))
        right_c = clamp(left + view_w, 1.0, float(self.w))
        bottom_c = clamp(top + view_h, 1.0, float(self.h))

        win_w = max(1.0, right_c - left_c)
        win_h = max(1.0, bottom_c - top_c)

        window = Window(col_off=left_c, row_off=top_c, width=win_w, height=win_h)

        arr = self.ds.read(
            self.band,
            window=window,
            out_shape=(out_h, out_w),
            resampling=self.resampling,
            masked=False,
        )

        extent = (left_c, left_c + win_w, top_c + win_h, top_c)  # (x0, x1, y0, y1), origin upper
        return arr, extent

    def _set_title(self):
        zoom = 1.0 / self.state.scale
        self.ax.set_title(
            f"{self.raster_path.name}  {self.w}x{self.h}  band={self.band}  zoom≈{zoom:.2f}x  "
            f"(scroll=zoom, drag=pan, right-click=probe, r=reset, 1=1:1, q=quit)"
        )

    def _render(self):
        self._set_title()
        arr, extent = self._read_window(self.state.cx, self.state.cy, self.state.scale)

        if self.nodata is not None:
            arr = np.where(arr == self.nodata, 0, arr)

        if np.issubdtype(arr.dtype, np.integer):
            cmap = plt.get_cmap("tab20", 256)
            vmin, vmax = 0, max(int(arr.max()) if arr.size else 1, 1)
        else:
            cmap = "viridis"
            vmin, vmax = None, None

        if self.im is None:
            self.im = self.ax.imshow(
                arr,
                origin="upper",
                extent=extent,
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            self.im.set_data(arr)
            self.im.set_extent(extent)
            self.im.set_cmap(cmap)
            if vmin is not None and vmax is not None:
                self.im.set_clim(vmin, vmax)

        self.fig.canvas.draw_idle()

    def _probe(self, x: float, y: float):
        col = int(x)
        row = int(y)
        if col < 0 or row < 0 or col >= self.w or row >= self.h:
            return
        window = Window(col_off=col, row_off=row, width=1, height=1)
        val = self.ds.read(self.band, window=window, out_shape=(1, 1), resampling=Resampling.nearest)[0, 0]
        try:
            val = int(val)
        except Exception:
            pass
        print(f"probe: col={col} row={row} value={val}")

    def _on_scroll(self, ev):
        if ev.inaxes != self.ax:
            return
        if ev.xdata is None or ev.ydata is None:
            return

        zoom_factor = 1.25 if ev.button == "up" else (1.0 / 1.25)
        new_scale = self.state.scale / zoom_factor
        new_scale = clamp(new_scale, 0.1, max(self.w, self.h))

        cx, cy, s = self.state.cx, self.state.cy, self.state.scale
        out_w, out_h = self._viewport_px()
        view_w = out_w * s
        view_h = out_h * s
        left = cx - view_w * 0.5
        top = cy - view_h * 0.5

        fx = (ev.xdata - left) / view_w if view_w > 0 else 0.5
        fy = (ev.ydata - top) / view_h if view_h > 0 else 0.5
        fx = clamp(fx, 0.0, 1.0)
        fy = clamp(fy, 0.0, 1.0)

        new_view_w = out_w * new_scale
        new_view_h = out_h * new_scale
        new_left = ev.xdata - fx * new_view_w
        new_top = ev.ydata - fy * new_view_h

        self.state.scale = new_scale
        self.state.cx = clamp(new_left + new_view_w * 0.5, 0.0, float(self.w))
        self.state.cy = clamp(new_top + new_view_h * 0.5, 0.0, float(self.h))
        self._render()

    def _on_press(self, ev):
        if ev.inaxes != self.ax:
            return
        if ev.button == 1:
            self.state.dragging = True
            self.state.drag_start = (ev.x, ev.y)
            self.state.drag_start_center = (self.state.cx, self.state.cy)
        elif ev.button == 3:
            if ev.xdata is not None and ev.ydata is not None:
                self._probe(ev.xdata, ev.ydata)

    def _on_release(self, ev):
        self.state.dragging = False
        self.state.drag_start = None
        self.state.drag_start_center = None

    def _on_motion(self, ev):
        if not self.state.dragging or self.state.drag_start is None or self.state.drag_start_center is None:
            return
        dx = ev.x - self.state.drag_start[0]
        dy = ev.y - self.state.drag_start[1]

        cx0, cy0 = self.state.drag_start_center
        self.state.cx = clamp(cx0 - dx * self.state.scale, 0.0, float(self.w))
        self.state.cy = clamp(cy0 - dy * self.state.scale, 0.0, float(self.h))
        self._render()

    def _on_key(self, ev):
        k = (ev.key or "").lower()
        if k in ("q", "escape"):
            plt.close(self.fig)
            return
        if k == "r":
            scale_fit = max(self.w / self.max_dim, self.h / self.max_dim, 1.0)
            self.state.cx = self.w * 0.5
            self.state.cy = self.h * 0.5
            self.state.scale = scale_fit
            self._render()
            return
        if k == "1":
            self.state.scale = 1.0
            self._render()
            return


def main() -> int:
    ap = argparse.ArgumentParser(description="View a mosaic of COG tiles by building/opening a VRT.")
    ap.add_argument("path", help="Directory of COGs or a .vrt file")
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument("--max-dim", type=int, default=1600, help="Max render dimension (smaller = faster)")
    ap.add_argument("--bilinear", action="store_true", help="Bilinear resampling (default nearest)")
    ap.add_argument("--vrt", default="", help="VRT path to create/use (default: <dir>/mosaic.vrt)")
    ap.add_argument("--rebuild-vrt", action="store_true", help="Force rebuild the VRT")
    ap.add_argument("--pause-on-exit", action="store_true", help="Pause before exiting if there is an error")
    args = ap.parse_args()

    p = Path(args.path).expanduser()
    if p.is_file() and p.suffix.lower() == ".vrt":
        raster_path = p
    elif p.is_dir():
        vrt_path = Path(args.vrt).expanduser() if args.vrt else (p / "mosaic.vrt")
        raster_path = build_vrt_from_dir(p, vrt_path, args.rebuild_vrt)
    else:
        print("Path must be a directory of COGs or a .vrt file.", file=sys.stderr)
        return 2

    viewer = RasterViewer(
        raster_path=raster_path,
        band=args.band,
        max_dim=args.max_dim,
        resampling="bilinear" if args.bilinear else "nearest",
    )
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