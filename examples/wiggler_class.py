# Compact, refactored Wiggler
# - Less duplication via helpers and component loops
# - Clear ensure() dependency gates
# - Generic sine-fitting used for fields and curvature
# - Small quality-of-life utilities (mid masks, magnitude, plotting core)
#
# Public API preserved (same method names), plus docstrings tightened.

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy as sc
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, Optional
from bpmeth import poly_fit
import matplotlib.pyplot as plt


# ========================== Small helper classes ==========================

@dataclass
class SineModel:
    Acos: Optional[np.ndarray] = None
    Asin: Optional[np.ndarray] = None
    k: Optional[np.ndarray] = None

    def is_ready(self) -> bool:
        return self.Acos is not None and self.Asin is not None and self.k is not None

    def eval(self, z: np.ndarray) -> np.ndarray:
        if not self.is_ready():
            return np.zeros_like(z, dtype=float)
        y = np.zeros_like(z, dtype=float)
        for A1, A2, kk in zip(self.Acos, self.Asin, self.k):
            y += A1 * np.cos(kk * z) + A2 * np.sin(kk * z)
        return y

    # Pretty-printer for the series (kept from your version)
    def series_string(
        self,
        *,
        in_terms_of: str = "s",  # "s" or "z" (only affects variable name)
        var: Optional[str] = None,
        coeff_fmt: str = ".6g",
        k_fmt: Optional[str] = None,
        tol: float = 0.0,
        include_cos: bool = True,
        include_sin: bool = True,
        sort_by_k: bool = True,
    ) -> str:
        if not self.is_ready():
            return "0"
        cf, kf = coeff_fmt, (k_fmt or coeff_fmt)
        if var is None:
            var = "s" if in_terms_of == "s" else "z"
            arg = var
        else:
            raise ValueError('in_terms_of must be "s" or "z"')
        idx = np.arange(len(self.k))
        if sort_by_k:
            idx = idx[np.argsort(self.k)]
        parts = []
        for i in idx:
            A1, A2, kk = float(self.Acos[i]), float(self.Asin[i]), float(self.k[i])
            kk_str = format(kk, kf)
            if include_cos and abs(A1) > tol:
                parts.append(f"{format(A1, cf)}*cos({kk_str}*{arg})")
            if include_sin and abs(A2) > tol:
                parts.append(f"{format(A2, cf)}*sin({kk_str}*{arg})")
        if not parts:
            return "0"
        expr = " + ".join(parts).replace("+ -", "- ")
        if expr.startswith("+ "):
            expr = expr[2:]
        return expr


@dataclass
class PiecewisePoly:
    starts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    polys: list = field(default_factory=list)  # list[Polynomial]

    def set_pieces(self, starts: Iterable[int], polys: Iterable[Polynomial]) -> None:
        starts = np.asarray(list(starts), dtype=int)
        order = np.argsort(starts)
        self.starts = starts[order]
        polys = list(polys)
        self.polys = [polys[i] for i in order]

    def eval(self, z: np.ndarray, z_full: np.ndarray) -> np.ndarray:
        if not self.polys:
            return np.zeros_like(z, dtype=float)
        i = np.searchsorted(z_full, z, side="left")
        pid = np.searchsorted(self.starts, i, side="right") - 1
        pid = np.clip(pid, 0, len(self.polys) - 1)
        out = np.empty_like(z, dtype=float)
        for p in np.unique(pid):
            mask = (pid == p)
            out[mask] = self.polys[p](z[mask])
        return out


@dataclass
class FieldChannel:
    data: Optional[np.ndarray] = None
    fit: Optional[np.ndarray] = None
    borders_idx: Optional[Tuple[int, int]] = None  # [i0:i1)
    borders_z: Optional[Tuple[float, float]] = None
    sine: SineModel = field(default_factory=SineModel)
    left: PiecewisePoly = field(default_factory=PiecewisePoly)
    right: PiecewisePoly = field(default_factory=PiecewisePoly)


# ================================ Main class ================================

class Wiggler:
    """Compact Wiggler fitter (Bx, By, Bz + curvature fits).

    Public API:
      - fit(), tune_slices_for_zero_integral(), evaluate(), results()
      - plot_fields(), plot_integral(), plot_raw_fields()
      - fit_transverse_parabolas(), plot_second_derivatives()
      - fit_second_derivative_sinusoids(), plot_second_derivative_fit()
      - curvature_series_string()
    """

    COMPONENTS = ("Bx", "By", "Bz")

    def __init__(
        self,
        file_path,
        xy_point=(0, 0),
        dx=0.001,
        dy=0.001,
        dz=0.001,
        degree=3,
        peak_window=(100, 2100),
        n_modes_x=None,
        n_modes_y=None,
        n_modes_z=None,
        x_left_slices=10,
        x_right_slices=10,
        y_left_slices=10,
        y_right_slices=10,
        z_left_slices=10,
        z_right_slices=10,
        verbose=False,
    ):
        self.file_path = file_path
        self.xy_point = xy_point
        self.dx, self.dy, self.dz = dx, dy, dz
        self.degree = degree
        self.peak_window = peak_window
        self.verbose = verbose

        self.n_modes = {"Bx": n_modes_x, "By": n_modes_y, "Bz": n_modes_z}
        self.slice_counts = {
            "Bx": {"left": x_left_slices, "right": x_right_slices},
            "By": {"left": y_left_slices, "right": y_right_slices},
            "Bz": {"left": z_left_slices, "right": z_right_slices},
        }

        self.df: Optional[pd.DataFrame] = None
        self.z_full: Optional[np.ndarray] = None
        self.fields: Dict[str, FieldChannel] = {c: FieldChannel() for c in self.COMPONENTS}

        self.primitives = {"data": {c: None for c in self.COMPONENTS},
                           "fit": {c: None for c in self.COMPONENTS}}
        self.parabola = None           # set by fit_transverse_parabolas
        self.curv_sines = {"x": {}, "y": {}}  # set by fit_second_derivative_sinusoids

    # ------------------------------ Utilities ------------------------------
    def _ensure(self, *needs: str) -> None:
        """Ensure preconditions ("data", "borders", "sines", "parabola")."""
        if "data" in needs and (self.df is None or self.z_full is None or any(self.fields[c].data is None for c in self.COMPONENTS)):
            self._parse_to_dataframe()
            self._select_xy()
        if "borders" in needs and any(self.fields[c].borders_idx is None for c in self.COMPONENTS):
            self._find_borders()
        if "sines" in needs and any(not self.fields[c].sine.is_ready() for c in self.COMPONENTS):
            self._fit_sinusoids()
        if "parabola" in needs and self.parabola is None:
            self.fit_transverse_parabolas(region="mid")

    def _mid_slice(self, comp: str) -> slice:
        i0, i1 = self.fields[comp].borders_idx
        return slice(i0, i1)

    # Magnitude helper
    def _mag(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        return np.sqrt(sum(data[c] ** 2 for c in self.COMPONENTS))

    # ------------------------------ Orchestrator ------------------------------
    def fit(
        self,
        *,
        curvature: bool = False,
        curvature_axes: tuple = ("x", "y"),
        curvature_fields: tuple = ("Bx", "By"),
        n_modes: Optional[dict] = None,
        reuse_borders: bool = True,
        center_xy: Tuple[int, int] = (0, 0),
    ) -> None:
        """Run the full field fit. If `curvature` is True, also fit 3‑point parabolas
        at `center_xy` (default (0,0)) in the mid region and sine-fit their second derivatives.

        Notes
        -----
        - If `n_modes` is provided, it now applies to **both** the base field fits (Bx, By, Bz)
          and the curvature fits. Keys should be any of {"Bx","By","Bz"}.
        """
        self._ensure("data")

        # Optional: apply per-field mode caps to base fits as well
        if n_modes is not None:
            for comp in self.COMPONENTS:
                val = n_modes.get(comp) if isinstance(n_modes, dict) else None
                if val is not None:
                    self.n_modes[comp] = val

        self._find_borders()
        self._fit_sinusoids()
        self._fit_edges()
        self._merge_sections()
        self._compute_primitives()
        if curvature:
            # Compute parabolas at requested center and fit curvatures along chosen axes
            self.fit_transverse_parabolas(center_xy=center_xy, region="mid")
            if "x" in curvature_axes:
                self.fit_second_derivative_sinusoids(axis="x", fields=curvature_fields, n_modes=n_modes, reuse_borders=reuse_borders)
            if "y" in curvature_axes:
                self.fit_second_derivative_sinusoids(axis="y", fields=curvature_fields, n_modes=n_modes, reuse_borders=reuse_borders)

    # ------------------------------ I/O ------------------------------
    def _parse_to_dataframe(self) -> None:
        df = pd.read_csv(
            self.file_path, sep=r"\s+", header=None, names=["X", "Y", "Z", "Bx", "By", "Bz"]
        )
        df.set_index(["X", "Y", "Z"], inplace=True)
        self.df = df

    def _select_xy(self) -> None:
        subset = self.df.xs(self.xy_point, level=["X", "Y"]).sort_index()
        z_idx = subset.index.to_numpy()
        self.z_full = z_idx * self.dz
        for c in self.COMPONENTS:
            self.fields[c].data = subset[c].to_numpy()

    # ------------------------------ Borders ------------------------------
    def _find_borders(self) -> None:
        z = self.z_full
        k0, k1 = self.peak_window
        def _filter(ix):
            return ix[(ix >= k0) & (ix <= k1)]

        borders = {}
        # Bx via valleys
        bx = self.fields["Bx"].data
        bx_val = _filter(find_peaks(-bx)[0])
        if len(bx_val) >= 4:
            i0x, i1x = int(bx_val[1]), int(bx_val[-2])
        elif len(bx_val) >= 2:
            i0x, i1x = int(bx_val[0]), int(bx_val[-1])
        else:
            i0x, i1x = max(1, k0), min(k1, len(z) - 1)
        i0x = max(1, min(i0x, len(z) - 2))
        i1x = max(i0x + 2, min(i1x, len(z) - 1))
        borders["Bx"] = (i0x, i1x)

        # By via peaks/valleys with fallback to Bx
        by = self.fields["By"].data
        by_pea = _filter(find_peaks(by)[0])
        by_val = _filter(find_peaks(-by)[0])
        if len(by_pea) >= 2 and len(by_val) >= 2:
            i0y, i1y = int(by_pea[1]), int(by_val[-2])
            if i0y >= i1y:
                i0y, i1y = borders["Bx"]
        else:
            i0y, i1y = borders["Bx"]
        i0y = max(1, min(i0y, len(z) - 2))
        i1y = max(i0y + 2, min(i1y, len(z) - 1))
        borders["By"] = (i0y, i1y)

        # Bz mirrors Bx with fallback to By
        bz = self.fields["Bz"].data
        bz_val = _filter(find_peaks(-bz)[0])
        if len(bz_val) >= 4:
            i0z, i1z = int(bz_val[1]), int(bz_val[-2])
        elif len(bz_val) >= 2:
            i0z, i1z = int(bz_val[0]), int(bz_val[-1])
        else:
            i0z, i1z = borders["By"]
        i0z = max(1, min(i0z, len(z) - 2))
        i1z = max(i0z + 2, min(i1z, len(z) - 1))
        borders["Bz"] = (i0z, i1z)

        for c in self.COMPONENTS:
            i0, i1 = borders[c]
            self.fields[c].borders_idx = (i0, i1)
            self.fields[c].borders_z = (z[i0], z[i1 - 1])

    # ------------------------------ Sinusoid fitting ------------------------------
    @staticmethod
    def _sinusoid_sum(s: np.ndarray, *params) -> np.ndarray:
        y = np.zeros_like(s, dtype=float)
        for A1, A2, k in zip(params[::3], params[1::3], params[2::3]):
            y += A1 * np.cos(k * s) + A2 * np.sin(k * s)
        return y

    def _find_frequencies_topN(self, s: np.ndarray, y: np.ndarray, n_modes: Optional[int]):
        yw = (y - np.mean(y)) * np.hanning(len(y))
        Y = np.fft.rfft(yw)
        f = np.fft.rfftfreq(len(y), d=np.mean(np.diff(s)))
        mag = np.abs(Y)
        start = 1
        peaks, _ = find_peaks(mag[start:])
        if peaks.size == 0:
            return np.array([]), np.array([])
        peaks = peaks + start
        if n_modes is not None and peaks.size > n_modes:
            sel = np.argpartition(mag[peaks], -n_modes)[-n_modes:]
            peaks = peaks[sel]
        order = np.argsort(mag[peaks])[::-1]
        peaks = peaks[order]
        return Y[peaks], f[peaks]

    def _fit_sine_to_segment(self, y: np.ndarray, z: np.ndarray, n_modes: Optional[int]) -> SineModel:
        """Generic multi-sinusoid fit on y(z) within the given *contiguous* segment.
        Internally recenters for conditioning, then returns a model in global z (no z0).
        """
        # Internal centering for frequency pick & non-linear fit
        zc = 0.5 * (z[0] + z[-1])
        s = z - zc
        amps, freqs = self._find_frequencies_topN(s, y, n_modes)
        if len(freqs) == 0:
            L = s[-1] - s[0]
            k_guess = 2 * np.pi / max(L / 2, 1e-6)
            params0 = np.array([y.ptp() / 2, 0.0, k_guess])
        else:
            A_cos = 2 * self.dz * amps.real
            A_sin = -2 * self.dz * amps.imag
            params0 = np.empty(3 * len(freqs))
            for i, (a1, a2, ff) in enumerate(zip(A_cos, A_sin, freqs)):
                params0[3 * i:3 * i + 3] = (a1, a2, 2 * np.pi * ff)
        lower = np.tile([-np.inf, -np.inf, 0.0], len(params0) // 3)
        upper = np.tile([np.inf, np.inf, np.inf], len(params0) // 3)
        popt, _ = curve_fit(self._sinusoid_sum, s, y, p0=params0, bounds=(lower, upper), maxfev=10000)
        m = len(popt) // 3
        # Rotate coefficients back to the global z-origin (eliminate z0):
        Acos, Asin, K = [], [], []
        for i in range(m):
            a1c, a2c, kk = popt[3*i], popt[3*i+1], popt[3*i+2]
            phi = kk * zc
            Acos.append(a1c * np.cos(phi) - a2c * np.sin(phi))
            Asin.append(a1c * np.sin(phi) + a2c * np.cos(phi))
            K.append(kk)
        return SineModel(Acos=np.array(Acos), Asin=np.array(Asin), k=np.array(K))

    def _fit_sinusoids(self) -> None:
        z = self.z_full
        for c in self.COMPONENTS:
            i0, i1 = self.fields[c].borders_idx
            model = self._fit_sine_to_segment(self.fields[c].data[i0:i1], z[i0:i1], self.n_modes[c])
            self.fields[c].sine = model

    # ------------------------------ Edge polynomials ------------------------------
    def _boundary_from_sine(self, z_mid: np.ndarray, sine: SineModel) -> np.ndarray:
        xL, xR = z_mid[0] - self.dz, z_mid[-1] + self.dz
        fL = fR = dL = dR = ddL = ddR = 0.0
        for A1, A2, k in zip(sine.Acos, sine.Asin, sine.k):
            fL += A1*np.cos(k*xL) + A2*np.sin(k*xL)
            fR += A1*np.cos(k*xR) + A2*np.sin(k*xR)
            dL += k*(A2*np.cos(k*xL) - A1*np.sin(k*xL))
            dR += k*(A2*np.cos(k*xR) - A1*np.sin(k*xR))
            ddL += -k*k*(A1*np.cos(k*xL) + A2*np.sin(k*xL))
            ddR += -k*k*(A1*np.cos(k*xR) + A2*np.sin(k*xR))
        return np.array([fL, fR, dL, dR, ddL, ddR], dtype=float)

    def _boundary_from_poly(self, z_prev: np.ndarray, poly: Polynomial) -> np.ndarray:
        xL, xR = z_prev[0] - self.dz, z_prev[-1] + self.dz
        dp, ddp = poly.deriv(), poly.deriv(2)
        return np.array([poly(xL), poly(xR), dp(xL), dp(xR), ddp(xL), ddp(xR)], dtype=float)

    @staticmethod
    def _balanced_slices(n: int, num_regions: int) -> list[slice]:
        base, rem = n // num_regions, n % num_regions
        slices, start = [], 0
        for i in range(num_regions):
            end = start + base + (1 if i < rem else 0)
            if end > start:
                slices.append(slice(start, end))
            start = end
        return slices

    def _fit_poly_side(self, z_region, b_region, z_sines, sine, num_slices, left_side):
        deg = self.degree
        slices = self._balanced_slices(len(z_region), num_slices)
        if left_side:
            slices = list(reversed(slices))
        fit_reg = np.zeros_like(z_region, dtype=float)
        pieces = []
        prev_poly = None
        prev_z = None
        for ix, s in enumerate(slices):
            z_this, b_this = z_region[s], b_region[s]
            boundaries = self._boundary_from_sine(z_sines, sine) if ix == 0 else self._boundary_from_poly(prev_z, prev_poly)
            if deg >= 5:
                if left_side:
                    dbL = (-3 * b_this[0] + 4 * b_this[1] - b_this[2]) / (2 * self.dz)
                    d2L = (2 * b_this[0] - 5 * b_this[1] + 4 * b_this[2] - b_this[3]) / (self.dz ** 2)
                    coeffs = poly_fit.poly_fit(N=deg, xdata=z_this, ydata=b_this,
                                               x0=[z_this[0], z_this[-1]], y0=[b_this[0], boundaries[0]],
                                               xp0=[z_this[0], z_this[-1]], yp0=[dbL, boundaries[2]],
                                               xpp0=[z_this[0], z_this[-1]], ypp0=[d2L, boundaries[4]])
                else:
                    dbR = (3 * b_this[-1] - 4 * b_this[-2] + b_this[-3]) / (2 * self.dz)
                    d2R = (2 * b_this[-1] - 5 * b_this[-2] + 4 * b_this[-3] - b_this[-4]) / (self.dz ** 2)
                    coeffs = poly_fit.poly_fit(N=deg, xdata=z_this, ydata=b_this,
                                               x0=[z_this[0], z_this[-1]], y0=[boundaries[1], b_this[-1]],
                                               xp0=[z_this[0], z_this[-1]], yp0=[boundaries[3], dbR],
                                               xpp0=[z_this[0], z_this[-1]], ypp0=[boundaries[5], d2R])
            elif deg >= 3:
                if left_side:
                    dbL = (-3 * b_this[0] + 4 * b_this[1] - b_this[2]) / (2 * self.dz)
                    coeffs = poly_fit.poly_fit(N=deg, xdata=z_this, ydata=b_this,
                                               x0=[z_this[0], z_this[-1]], y0=[b_this[0], boundaries[0]],
                                               xp0=[z_this[0], z_this[-1]], yp0=[dbL, boundaries[2]])
                else:
                    dbR = (3 * b_this[-1] - 4 * b_this[-2] + b_this[-3]) / (2 * self.dz)
                    coeffs = poly_fit.poly_fit(N=deg, xdata=z_this, ydata=b_this,
                                               x0=[z_this[0], z_this[-1]], y0=[boundaries[1], b_this[-1]],
                                               xp0=[z_this[0], z_this[-1]], yp0=[boundaries[3], dbR])
            else:
                if left_side:
                    coeffs = poly_fit.poly_fit(N=deg, xdata=z_this, ydata=b_this,
                                               x0=[z_this[0], z_this[-1]], y0=[b_this[0], boundaries[0]])
                else:
                    coeffs = poly_fit.poly_fit(N=deg, xdata=z_this, ydata=b_this,
                                               x0=[z_this[0], z_this[-1]], y0=[boundaries[1], b_this[-1]])
            poly = Polynomial(coeffs)
            fit_reg[s] = poly(z_this)
            prev_poly, prev_z = poly, z_this
            pieces.append((s.start, poly))
        return fit_reg, pieces

    def _fit_edges(self) -> None:
        z = self.z_full
        for c in self.COMPONENTS:
            ch = self.fields[c]
            i0, i1 = ch.borders_idx
            z_left, b_left = z[:i0], ch.data[:i0]
            z_right, b_right = z[i1:], ch.data[i1:]
            z_mid = z[i0:i1]
            ns_left, ns_right = self.slice_counts[c]["left"], self.slice_counts[c]["right"]
            fitL, piecesL = self._fit_poly_side(z_left, b_left, z_mid, ch.sine, ns_left, True)
            fitR, piecesR = self._fit_poly_side(z_right, b_right, z_mid, ch.sine, ns_right, False)
            ch.left.set_pieces([p[0] for p in piecesL], [p[1] for p in piecesL])
            ch.right.set_pieces([i1 + p[0] for p in piecesR], [p[1] for p in piecesR])
            ch.fit = np.zeros_like(ch.data, dtype=float)
            ch.fit[:i0] = fitL
            ch.fit[i1:] = fitR

    # ------------------------------ Merge & eval ------------------------------
    def _merge_sections(self) -> None:
        z = self.z_full
        for c in self.COMPONENTS:
            ch = self.fields[c]
            i0, i1 = ch.borders_idx
            mid_mask = (z >= z[i0]) & (z <= z[i1 - 1])
            ch.fit[mid_mask] = ch.sine.eval(z[mid_mask])

    def evaluate(self, zq: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for c in self.COMPONENTS:
            ch = self.fields[c]
            i0, i1 = ch.borders_idx
            z0, z1 = self.z_full[i0], self.z_full[i1 - 1]
            left_mask = zq < z0
            mid_mask = (zq >= z0) & (zq <= z1)
            right_mask = zq > z1
            y = np.empty_like(zq, dtype=float)
            y[left_mask] = ch.left.eval(zq[left_mask], self.z_full)
            y[mid_mask] = ch.sine.eval(zq[mid_mask])
            y[right_mask] = ch.right.eval(zq[right_mask], self.z_full)
            out[c] = y
        return out

    # ------------------------------ Primitives & plotting ------------------------------
    def _compute_primitives(self) -> None:
        z = self.z_full
        for c in self.COMPONENTS:
            if self.fields[c].data is not None:
                self.primitives["data"][c] = sc.integrate.cumulative_trapezoid(self.fields[c].data, x=z, initial=0.0)
            if self.fields[c].fit is not None:
                self.primitives["fit"][c] = sc.integrate.cumulative_trapezoid(self.fields[c].fit, x=z, initial=0.0)

    def plot_fields(self) -> None:
        z = self.z_full
        fig, axes = plt.subplots(4, sharex=True, figsize=(9, 9))
        for name, ax in zip(self.COMPONENTS, axes[:3]):
            ch = self.fields[name]
            ax.plot(z, ch.data, label=f"{name} data")
            ax.plot(z, ch.fit, label=f"{name} fit")
            i0, i1 = ch.borders_idx
            ax.axvline(z[i0], color="k", linestyle="--", linewidth=1)
            ax.axvline(z[i1 - 1], color="k", linestyle="--", linewidth=1)
            ax.set_ylabel(f"{name} [T]")
            ax.grid(True)
            ax.legend(loc="upper right")
        # magnitude
        data_map = {c: self.fields[c].data for c in self.COMPONENTS}
        fit_map = {c: self.fields[c].fit for c in self.COMPONENTS}
        axes[3].plot(z, self._mag(data_map), label="|B| data")
        axes[3].plot(z, self._mag(fit_map), label="|B| fit")
        axes[3].set_xlabel("s [m]")
        axes[3].set_ylabel("|B| [T]")
        axes[3].grid(True)
        axes[3].legend(loc="upper right")
        fig.suptitle(f"Magnetic Field at (X, Y) = {self.xy_point}")
        plt.tight_layout()
        plt.show()

    def plot_integral(self) -> None:
        if any(self.primitives["data"][k] is None or self.primitives["fit"][k] is None for k in self.COMPONENTS):
            self._compute_primitives()
        z = self.z_full
        fig, axes = plt.subplots(3, sharex=True, figsize=(9, 8))
        for name, ax in zip(self.COMPONENTS, axes):
            ax.plot(z, self.primitives["data"][name], label=fr"$\int {name}\,ds$ data")
            ax.plot(z, self.primitives["fit"][name], label=fr"$\int {name}\,ds$ fit")
            i0, i1 = self.fields[name].borders_idx
            ax.axvline(z[i0], color="k", linestyle="--", linewidth=1)
            ax.axvline(z[i1 - 1], color="k", linestyle="--", linewidth=1)
            ax.set_ylabel(fr"$\int {name}\,ds$ [Tm]")
            ax.grid(True)
            ax.legend(loc="best")
        axes[-1].set_xlabel("s [m]")
        fig.suptitle(f"Cumulative integrals at (X, Y) = {self.xy_point}")
        plt.tight_layout()
        plt.show()

    def plot_raw_fields(self, zlim=None, show_magnitude=False, channels=("Bx", "By", "Bz"), figsize=(9, 8)):
        self._ensure("data")
        to_plot = [(k, self.fields[k].data) for k in channels]
        if show_magnitude:
            data_map = {c: self.fields[c].data for c in self.COMPONENTS}
            to_plot.append(("|B|", self._mag(data_map)))
        nrows = len(to_plot)
        fig, axes = plt.subplots(nrows, 1, sharex=True, figsize=figsize)
        axes = [axes] if nrows == 1 else axes
        z = self.z_full
        for ax, (label, y) in zip(axes, to_plot):
            ax.plot(z, y, label=f"{label} data")
            ax.set_ylabel(f"{label} [T]" if label != "|B|" else "|B| [T]")
            ax.grid(True)
            ax.legend(loc="best")
        if zlim is not None:
            axes[-1].set_xlim(*zlim)
        axes[-1].set_xlabel("s [m]")
        fig.suptitle(f"Raw magnetic field at (X, Y) = {self.xy_point}")
        fig.tight_layout()
        return fig, axes

    # ------------------------------ Parabolas & curvature ------------------------------
    def fit_transverse_parabolas(self, *, enforce_even_x=True, enforce_even_y=True, dx=None, dy=None, region="mid", center_xy: Tuple[int, int] = (0, 0)):
        """Fit 3-point parabolas along x and y per z for Bx, By.
        Evaluated at the transverse *center* `center_xy` (default (0,0)), not at `self.xy_point`.
        Stores arrays aligned to z in self.parabola with NaNs outside the chosen region.
        region: "mid" (default) fits only inside each channel's mid borders; "all" fits everywhere.
        """
        self._ensure("data", "borders")
        if center_xy is None:
            x0, y0 = self.xy_point
        else:
            x0, y0 = center_xy
        dx = self.dx if dx is None else dx
        dy = self.dy if dy is None else dy
        def line(field, x=None, y=None):
            ser = self.df.xs((x, y), level=["X", "Y"])[field].sort_index()
            return ser.to_numpy()
        # X-lines at fixed y0
        Bx_xm, Bx_x0, Bx_xp = line("Bx", x=x0 - 1, y=y0), line("Bx", x=x0, y=y0), line("Bx", x=x0 + 1, y=y0)
        By_xm, By_x0, By_xp = line("By", x=x0 - 1, y=y0), line("By", x=x0, y=y0), line("By", x=x0 + 1, y=y0)
        # Y-lines at fixed x0
        Bx_ym, Bx_y0, Bx_yp = line("Bx", x=x0, y=y0 - 1), line("Bx", x=x0, y=y0), line("Bx", x=x0, y=y0 + 1)
        By_ym, By_y0, By_yp = line("By", x=x0, y=y0 - 1), line("By", x=x0, y=y0), line("By", x=x0, y=y0 + 1)
        nZ = len(self.z_full)
        def quad_coeffs(fm, f0, fp, d):
            a = (fp - 2.0 * f0 + fm) / (2.0 * d * d)
            b = (fp - fm) / (2.0 * d)
            c = f0
            return a, b, c
        def alloc():
            return (np.full(nZ, np.nan), np.full(nZ, np.nan), np.full(nZ, np.nan))
        ax_Bx, bx_Bx, cx_Bx = alloc(); ax_By, bx_By, cx_By = alloc()
        ay_Bx, by_Bx, cy_Bx = alloc(); ay_By, by_By, cy_By = alloc()
        mid_mask_Bx, mid_mask_By = np.zeros(nZ, bool), np.zeros(nZ, bool)
        i0x, i1x = self.fields["Bx"].borders_idx
        i0y, i1y = self.fields["By"].borders_idx
        if region == "mid":
            mid_mask_Bx[i0x:i1x] = True
            mid_mask_By[i0y:i1y] = True
        elif region == "all":
            mid_mask_Bx[:] = True; mid_mask_By[:] = True
        else:
            raise ValueError('region must be "mid" or "all"')
        ax_Bx[mid_mask_Bx], bx_Bx[mid_mask_Bx], cx_Bx[mid_mask_Bx] = quad_coeffs(Bx_xm[mid_mask_Bx], Bx_x0[mid_mask_Bx], Bx_xp[mid_mask_Bx], dx)
        ax_By[mid_mask_Bx], bx_By[mid_mask_Bx], cx_By[mid_mask_Bx] = quad_coeffs(By_xm[mid_mask_Bx], By_x0[mid_mask_Bx], By_xp[mid_mask_Bx], dx)
        ay_Bx[mid_mask_By], by_Bx[mid_mask_By], cy_Bx[mid_mask_By] = quad_coeffs(Bx_ym[mid_mask_By], Bx_y0[mid_mask_By], Bx_yp[mid_mask_By], dy)
        ay_By[mid_mask_By], by_By[mid_mask_By], cy_By[mid_mask_By] = quad_coeffs(By_ym[mid_mask_By], By_y0[mid_mask_By], By_yp[mid_mask_By], dy)
        if enforce_even_x:
            bx_Bx[mid_mask_Bx] = 0.0; bx_By[mid_mask_Bx] = 0.0
        if enforce_even_y:
            by_Bx[mid_mask_By] = 0.0; by_By[mid_mask_By] = 0.0
        self.parabola = {
            "Bx": {"ax": ax_Bx, "bx": bx_Bx, "cx": cx_Bx, "ay": ay_Bx, "by": by_Bx, "cy": cy_Bx,
                    "dBx_dx": bx_Bx, "d2Bx_dx2": 2.0 * ax_Bx, "dBx_dy": by_Bx, "d2Bx_dy2": 2.0 * ay_Bx,
                    "_mask_x": mid_mask_Bx, "_mask_y": mid_mask_By},
            "By": {"ax": ax_By, "bx": bx_By, "cx": cx_By, "ay": ay_By, "by": by_By, "cy": cy_By,
                    "dBy_dx": bx_By, "d2By_dx2": 2.0 * ax_By, "dBy_dy": by_By, "d2By_dy2": 2.0 * ay_By,
                    "_mask_x": mid_mask_Bx, "_mask_y": mid_mask_By},
        }
        return self.parabola

    def plot_second_derivatives(self, fields=("Bx", "By"), directions=("x", "y"), smooth=None, **parabola_kwargs):
        if self.parabola is None:
            self.fit_transverse_parabolas(**parabola_kwargs)
        z = self.z_full
        curves, labels = [], []
        for fld in fields:
            if "x" in directions:
                arr = self.parabola[fld][f"d2{fld}_dx2"]; curves.append(arr); labels.append(fr"$\partial^2 {fld}/\partial x^2$")
            if "y" in directions:
                arr = self.parabola[fld][f"d2{fld}_dy2"]; curves.append(arr); labels.append(fr"$\partial^2 {fld}/\partial y^2$")
        if smooth is not None:
            wlen, pord = smooth
            for i, arr in enumerate(curves):
                mask = np.isfinite(arr)
                if mask.sum() >= max(wlen, pord + 2):
                    tmp = arr.copy(); tmp[mask] = savgol_filter(arr[mask], wlen, pord); curves[i] = tmp
        plt.figure(figsize=(9, 4))
        for arr, lab in zip(curves, labels):
            plt.plot(z, arr, label=lab)
        plt.xlabel("s [m]")
        plt.ylabel(r"$\partial^2 B/\partial(\cdot)^2$ [T/m$^2$]")
        plt.grid(True)
        plt.legend(loc="best")
        plt.title(f"Transverse second derivatives at (X, Y) = {self.xy_point}")
        plt.tight_layout(); plt.show()

    def fit_second_derivative_sinusoids(self, axis: str = "x", fields=("Bx", "By"), n_modes: Optional[dict] = None, reuse_borders: bool = True, verbose: Optional[bool] = None, method: str = "locked"):
        """Fit sinusoids to curvature traces d²(field)/d{axis}² vs s.

        Parameters
        ----------
        axis : {"x","y"}
            Which transverse curvature to use.
        fields : iterable
            Subset of {"Bx","By"}.
        n_modes : dict|None
            Per-field cap on number of modes. If None, uses self.n_modes[field].
        reuse_borders : bool
            Use each field's mid borders as the fit window.
        method : {"locked","nonlinear","auto"}
            - "locked" (default): reuse *frequencies* from the already-fitted field sine, and solve only
              for amplitudes by linear least squares on cos(k z), sin(k z). No phase origin.
            - "nonlinear": generic non-linear fit (old behavior, but returns no z0).
            - "auto": try locked first, fall back to nonlinear.
        """
        if verbose is None:
            verbose = self.verbose
        self._ensure("data", "borders", "sines", "parabola")
        z = self.z_full

        def _window(curv, fld):
            if reuse_borders:
                i0, i1 = self.fields[fld].borders_idx
            else:
                k0, k1 = self.peak_window
                i0 = max(1, k0); i1 = max(i0 + 2, min(k1, len(z) - 1))
            z_seg, y_seg = z[i0:i1], curv[i0:i1]
            m = np.isfinite(y_seg)
            return z_seg[m], y_seg[m]

        def _fit_locked(fld, z_used, y_used, nm_cap):
            base = self.fields[fld].sine
            if not base.is_ready() or base.k.size == 0:
                raise RuntimeError("base sine not available for locked-mode fit")
            k_list = np.array(base.k, float)
            if nm_cap is not None and len(k_list) > nm_cap:
                order = np.argsort(k_list)
                k_list = k_list[order[:nm_cap]]
            cols = []
            for kk in k_list:
                cols.append(np.cos(kk * z_used))
                cols.append(np.sin(kk * z_used))
            X = np.column_stack(cols)
            coef, *_ = np.linalg.lstsq(X, y_used, rcond=None)
            Acos = coef[0::2]; Asin = coef[1::2]
            return SineModel(Acos=Acos, Asin=Asin, k=k_list)

        def _fit_nonlinear(fld, z_used, y_used, nm_cap):
            nm = nm_cap if nm_cap is not None else self.n_modes.get(fld)
            return self._fit_sine_to_segment(y_used, z_used, nm)

        for fld in fields:
            curv = self.parabola[fld][f"d2{fld}_d{axis}2"]
            z_used, y_used = _window(curv, fld)
            if y_used.size < 3:
                raise ValueError(f"Not enough finite curvature points to fit {fld} along {axis}.")
            nm_cap = None if n_modes is None else n_modes.get(fld)
            model = None
            if method in ("locked", "auto"):
                try:
                    model = _fit_locked(fld, z_used, y_used, nm_cap)
                except Exception:
                    if method != "auto":
                        raise
            if model is None:
                model = _fit_nonlinear(fld, z_used, y_used, nm_cap)
            self.curv_sines[axis][fld] = model
            if verbose:
                print(f"[curv {axis}] {fld}: fitted {len(model.k)} modes (method={method})")
        return self.curv_sines[axis]

        def _fit_nonlinear(fld, z_used, y_used, nm_cap):
            nm = nm_cap if nm_cap is not None else self.n_modes.get(fld)
            return self._fit_sine_to_segment(y_used, z_used, nm)

        for fld in fields:
            # pick series
            curv = self.parabola[fld][f"d2{fld}_d{axis}2"]
            z_used, y_used = _window(curv, fld)
            if y_used.size < 3:
                raise ValueError(f"Not enough finite curvature points to fit {fld} along {axis}.")
            nm_cap = None if n_modes is None else n_modes.get(fld)

            model = None
            if method in ("locked", "auto"):
                try:
                    model = _fit_locked(fld, z_used, y_used, nm_cap)
                except Exception as e:
                    if method == "auto":
                        model = None
                    else:
                        raise
            if model is None:  # nonlinear or auto fallback
                model = _fit_nonlinear(fld, z_used, y_used, nm_cap)

            self.curv_sines[axis][fld] = model
            if verbose:
                print(f"[curv {axis}] {fld}: fitted {len(model.k)} modes (method={method}); z0={model.z0:.6g}")
        return self.curv_sines[axis]

    def plot_second_derivative_fit(self, field: str = "Bx", axis: str = "x", show_series_string: bool = False, **series_kwargs):
        if self.parabola is None:
            self.fit_transverse_parabolas()
        if field not in self.curv_sines.get(axis, {}):
            self.fit_second_derivative_sinusoids(axis=axis, fields=(field,))
        z = self.z_full
        curv = self.parabola[field][f"d2{field}_d{axis}2"]
        model = self.curv_sines[axis][field]
        i0, i1 = self.fields[field].borders_idx
        plt.figure(figsize=(9, 4))
        plt.plot(z, curv, label=f"data: d²{field}/d{axis}²")
        plt.plot(z, model.eval(z), label="sine fit")
        plt.axvline(z[i0], color="k", linestyle="--", linewidth=1)
        plt.axvline(z[i1 - 1], color="k", linestyle="--", linewidth=1)
        plt.xlabel("s [m]")
        plt.ylabel(r"T/m$^2$")
        plt.grid(True)
        plt.legend(loc="best")
        plt.title(f"Curvature fit for {field} along {axis} at (X,Y)={self.xy_point}")
        plt.tight_layout(); plt.show()
        if show_series_string:
            print(model.series_string(in_terms_of="z", **series_kwargs))

    def curvature_series_string(self, field: str = "Bx", axis: str = "x", *, ensure_fit: bool = True, **fmt_kwargs) -> str:
        if ensure_fit and (self.parabola is None or field not in self.curv_sines.get(axis, {})):
            self.fit_second_derivative_sinusoids(axis=axis, fields=(field,))
        return self.curv_sines[axis][field].series_string(**fmt_kwargs)

    # ------------------------------ Tuning ------------------------------
    def tune_slices_for_zero_integral(self, field="all", left_candidates=None, right_candidates=None, tradeoff_mse=0.0, verbose=None):
        if verbose is None:
            verbose = self.verbose
        self._ensure("data", "borders", "sines")
        def _score(y_fit, y_data, z):
            I = sc.integrate.cumulative_trapezoid(y_fit, x=z, initial=0.0)
            I_end = I[-1]
            if tradeoff_mse:
                mse = np.mean((y_fit - y_data) ** 2)
                return abs(I_end) + tradeoff_mse * mse, I_end, mse
            return abs(I_end), I_end, None
        def _tune_one(name: str):
            ch = self.fields[name]; z = self.z_full; i0, i1 = ch.borders_idx
            z_left, b_left, z_right, b_right = z[:i0], ch.data[:i0], z[i1:], ch.data[i1:]
            z_mid = z[i0:i1]
            Ls = range(max(2, self.slice_counts[name]["left"] - 8), self.slice_counts[name]["left"] + 9) if left_candidates is None else list(left_candidates)
            Rs = range(max(2, self.slice_counts[name]["right"] - 8), self.slice_counts[name]["right"] + 9) if right_candidates is None else list(right_candidates)
            best = None
            for L in Ls:
                for R in Rs:
                    fitL, piecesL = self._fit_poly_side(z_left, b_left, z_mid, ch.sine, L, True)
                    fitR, piecesR = self._fit_poly_side(z_right, b_right, z_mid, ch.sine, R, False)
                    y_fit = np.zeros_like(ch.data, dtype=float)
                    y_fit[:i0] = fitL; y_fit[i1:] = fitR
                    mid_mask = (z >= z[i0]) & (z <= z[i1 - 1])
                    y_fit[mid_mask] = ch.sine.eval(z[mid_mask])
                    score, I_end, mse = _score(y_fit, ch.data, z)
                    if best is None or score < best["score"]:
                        best = {"score": score, "I_end": I_end, "mse": mse, "L": L, "R": R,
                                "fit": y_fit, "piecesL": piecesL, "piecesR": piecesR}
            if verbose:
                msg = f"[{name}] best slices: left={best['L']} right={best['R']} |I_end|={abs(best['I_end']):.3e}"
                if tradeoff_mse and best["mse"] is not None:
                    msg += f"  mse={best['mse']:.3e}"
                print(msg)
            ch.fit = best["fit"]
            ch.left.set_pieces([p[0] for p in best["piecesL"]], [p[1] for p in best["piecesL"]])
            ch.right.set_pieces([i1 + p[0] for p in best["piecesR"]], [p[1] for p in best["piecesR"]])
            self.slice_counts[name]["left"], self.slice_counts[name]["right"] = best["L"], best["R"]
        targets = [field] if field in self.COMPONENTS else (["Bx", "By"] if field == "both" else list(self.COMPONENTS))
        for name in targets:
            _tune_one(name)
        self._merge_sections(); self._compute_primitives()

    # ------------------------------ Results ------------------------------
    def results(self) -> Dict[str, object]:
        return {"z": self.z_full, **{c: self.fields[c] for c in self.COMPONENTS}, "primitives": self.primitives}
