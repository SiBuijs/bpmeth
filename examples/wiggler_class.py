# --- deps ---
import numpy as np
import pandas as pd
import scipy as sc
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
from bpmeth import poly_fit


# ===== Small helper classes (no type hints in function signatures) =====

class SineModel:
    def __init__(self):
        self.Acos = None   # np.ndarray or None
        self.Asin = None   # np.ndarray or None
        self.k    = None   # np.ndarray or None
        self.z0   = 0.0

    def is_ready(self):
        return self.Acos is not None and self.Asin is not None and self.k is not None

    def eval(self, z):
        if not self.is_ready():
            return np.zeros_like(z, dtype=float)
        s = z - self.z0
        y = np.zeros_like(z, dtype=float)
        for A1, A2, kk in zip(self.Acos, self.Asin, self.k):
            y += A1 * np.cos(kk * s) + A2 * np.sin(kk * s)
        return y


class PiecewisePoly:
    def __init__(self):
        self.starts = np.empty(0, dtype=int)  # absolute start indices in z_full
        self.polys  = []                      # list of Polynomial

    def set_pieces(self, starts, polys):
        order = np.argsort(starts)
        self.starts = np.asarray(starts, dtype=int)[order]
        self.polys  = [polys[i] for i in order]

    def eval(self, z, z_full):
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


class FieldChannel:
    def __init__(self):
        self.data = None
        self.fit  = None
        self.borders_idx = None   # (i0, i1) -> mid region is [i0:i1)
        self.borders_z   = None   # (z0, z1)
        self.sine  = SineModel()
        self.left  = PiecewisePoly()
        self.right = PiecewisePoly()


# ============================== Main class ==============================

class Wiggler:
    def __init__(
        self,
        file_path,
        xy_point=(0, 0),
        dz=0.001,
        degree=3,
        peak_window=(100, 2100),
        n_modes_x=None,
        n_modes_y=None,
        x_left_slices=10,
        x_right_slices=10,
        y_left_slices=10,
        y_right_slices=10,
        verbose=False,
    ):
        self.file_path = file_path
        self.xy_point  = xy_point
        self.dz        = dz
        self.degree    = degree
        self.peak_window = peak_window
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y
        self.slice_counts = {
            "Bx": {"left": x_left_slices, "right": x_right_slices},
            "By": {"left": y_left_slices, "right": y_right_slices},
        }
        self.verbose = verbose

        self.df = None
        self.z_full = None
        self.fields = {"Bx": FieldChannel(), "By": FieldChannel()}

        self.primitives = {
            "data": {"Bx": None, "By": None},
            "fit":  {"Bx": None, "By": None},
        }

    # ---------------- Orchestrator ----------------
    def fit(self):
        self._parse_to_dataframe()
        self._select_xy()
        self._find_borders()
        self._fit_sinusoids()
        self._fit_edges()
        self._merge_sections()
        self._compute_primitives()

    def tune_slices_for_zero_integral(
            self,
            field="both",  # "Bx", "By", or "both"
            left_candidates=None,  # iterable of ints; default = range based on current counts
            right_candidates=None,  # iterable of ints
            tradeoff_mse=0.0,  # optional: add penalty * MSE to avoid silly overfits
            verbose=None,
    ):
        """
        Grid-search the number of polynomial slices on each edge to minimize |∫ B ds| at z_rightmost.
        Keeps borders and sinusoid fixed; re-fits edges for each (L,R) pair.
        Updates channel pieces to the best configuration and recomputes primitives.

        Parameters
        ----------
        field : "Bx" | "By" | "both"
            Which channels to tune.
        left_candidates, right_candidates : iterable[int] or None
            Numbers of slices to try on the left/right. If None, use a small neighborhood
            around current values, e.g. current±8 with step 1.
        tradeoff_mse : float
            Optional weight to add mean squared error to the score: score = |I_end| + tradeoff_mse*MSE.
            Use 0.0 to only target the integral.
        """
        if verbose is None:
            verbose = self.verbose

        # Ensure we have data, borders, and sinusoid already
        if self.z_full is None or not self.fields["Bx"].sine.is_ready() or not self.fields["By"].sine.is_ready():
            # run the upstream steps once
            self._parse_to_dataframe()
            self._select_xy()
            self._find_borders()
            self._fit_sinusoids()

        def _score_fit(y_fit, y_data, z):
            # cumulative_trapezoid with initial=0.0 aligns with z
            I = sc.integrate.cumulative_trapezoid(y_fit, x=z, initial=0.0)
            I_end = I[-1]
            if tradeoff_mse:
                mse = np.mean((y_fit - y_data) ** 2)
                return abs(I_end) + tradeoff_mse * mse, I_end, mse
            return abs(I_end), I_end, None

        def _tune_one(name):
            ch = self.fields[name]
            z = self.z_full
            i0, i1 = ch.borders_idx
            z_left, b_left = z[:i0], ch.data[:i0]
            z_right, b_right = z[i1:], ch.data[i1:]
            z_mid = z[i0:i1]

            # default candidate grids: sweep around current values
            if left_candidates is None:
                L0 = self.slice_counts[name]["left"]
                Ls = range(max(2, L0 - 8), L0 + 9)  # inclusive neighborhood
            else:
                Ls = list(left_candidates)
            if right_candidates is None:
                R0 = self.slice_counts[name]["right"]
                Rs = range(max(2, R0 - 8), R0 + 9)
            else:
                Rs = list(right_candidates)

            best = None

            for L in Ls:
                for R in Rs:
                    # Fit edges for this (L,R) quickly (do not mutate the class yet)
                    fitL, piecesL = self._fit_poly_side(
                        z_region=z_left, b_region=b_left,
                        z_sines=z_mid, sine=ch.sine,
                        num_slices=L, left_side=True,
                    )
                    fitR, piecesR = self._fit_poly_side(
                        z_region=z_right, b_region=b_right,
                        z_sines=z_mid, sine=ch.sine,
                        num_slices=R, left_side=False,
                    )

                    # Compose a temporary stitched fit for scoring
                    y_fit = np.zeros_like(ch.data, dtype=float)
                    y_fit[:i0] = fitL
                    y_fit[i1:] = fitR
                    mid_mask = (z >= z[i0]) & (z <= z[i1 - 1])
                    y_fit[mid_mask] = ch.sine.eval(z[mid_mask])

                    score, I_end, mse = _score_fit(y_fit, ch.data, z)

                    if (best is None) or (score < best["score"]):
                        best = {
                            "score": score,
                            "I_end": I_end,
                            "mse": mse,
                            "L": L, "R": R,
                            "fit": y_fit,
                            "piecesL": piecesL,
                            "piecesR": piecesR,
                        }

            # Apply best configuration to the channel
            if verbose:
                msg = f"[{name}] best slices: left={best['L']} right={best['R']} |I_end|={abs(best['I_end']):.3e}"
                if tradeoff_mse and best["mse"] is not None:
                    msg += f"  mse={best['mse']:.3e}"
                print(msg)

            ch.fit = best["fit"]
            # store pieces with absolute starts
            ch.left.set_pieces(
                starts=[p[0] for p in best["piecesL"]],
                polys=[p[1] for p in best["piecesL"]],
            )
            ch.right.set_pieces(
                starts=[i1 + p[0] for p in best["piecesR"]],
                polys=[p[1] for p in best["piecesR"]],
            )
            # remember counts
            self.slice_counts[name]["left"] = best["L"]
            self.slice_counts[name]["right"] = best["R"]

        # Run for the requested channels
        if field in ("Bx", "both"):
            _tune_one("Bx")
        if field in ("By", "both"):
            _tune_one("By")

        # Rebuild primitives after updates (and ensure middle matches sine exactly)
        self._merge_sections()
        self._compute_primitives()

    # ---------------- I/O ----------------
    def _parse_to_dataframe(self):
        df = pd.read_csv(
            self.file_path,
            sep=r"\s+",
            header=None,
            names=["X", "Y", "Z", "Bx", "By", "Bz"],
        )
        df.set_index(["X", "Y", "Z"], inplace=True)
        self.df = df

    def _select_xy(self):
        subsetz = self.df.xs(self.xy_point, level=["X", "Y"])
        z_idx = subsetz.index.to_numpy()
        self.z_full = z_idx * self.dz
        self.fields["Bx"].data = subsetz["Bx"].to_numpy()
        self.fields["By"].data = subsetz["By"].to_numpy()

        dzs = np.diff(self.z_full)
        #if not np.allclose(dzs, dzs[0], rtol=1e-6, atol=1e-12) and self.verbose:
        #    print("Warning: z grid not perfectly uniform; using mean spacing in FFT.")

    # ---------------- Borders ----------------
    def _find_borders(self):
        bx = self.fields["Bx"].data
        by = self.fields["By"].data
        z  = self.z_full

        k0, k1 = self.peak_window
        def _filter_to_window(arr):
            return arr[(arr >= k0) & (arr <= k1)]

        bx_peaks   = _filter_to_window(find_peaks(bx)[0])
        bx_valleys = _filter_to_window(find_peaks(-bx)[0])
        by_peaks   = _filter_to_window(find_peaks(by)[0])
        by_valleys = _filter_to_window(find_peaks(-by)[0])

        if self.verbose:
            print(f"Bx peaks: {bx_peaks[:8]} ...")
            print(f"Bx valleys: {bx_valleys[:8]} ...")
            print(f"By peaks: {by_peaks[:8]} ...")
            print(f"By valleys: {by_valleys[:8]} ...")

        # Heuristics like your script
        if len(bx_valleys) >= 4:
            i0x, i1x = int(bx_valleys[1]), int(bx_valleys[-2])
        elif len(bx_valleys) >= 2:
            i0x, i1x = int(bx_valleys[0]), int(bx_valleys[-1])
        else:
            i0x, i1x = max(1, k0), min(k1, len(z) - 1)

        if len(by_peaks) >= 2 and len(by_valleys) >= 2:
            i0y, i1y = int(by_peaks[1]), int(by_valleys[-2])
            if i0y >= i1y:
                i0y, i1y = i0x, i1x
        else:
            i0y, i1y = i0x, i1x

        i0x = max(1, min(i0x, len(z) - 2))
        i1x = max(i0x + 2, min(i1x, len(z) - 1))
        i0y = max(1, min(i0y, len(z) - 2))
        i1y = max(i0y + 2, min(i1y, len(z) - 1))

        self.fields["Bx"].borders_idx = (i0x, i1x)
        self.fields["Bx"].borders_z   = (z[i0x], z[i1x - 1])
        self.fields["By"].borders_idx = (i0y, i1y)
        self.fields["By"].borders_z   = (z[i0y], z[i1y - 1])

    # ---------------- Sinusoid fitting ----------------
    @staticmethod
    def _sinusoid_sum(s, *params):
        y = np.zeros_like(s, dtype=float)
        for A1, A2, k in zip(params[::3], params[1::3], params[2::3]):
            y += A1 * np.cos(k * s) + A2 * np.sin(k * s)
        return y

    def _find_frequencies_topN(self, z, y, n_modes):
        """
        RFFT of y(z). Pick local-max peaks in |Y| (excluding DC), then keep the top-N by magnitude.
        Returns (complex_amps_at_peaks, freqs_hz_at_peaks).
        """
        yw = (y - np.mean(y)) * np.hanning(len(y))
        Y = np.fft.rfft(yw)
        f = np.fft.rfftfreq(len(y), d=np.mean(np.diff(z)))
        mag = np.abs(Y)

        # exclude DC bin
        start = 1
        peaks, _ = find_peaks(mag[start:])
        if peaks.size == 0:
            return np.array([]), np.array([])

        peaks = peaks + start

        if n_modes is not None and peaks.size > n_modes:
            sel = np.argpartition(mag[peaks], -n_modes)[-n_modes:]
            peaks = peaks[sel]

        # sort by descending magnitude for reproducibility
        order = np.argsort(mag[peaks])[::-1]
        peaks = peaks[order]

        return Y[peaks], f[peaks]

    def _fit_sinusoids(self):
        z = self.z_full

        def _fit_channel(name, n_modes):
            ch = self.fields[name]
            i0, i1 = ch.borders_idx
            z_mid = z[i0:i1]
            z0 = 0.5 * (z_mid[0] + z_mid[-1])
            s = z_mid - z0
            y = ch.data[i0:i1]

            amps, freqs = self._find_frequencies_topN(s, y, n_modes)
            if len(freqs) == 0:
                # emergency single-mode guess
                L = s[-1] - s[0]
                k_guess = 2 * np.pi / max(L / 2, 1e-6)
                params0 = np.array([y.ptp()/2, 0.0, k_guess])
            else:
                A_cos =  2 * self.dz * amps.real
                A_sin = -2 * self.dz * amps.imag
                params0 = np.empty(3 * len(freqs))
                for i, (a1, a2, ff) in enumerate(zip(A_cos, A_sin, freqs)):
                    params0[3*i:3*i+3] = (a1, a2, 2*np.pi*ff)

            lower = np.tile([-np.inf, -np.inf, 0.0], len(params0)//3)
            upper = np.tile([ np.inf,  np.inf, np.inf], len(params0)//3)

            popt, _ = curve_fit(
                self._sinusoid_sum, s, y,
                p0=params0, bounds=(lower, upper), maxfev=10000
            )

            m = len(popt)//3
            ch.sine.Acos = np.array([popt[3*i]   for i in range(m)])
            ch.sine.Asin = np.array([popt[3*i+1] for i in range(m)])
            ch.sine.k    = np.array([popt[3*i+2] for i in range(m)])
            ch.sine.z0   = z0

        _fit_channel("Bx", self.n_modes_x)
        _fit_channel("By", self.n_modes_y)

    # ---------------- Edge fitting ----------------
    def _boundary_from_sine(self, z_mid, sine):
        xL = z_mid[0]  - self.dz
        xR = z_mid[-1] + self.dz

        fL = fR = dL = dR = ddL = ddR = 0.0
        for A1, A2, k in zip(sine.Acos, sine.Asin, sine.k):
            sL = xL - sine.z0
            sR = xR - sine.z0
            fL += A1*np.cos(k*sL) + A2*np.sin(k*sL)
            fR += A1*np.cos(k*sR) + A2*np.sin(k*sR)
            dL += k*(A2*np.cos(k*sL) - A1*np.sin(k*sL))
            dR += k*(A2*np.cos(k*sR) - A1*np.sin(k*sR))
            ddL += -k*k*(A1*np.cos(k*sL) + A2*np.sin(k*sL))
            ddR += -k*k*(A1*np.cos(k*sR) + A2*np.sin(k*sR))
        return np.array([fL, fR, dL, dR, ddL, ddR], dtype=float)

    def _boundary_from_poly(self, z_prev, poly):
        xL = z_prev[0]  - self.dz
        xR = z_prev[-1] + self.dz
        dp  = poly.deriv()
        ddp = poly.deriv(2)
        return np.array([poly(xL), poly(xR), dp(xL), dp(xR), ddp(xL), ddp(xR)], dtype=float)

    def _balanced_slices(self, n, num_regions):
        base = n // num_regions
        rem  = n %  num_regions
        slices, start = [], 0
        for i in range(num_regions):
            end = start + base + (1 if i < rem else 0)
            if end > start:
                slices.append(slice(start, end))
            start = end
        return slices

    def _fit_poly_side(self, z_region, b_region, z_sines, sine, num_slices, left_side):
        deg = self.degree
        sl = self._balanced_slices(len(z_region), num_slices)
        if left_side:
            sl = list(reversed(sl))

        fit_reg = np.zeros_like(z_region, dtype=float)
        pieces = []

        prev_poly = None
        prev_z = None

        for ix, s in enumerate(sl):
            z_this = z_region[s]
            b_this = b_region[s]

            if ix == 0:
                boundaries = self._boundary_from_sine(z_sines, sine)
            else:
                boundaries = self._boundary_from_poly(prev_z, prev_poly)

            if deg >= 5:
                if left_side:
                    db_dz_left   = (-3*b_this[0] + 4*b_this[1] - b_this[2]) / (2*self.dz)
                    d2b_dz2_left = (2*b_this[0] - 5*b_this[1] + 4*b_this[2] - b_this[3]) / (self.dz**2)
                    coeffs = poly_fit.poly_fit(
                        N=deg, xdata=z_this, ydata=b_this,
                        x0=[z_this[0], z_this[-1]], y0=[b_this[0], boundaries[0]],
                        xp0=[z_this[0], z_this[-1]], yp0=[db_dz_left, boundaries[2]],
                        xpp0=[z_this[0], z_this[-1]], ypp0=[d2b_dz2_left, boundaries[4]],
                    )
                else:
                    db_dz_right   = (3*b_this[-1] - 4*b_this[-2] + b_this[-3]) / (2*self.dz)
                    d2b_dz2_right = (2*b_this[-1] - 5*b_this[-2] + 4*b_this[-3] - b_this[-4]) / (self.dz**2)
                    coeffs = poly_fit.poly_fit(
                        N=deg, xdata=z_this, ydata=b_this,
                        x0=[z_this[0], z_this[-1]], y0=[boundaries[1], b_this[-1]],
                        xp0=[z_this[0], z_this[-1]], yp0=[boundaries[3], db_dz_right],
                        xpp0=[z_this[0], z_this[-1]], ypp0=[boundaries[5], d2b_dz2_right],
                    )
            elif deg >= 3:
                if left_side:
                    db_dz_left = (-3*b_this[0] + 4*b_this[1] - b_this[2]) / (2*self.dz)
                    coeffs = poly_fit.poly_fit(
                        N=deg, xdata=z_this, ydata=b_this,
                        x0=[z_this[0], z_this[-1]], y0=[b_this[0], boundaries[0]],
                        xp0=[z_this[0], z_this[-1]], yp0=[db_dz_left, boundaries[2]],
                    )
                else:
                    db_dz_right = (3*b_this[-1] - 4*b_this[-2] + b_this[-3]) / (2*self.dz)
                    coeffs = poly_fit.poly_fit(
                        N=deg, xdata=z_this, ydata=b_this,
                        x0=[z_this[0], z_this[-1]], y0=[boundaries[1], b_this[-1]],
                        xp0=[z_this[0], z_this[-1]], yp0=[boundaries[3], db_dz_right],
                    )
            else:
                if left_side:
                    coeffs = poly_fit.poly_fit(
                        N=deg, xdata=z_this, ydata=b_this,
                        x0=[z_this[0], z_this[-1]], y0=[b_this[0], boundaries[0]],
                    )
                else:
                    coeffs = poly_fit.poly_fit(
                        N=deg, xdata=z_this, ydata=b_this,
                        x0=[z_this[0], z_this[-1]], y0=[boundaries[1], b_this[-1]],
                    )

            poly = Polynomial(coeffs)
            fit_reg[s] = poly(z_this)

            prev_poly, prev_z = poly, z_this
            pieces.append((s.start, poly))

        return fit_reg, pieces

    def _fit_edges(self):
        z = self.z_full
        for name in ("Bx", "By"):
            ch = self.fields[name]
            i0, i1 = ch.borders_idx
            z_left,  b_left  = z[:i0],  ch.data[:i0]
            z_right, b_right = z[i1:], ch.data[i1:]
            z_mid            = z[i0:i1]

            ns_left  = self.slice_counts[name]["left"]
            ns_right = self.slice_counts[name]["right"]

            fitL, piecesL = self._fit_poly_side(
                z_region=z_left,  b_region=b_left,
                z_sines=z_mid, sine=ch.sine,
                num_slices=ns_left, left_side=True,
            )
            fitR, piecesR = self._fit_poly_side(
                z_region=z_right, b_region=b_right,
                z_sines=z_mid, sine=ch.sine,
                num_slices=ns_right, left_side=False,
            )

            ch.left.set_pieces(starts=[p[0] for p in piecesL], polys=[p[1] for p in piecesL])
            ch.right.set_pieces(starts=[i1 + p[0] for p in piecesR], polys=[p[1] for p in piecesR])

            ch.fit = np.zeros_like(ch.data, dtype=float)
            ch.fit[:i0] = fitL
            ch.fit[i1:] = fitR  # middle filled in merge

    # ---------------- Merge & eval ----------------
    def _merge_sections(self):
        z = self.z_full
        for name in ("Bx", "By"):
            ch = self.fields[name]
            i0, i1 = ch.borders_idx
            mid_mask = (z >= z[i0]) & (z <= z[i1-1])
            ch.fit[mid_mask] = ch.sine.eval(z[mid_mask])

    def evaluate(self, zq):
        out = {}
        for name in ("Bx", "By"):
            ch = self.fields[name]
            i0, i1 = ch.borders_idx
            z0, z1 = self.z_full[i0], self.z_full[i1-1]
            left_mask  = zq <  z0
            mid_mask   = (zq >= z0) & (zq <= z1)
            right_mask = zq >  z1
            y = np.empty_like(zq, dtype=float)
            y[left_mask]  = ch.left.eval(zq[left_mask],  self.z_full)
            y[mid_mask]   = ch.sine.eval(zq[mid_mask])
            y[right_mask] = ch.right.eval(zq[right_mask], self.z_full)
            out[name] = y
        return out

    # ---------------- Primitives & plotting ----------------
    def _compute_primitives(self):
        z = self.z_full
        for name in ("Bx", "By"):
            y_data = self.fields[name].data
            y_fit  = self.fields[name].fit
            if y_data is not None:
                p = sc.integrate.cumulative_trapezoid(y_data, x=z, initial=0.0)
                self.primitives["data"][name] = p
            if y_fit is not None:
                p = sc.integrate.cumulative_trapezoid(y_fit,  x=z, initial=0.0)
                self.primitives["fit"][name] = p

    def plot_fields(self):
        import matplotlib.pyplot as plt
        z = self.z_full
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(9, 7))
        for name, ax in zip(("Bx", "By"), (ax1, ax2)):
            ch = self.fields[name]
            ax.plot(z, ch.data, label=f"{name} data")
            ax.plot(z, ch.fit,  label=f"{name} fit")
            i0, i1 = ch.borders_idx
            ax.axvline(z[i0],   color="k", linestyle="--", linewidth=1)
            ax.axvline(z[i1-1], color="k", linestyle="--", linewidth=1)
            ax.set_ylabel(f"{name} [T]")
            ax.grid(True)
            ax.legend(loc="upper right")
        bt = np.sqrt(self.fields["Bx"].data**2 + self.fields["By"].data**2)
        bt_fit = np.sqrt(self.fields["Bx"].fit**2 + self.fields["By"].fit**2)
        ax3.plot(z, bt, label="|B| data")
        ax3.plot(z, bt_fit, label="|B| fit")
        ax3.set_xlabel("s [m]")
        ax3.set_ylabel("|B| [T]")
        ax3.grid(True)
        ax3.legend(loc="upper right")
        fig.suptitle(f"Magnetic Field at (X, Y) = {self.xy_point}")
        plt.tight_layout()
        plt.show()

    def plot_integral(self):
        """
        Plot cumulative integrals ∫Bx ds and ∫By ds versus z for both data and stitched fit.
        Assumes you've already run .fit(). If primitives aren't computed yet, they'll be built.
        """
        import matplotlib.pyplot as plt

        # Ensure primitives exist (safe to call again)
        if self.primitives["data"]["Bx"] is None or self.primitives["fit"]["Bx"] is None \
                or self.primitives["data"]["By"] is None or self.primitives["fit"]["By"] is None:
            self._compute_primitives()

        z = self.z_full
        bx_int_data = self.primitives["data"]["Bx"]
        bx_int_fit = self.primitives["fit"]["Bx"]
        by_int_data = self.primitives["data"]["By"]
        by_int_fit = self.primitives["fit"]["By"]

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(9, 6))

        # Bx integral
        ax1.plot(z, bx_int_data, label=r"$\int B_x\,ds$ data")
        ax1.plot(z, bx_int_fit, label=r"$\int B_x\,ds$ fit")
        i0x, i1x = self.fields["Bx"].borders_idx
        ax1.axvline(z[i0x], color="k", linestyle="--", linewidth=1)
        ax1.axvline(z[i1x - 1], color="k", linestyle="--", linewidth=1)
        ax1.set_ylabel(r"$\int B_x\,ds$ [Tm]")
        ax1.grid(True)
        ax1.legend(loc="best")

        # By integral
        ax2.plot(z, by_int_data, label=r"$\int B_y\,ds$ data")
        ax2.plot(z, by_int_fit, label=r"$\int B_y\,ds$ fit")
        i0y, i1y = self.fields["By"].borders_idx
        ax2.axvline(z[i0y], color="k", linestyle="--", linewidth=1)
        ax2.axvline(z[i1y - 1], color="k", linestyle="--", linewidth=1)
        ax2.set_xlabel("s [m]")
        ax2.set_ylabel(r"$\int B_y\,ds$ [Tm]")
        ax2.grid(True)
        ax2.legend(loc="best")

        fig.suptitle(f"Cumulative integrals at (X, Y) = {self.xy_point}")
        plt.tight_layout()
        plt.show()

    def results(self):
        return {
            "z": self.z_full,
            "Bx": self.fields["Bx"],
            "By": self.fields["By"],
            "primitives": self.primitives,
        }
