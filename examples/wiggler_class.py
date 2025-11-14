from __future__ import annotations

import numpy as np
import pandas as pd
import scipy as sc
import sympy as sp
import xtrack as xt

import bpmeth as bp
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from line_profiler import profile

# ================================ Main class ================================

class WigglerFieldFitter:

    def __init__(
            self,
            file_path,
            xy_point=(0, 0),
            dx=0.001,
            dy=0.001,
            ds=0.001,
            peak_window=(100, 2100),
            n_modes=6,
            poly_pieces=[15, 15],
            deg=0,
            filter_params=None,
    ):

        self.file_path = file_path
        self.xy_point = xy_point
        self.dx, self.dy, self.ds = dx, dy, ds
        self.peak_window = peak_window
        self.n_modes = n_modes
        self.n_pieces_L, self.n_pieces_R = poly_pieces

        # List that holds a list of borders for each field
        self.borders_idx  = []
        self.poly_borders = []

        # Empty symbolic dicts for the fit parameters.
        self.am_polys = {}      # Corresponds to d^m Bx / dx^m of the polynomials
        self.bm_polys = {}      # Corresponds to d^m By / dx^m of the polynomials
        self.bs_polys = {}      # Corresponds to Bs of the polynomials
        self.am_sines = {}      # Corresponds to d^m Bx / dx^m of the sinusoids
        self.bm_sines = {}      # Corresponds to d^m By / dx^m of the sinusoids
        self.bs_sines = {}      # Corresponds to Bs of the sinusoids

        # NOTE: Filter noise is now only used for the right tails, because those are noisy.
        # We can add a more general functionality later.
        self.filter_params = filter_params

        self.df_raw_data = None
        self.df_on_axis_raw  = None
        self.df_on_axis_fit = None
        self.df_fit_pars = None
        self.s_full = None
        self.length = None
        self.deg = deg
        self.Bs_tol = 1e-3
        self.Bs_fit = True

    # PUBLIC
    # Setter method that calls all the other methods to arrive at a fit.
    def set(self):
        self._parse_to_dataframe()
        self._set_df_fit_pars()
        self.select_xy()
        if self.filter_params != None:
            self._filter_noise()
        self._find_regions()
        self._fit_sinusoids()
        self._fit_edges()

    # Naslagwerkje:
    """
    Use
    pandas
    indexing.Examples
    for `self.df_fit_pars`:
    
    # Inspect structure first
    print(self.df_fit_pars.columns)
    print(self.df_fit_pars.index)
    
    # Single column by label
    col = self.df_fit_pars['column_name']  # simple columns
    # or
    col = self.df_fit_pars.loc[:, 'column_name']
    
    # MultiIndex column (use a tuple)
    col = self.df_fit_pars[('edge_L', 0, 'Bx')]
    # or with .loc
    col = self.df_fit_pars.loc[:, ('edge_L', 0, 'Bx')]
    
    # Slice with pd.IndexSlice for MultiIndex
    import pandas as pd
    idx = pd.IndexSlice
    sub = self.df_fit_pars.loc[:, idx['edge_L', 0, :]]  # all fields at derivative 0 for edge_L

    # Single row or single scalar
    row = self.df_fit_pars.loc[row_label]
    value = self.df_fit_pars.at[row_label, ('edge_L', 0, 'Bx')]
    # by integer position
    col_by_pos = self.df_fit_pars.iloc[:, 2]
    val_by_pos = self.df_fit_pars.iat[3, 2]
    """

    ####################################################################################################################
    # EVALUATION FUNCTIONS
    ####################################################################################################################

    # PRIVATE
    # This function evaluates a sum of (co)sines at the given x values.
    # The parameters are given as a flat list, where each mode has three parameters:
    #  - Amplitude of the cosine term
    #  - Amplitude of the sine term
    #  - Wave number
    # Thus, for n modes, the parameter list has length 3*n.
    @staticmethod
    def _sinusoid(x, *params):
        # Define the output array
        y = np.zeros_like(x, dtype=np.float64)

        # The range depends on the number of sinusoids there are present in the function.
        # B_x is best approximated with two, whereas B_y only needs one.
        # Note the integer division by 3, because each mode has three parameters.
        for A1, A2, k in zip(params[::3], params[1::3], params[2::3]):
            # General sinusoidal part is a linear combination of cosine and sine.
            # This is equivalent to a single function with a phase-offset, but more numerically stable.
            y += A1 * np.cos(k * x) + A2 * np.sin(k * x)

        if (len(params) % 3) == 1:
            y += params[-1]

        return y

    # PRIVATE
    # Polynomials, which coefficients are determined by the boundary conditions and integral over the interval.
    @staticmethod
    def _poly(s0, s1, coeffs):
        c1, c2, c3, c4, c5 = coeffs
        L = s1 - s0
        t = np.polynomial.Polynomial([-s0 / L, 1 / L])

        # basis functions on [0,1]
        b1_coeffs = [1, 0, -18, 32, -15]
        b2_coeffs = [0, 1, -4.5, 6, -2.5]
        b3_coeffs = [0, 0, -12, 28, -15]
        b4_coeffs = [0, 0, 1.5, -4, 2.5]
        b5_coeffs = [0, 0, 30, -60, 30]
        b1_poly = np.polynomial.Polynomial(b1_coeffs)
        b2_poly = np.polynomial.Polynomial(b2_coeffs)
        b3_poly = np.polynomial.Polynomial(b3_coeffs)
        b4_poly = np.polynomial.Polynomial(b4_coeffs)
        b5_poly = np.polynomial.Polynomial(b5_coeffs)

        # combine with correct scaling for derivatives/integral
        poly_t = c1 * b1_poly + L * c2 * b2_poly + c3 * b3_poly + L * c4 * b4_poly + (c5 / L) * b5_poly
        poly_s = poly_t(t)
        return poly_s

    # Workflow:
    # 1. Define the interval [x0, x1]: We know this from the slices.
    # 2. Compute the function value and derivative at the sinusoidal side (analytically), gives c_3 and c_4 (left side) or c_1 and c_2 (right side).
    # 3. Compute the function value and derivative at the "data side" (numerically), gives c_1 and c_2 (left side) or c_3 and c_4 (right side).
    # 4. Compute the integral over the interval (numerically), gives c_5.
    # 5. This immediately gives the polynomial coefficients for that slice.
    ####################################################################################################################
    # IDENTIFYING REGIONS AND SETTING BORDERS IN DATA CLASSES
    ####################################################################################################################
    # PRIVATE
    # This method reads the data from the file and stores it in a pandas DataFrame.
    def _parse_to_dataframe(self) -> None:
        df = pd.read_csv(
            self.file_path, sep=r"\s+", header=None, names=["X", "Y", "Z", "Bx", "By", "Bs"]
        )
        df.set_index(["X", "Y", "Z"], inplace=True)
        self.df_raw_data = df

    def _set_df_fit_pars(self):
        # Build a DataFrame of fit parameters for each derivative order.
        colums = ['field_component', 'derivative_x', 'region', 'func_type', 'params']
        results = pd.DataFrame(columns=colums)
        for der_order in range(self.deg + 1):
            if der_order == 1:
                fields = ["Bx", "By", "Bs"]
            else:
                fields = ["Bx", "By"]
            for field in fields:
                # Loop through left edge.
                for i in range(1, self.n_pieces_L + 1):
                    if field == "Bx":
                        par_dict = self.am_polys.copy()
                    elif field == "By":
                        par_dict = self.bm_polys.copy()
                    else:  # Bs
                        par_dict = self.bs_polys.copy()
                    results.loc[len(results)] = [field, der_order, f'L_poly_{i}', 'polynomial', par_dict]

                # Sinusoidal center region.
                if field == "Bx":
                    par_dict = self.am_sines.copy()
                elif field == "By":
                    par_dict = self.bm_sines.copy()
                else:  # Bs
                    par_dict = self.bs_sines.copy()
                results.loc[len(results)] = [field, der_order, 'center_sine', 'sinusoid', par_dict]

                # Loop through right edge.
                for i in range(1, self.n_pieces_R + 1):
                    if field == "Bx":
                        par_dict = self.am_polys.copy()
                    elif field == "By":
                        par_dict = self.bm_polys.copy()
                    else:  # Bs
                        par_dict = self.bs_polys.copy()
                    results.loc[len(results)] = [field, der_order, f'R_poly_{i}', 'polynomial', par_dict]
        self.df_fit_pars = results

    """
    # ALTERNATIVE IMPLEMENTATION OF _set_df_fit_pars USING MULTIINDEX DATAFRAME AND NESTED DICT
    def _set_df_fit_pars(self):
        # Build both a human-friendly DataFrame and a nested dict `self.fit_pars`.
        rows = []
        index = []
        fit_pars = {"edge_L": {}, "sines": {}, "edge_R": {}}

        for der_order in range(self.deg + 1):
            # Exclude longitudinal field derivatives: Bs appears only for derivative 0
            if der_order == 0:
                fields = ["Bx", "By", "Bs"]
            else:
                fields = ["Bx", "By"]

            key = str(der_order)
            fit_pars["edge_L"].setdefault(key, {})
            fit_pars["sines"].setdefault(key, {})
            fit_pars["edge_R"].setdefault(key, {})

            for field in fields:
                # choose correct parameter prototypes depending on field
                if field == "Bx":
                    poly_proto = self.am_polys.copy()
                    sine_proto = self.am_sines.copy()
                elif field == "By":
                    poly_proto = self.bm_polys.copy()
                    sine_proto = self.bm_sines.copy()
                else:  # Bs
                    poly_proto = self.bs_polys.copy()
                    sine_proto = self.bs_sines.copy()

                # left edge pieces
                left_list = []
                for i in range(1, self.n_pieces_L + 1):
                    left_list.append(poly_proto.copy())
                    index.append((field, der_order, f"L_poly_{i}"))
                    rows.append(poly_proto.copy())

                # center sinusoid region
                fit_pars["sines"][key][field] = sine_proto.copy()
                index.append((field, der_order, "center_sine"))
                rows.append(sine_proto.copy())

                # right edge pieces
                right_list = []
                for i in range(1, self.n_pieces_R + 1):
                    right_list.append(poly_proto.copy())
                    index.append((field, der_order, f"R_poly_{i}"))
                    rows.append(poly_proto.copy())

                fit_pars["edge_L"][key][field] = left_list
                fit_pars["edge_R"][key][field] = right_list

        # MultiIndex DataFrame: index = (field, derivative_x, region), column 'params'
        idx = pd.MultiIndex.from_tuples(index, names=["field_component", "derivative_x", "region"])
        self.df_fit_pars = pd.DataFrame({"params": rows}, index=idx)

        # Nested dict for programmatic access (kept in string keys to match existing usage patterns)
        self.fit_pars = fit_pars
    """

    # PRIVATE
    # This method generates symbolic variables for the sinusoidal fit parameters.
    # The keys are:
    # Aa{m}_{d} : Amplitude of cosine term for mode m, derivative order d of Bx
    # Ba{m}_{d} : Amplitude of sine term for mode m, derivative order d of Bx
    # ka{m}_{d} : Wave number for mode m, derivative order d of Bx
    # Ab{m}_{d} : Amplitude of cosine term for mode m, derivative order d of By
    # Bb{m}_{d} : Amplitude of sine term for mode m, derivative order d of By
    # kb{m}_{d} : Wave number for mode m, derivative order d of By
    # As{m}     : Amplitude of cosine term for mode m of Bs
    # Bs{m}     : Amplitude of sine term for mode m of Bs
    # ks{m}     : Wave number for mode m of Bs
    # The DC terms are:
    # Aa_dc_{d} : DC term for derivative order d of Bx
    # Ab_dc_{d} : DC term for derivative order d of By
    # As_dc     : DC term for Bs
    def _generate_sine_symb_dict(self):
        n_modes = self.n_modes
        n_ders = self.deg

        Aa = {sp.Symbol(f"Aa{m}_{d}"): None for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}
        Ba = {sp.Symbol(f"Ba{m}_{d}"): None for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}
        ka = {sp.Symbol(f"ka{m}_{d}"): None for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}
        Aa_dc = {sp.Symbol(f"Aa_dc_{d}"): None for d in range(1, n_ders + 2)}
        self.am_sines = {**Aa, **Ba, **ka, **Aa_dc}

        Ab = {sp.Symbol(f"Ab{m}_{d}"): None for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}
        Bb = {sp.Symbol(f"Bb{m}_{d}"): None for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}
        kb = {sp.Symbol(f"kb{m}_{d}"): None for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}
        Ab_dc = {sp.Symbol(f"Ab_dc_{d}"): None for d in range(1, n_ders + 2)}
        self.bm_sines = {**Ab, **Bb, **kb, **Ab_dc}

        # Logic: If Bs is 10^-3 times smaller than max(Bx) or max(By), we do not fit it.
        # In that case, we immediately set all bs coefficients to zero.
        # Otherwise, the coefficients are left as None and assigned to numerical values later.
        # use the on-axis subset extracted from self.df_raw_data above
        bx_vals = self.df_on_axis_raw["Bx_0"].abs().dropna()
        by_vals = self.df_on_axis_raw["By_0"].abs().dropna()
        bs_vals = self.df_on_axis_raw["Bs_0"].abs().dropna()

        if len(bs_vals) and len(bx_vals) and len(by_vals):
            bs_max = bs_vals.max()
            denom = min(bx_vals.max(), by_vals.max())
            if denom > 0 and (bs_max / denom) < self.Bs_tol:
                self.Bs_fit = False

        print(f"self.Bs_fit = {self.Bs_fit}")

        if self.Bs_fit:
            dict_entry = None
        else:
            dict_entry = 0.0

        As    = {sp.Symbol(f"As{m}"): dict_entry for m in range(1, n_modes + 1)}
        Bs    = {sp.Symbol(f"Bs{m}"): dict_entry for m in range(1, n_modes + 1)}
        ks    = {sp.Symbol(f"ks{m}"): dict_entry for m in range(1, n_modes + 1)}
        As_dc = {sp.Symbol("As_dc") : dict_entry}

        self.bs_sines = {**As, **Bs, **ks, **As_dc}


    # PRIVATE
    # This method generates symbolic variables for the polynomial fit parameters.
    # The keys are:
    # a{d}_{p} : Coefficient of s^p for derivative order d of Bx
    # b{d}_{p} : Coefficient of s^p for derivative order d of By
    # bs_{p}   : Coefficient of s^p for Bs
    # The degree of the polynomials is fixed at 4 (5 coefficients), because of the basis that we choose.
    def _generate_poly_symb_dict(self):
        n_ders = self.deg
        deg_poly = 4  # degree 4 -> coefficients 0..4

        # create symbols a{der}_{power} and b{der}_{power} for each derivative (der = 1..n_ders+1)
        self.am_polys = {
            sp.Symbol(f"a{d}_{p}"): None
            for d in range(1, n_ders + 2)
            for p in range(deg_poly + 1)
        }
        self.bm_polys = {
            sp.Symbol(f"b{d}_{p}"): None
            for d in range(1, n_ders + 2)
            for p in range(deg_poly + 1)
        }

        # Logic: If Bs is 10^-3 times smaller than max(Bx) or max(By), we do not fit it.
        # In that case, we immediately set all bs coefficients to zero.
        # Otherwise, the coefficients are left as None and assigned to numerical values later.
        try:
            # use the on-axis subset extracted from self.df_raw_data above
            bx_vals = self.df_on_axis_raw["Bx_0"].abs().dropna()
            by_vals = self.df_on_axis_raw["By_0"].abs().dropna()
            bs_vals = self.df_on_axis_raw["Bs_0"].abs().dropna()

            if len(bs_vals) and len(bx_vals) and len(by_vals):
                bs_max = bs_vals.max()
                denom = min(bx_vals.max(), by_vals.max())
                if denom > 0 and (bs_max / denom) < self.Bs_tol:
                    self.Bs_fit = False

        except Exception:
            # On any unexpected issue, leave Bs fitting enabled (fail-safe)
            self.Bs_fit = True

        if self.Bs_fit:
            dict_entry = None
        else:
            dict_entry = 0.0

        # bs coefficients are not per-derivative here (names: bs_0 .. bs_4)
        self.bs_polys = {sp.Symbol(f"bs_{p}"): dict_entry for p in range(deg_poly + 1)}

    # PRIVATE
    # This method extracts on-axis data from the raw DataFrame and computes transverse derivatives.
    # This data is stored in self.df_on_axis_raw.
    def _set_derivative_df(self):
        # Build a DataFrame of on-axis data with columns labeled per derivative order:
        # e.g. Bx_0, By_0, Bs_0, Bx_1, By_1, ...
        subset = self.df_raw_data.xs(self.xy_point, level=("X", "Y")).sort_index().copy()
        # 0th derivative columns
        subset.rename(columns={'Bx': 'Bx_0', 'By': 'By_0', 'Bs': 'Bs_0'}, inplace=True)

        self.df_on_axis_raw = subset

        # compute transverse derivatives for der > 0 and add as columns (skip Bs derivatives)
        for der in range(1, self.deg + 1):
            derivs = self._fit_transverse_polynomials(der=der)
            subset[f'Bx_{der}'] = derivs['Bx']
            subset[f'By_{der}'] = derivs['By']
            # intentionally do not compute/store Bs_{der}

        # use the on-axis subset extracted from self.df_raw_data above
        bs_vals = subset["Bs_0"].abs().dropna()
        bx_vals = subset["Bx_0"].abs().dropna()
        by_vals = subset["By_0"].abs().dropna()

        if len(bs_vals) and len(bx_vals) and len(by_vals):
            bs_max = bs_vals.max()
            denom = min(bx_vals.max(), by_vals.max())
            if denom > 0 and (bs_max / denom) < self.Bs_tol:
                self.Bs_fit = False
                # Make all elements of bs_polys and bs_sines zero
                for key in list(self.bs_polys.keys()):
                    self.bs_polys[key] = 0.0

                for key in list(self.bs_sines.keys()):
                    self.bs_sines[key] = 0.0

    # PRIVATE
    # This method first finds the peaks and valleys in the data for Bx and By
    # Then, it combines them into one array for the extrema of Bx and By
    # Finally, it sets the borders_idx attribute of the FieldChannel objects for Bx and By
    def _find_regions(self):
        w_left = self.peak_window[0]
        w_right = self.peak_window[1]

        field = "Bx"

        # Use df_raw_data column values (fall back to .loc if necessary) as 1D array for peak finding
        series = self.df_raw_data[field].values

        field_peaks = find_peaks(series)[0]
        field_valleys = find_peaks(-series)[0]
        field_peaks = field_peaks[np.logical_and(field_peaks > w_left, field_peaks < w_right)]
        field_valleys = field_valleys[np.logical_and(field_valleys > w_left, field_valleys < w_right)]
        field_extrema = np.sort(np.concatenate((field_peaks, field_valleys)))

        self.borders_idx = [field_extrema[4], field_extrema[-4]]

    # PRIVATE
    # This is an impromptu noise-filter.
    # Will probably need to be improved later.
    # For now, it only filters noise from the right tail of the data, because that's where the noise is present.
    """
    def _filter_noise(self):
        from scipy.signal import savgol_filter
        from scipy.signal import medfilt

        left_idx = self.filter_params[0]
        right_idx = self.filter_params[1]
        kernel_size = self.filter_params[2]
        window_length = self.filter_params[3]
        polyorder = self.filter_params[4]

        for field in ["Bx", "By", "Bs"]:

            # Because Bs is very noisy, we also filter the left tail.
            # This is not necessary for Bx and By, because their left tails are not noisy.
            if field == "Bs":
                tail = slice(None, left_idx)
                m = medfilt(self.raw_data[field][tail].copy(), kernel_size=kernel_size)
                y_sm = savgol_filter(m, window_length=window_length, polyorder=polyorder)
                self.raw_data[field][tail] = y_sm

                tail = slice(right_idx, None)
                m = medfilt(self.raw_data[field][tail].copy(), kernel_size=kernel_size)  # kernel_size must be odd
                y_sm = savgol_filter(m, window_length=window_length, polyorder=polyorder)
                self.raw_data[field][tail] = y_sm

            tail = slice(right_idx, None)
            m = medfilt(self.raw_data[field][tail].copy(), kernel_size=kernel_size)  # kernel_size must be odd
            y_sm = savgol_filter(m, window_length=window_length, polyorder=polyorder)
            self.raw_data[field][tail] = y_sm
    """

    ####################################################################################################################
    # SINUSOID FITTING
    ####################################################################################################################

    # PRIVATE
    # This function finds the amplitudes and frequencies of the (co)sines present in the data.
    @staticmethod
    def _find_modes(x, y, dx, n_modes):
        # Fourier Transform and frequencies
        fft = sc.fft.fft(y)
        fftfreq = sc.fft.fftfreq(len(x), dx)

        # Only keep positive frequencies
        fft = fft[fftfreq > 0]
        fftfreq = fftfreq[fftfreq > 0]

        # Find peaks in the magnitude spectrum
        peaks = find_peaks(np.abs(fft))[0]

        # Order peaks by magnitude, descending
        order = np.argsort(np.abs(fft[peaks]))[::-1]
        peaks = peaks[order][:n_modes]

        # Extract amplitudes and frequencies in the same order
        amplitudes = fft[peaks]
        cos_amps = 2 * dx * amplitudes.real
        sin_amps = -2 * dx * amplitudes.imag
        k_values = 2 * np.pi * fftfreq[peaks]

        return cos_amps, sin_amps, k_values

    # PRIVATE
    # This method fits sinusoids to the data in the regions defined by borders_idx.
    # The fitted parameters are stored in the fit_pars attribute.
    # The fitted data is stored in the fit_data attribute.
    # It uses the _find_modes function to get initial guesses for the parameters.
    def _fit_sinusoids(self, fun=True):
        cols = []
        for der in range(0, self.deg + 1):
            cols.append(f"Bx_{der}")
            cols.append(f"By_{der}")
            if der == 0 and self.Bs_fit:
                cols.append("Bs_0")

        for col in cols:
            sin_slice = slice(self.borders_idx[0], self.borders_idx[1])
            s_reg = self.s_full[sin_slice]

            # parse field and derivative from column name like "Bx_0"
            field, der_order = col.split('_')
            der_order = int(der_order)

            # get the data values for this field in the sinusoidal window
            field_vals = self.df_on_axis_raw[col].iloc[sin_slice].values

            cos_amps, sin_amps, k_modes = self._find_modes(s_reg, field_vals, self.ds, self.n_modes)

            p0 = np.zeros(self.n_modes * 3 + 1)
            p0[-1] = float(np.mean(field_vals))  # DC guess

            for ii in range(self.n_modes):
                p0[3 * ii + 0] = cos_amps[ii]
                p0[3 * ii + 1] = sin_amps[ii]
                p0[3 * ii + 2] = k_modes[ii]

            popt, pcov = curve_fit(self._sinusoid, s_reg, field_vals, p0=p0)

            # Store fitted parameters into the params dict inside self.df_fit_pars
            mask = (
                    (self.df_fit_pars['field_component'] == field)
                    & (self.df_fit_pars['derivative_x'] == der_order)
                    & (self.df_fit_pars['region'] == 'center_sine')
            )
            if mask.any():
                par_dict = dict(self.df_fit_pars.loc[mask, 'params'].values[0])  # make a mutable copy
                d = der_order + 1  # symbolic derivative index starts at 1

                if field in ("Bx", "By"):
                    pref_A = "Aa" if field == "Bx" else "Ab"
                    pref_B = "Ba" if field == "Bx" else "Bb"
                    pref_k = "ka" if field == "Bx" else "kb"
                    for m in range(self.n_modes):
                        i = 3 * m
                        if i < len(popt):
                            par_dict[sp.Symbol(f"{pref_A}{m + 1}_{d}")] = float(popt[i])
                        if i + 1 < len(popt):
                            par_dict[sp.Symbol(f"{pref_B}{m + 1}_{d}")] = float(popt[i + 1])
                        if i + 2 < len(popt):
                            par_dict[sp.Symbol(f"{pref_k}{m + 1}_{d}")] = float(popt[i + 2])
                    if (len(popt) % 3) == 1:
                        par_dict[sp.Symbol(f"{pref_A}_dc_{d}")] = float(popt[-1])
                else:  # "Bs"
                    for m in range(self.n_modes):
                        i = 3 * m
                        if i < len(popt):
                            par_dict[sp.Symbol(f"As{m + 1}")] = float(popt[i])
                        if i + 1 < len(popt):
                            par_dict[sp.Symbol(f"Bs{m + 1}")] = float(popt[i + 1])
                        if i + 2 < len(popt):
                            par_dict[sp.Symbol(f"ks{m + 1}")] = float(popt[i + 2])
                    if (len(popt) % 3) == 1:
                        par_dict[sp.Symbol("As_dc")] = float(popt[-1])

                # write the updated dict back into the DataFrame
                self.df_fit_pars.loc[mask, 'params'] = [par_dict]

            self.df_on_axis_fit[col].iloc[sin_slice] = self._sinusoid(s_reg, *popt)


        # If Bs was not fitted, create matching zero entries for Bs that mirror Bx/By shapes
        if not self.Bs_fit:
            for der_order in range(self.deg + 1):
                bx = self.fit_pars["sines"][der_order].get("Bx")
                by = self.fit_pars["sines"][der_order].get("By")
                ref = bx if bx is not None else by
                if ref is None:
                    self.fit_pars["sines"][der_order]["Bs"] = np.zeros(1)
                else:
                    self.fit_pars["sines"][der_order]["Bs"] = np.zeros_like(ref)

    ####################################################################################################################
    # PIECEWISE POLYNOMIAL FITTING
    ####################################################################################################################

    # PRIVATE
    # Takes the fit parameters from the sinusoidal fit
    # and computes the boundary conditions for the polynomial fits
    # fL, fR are the function values at the left and right boundaries
    # dL, dR are the first derivatives at the left and right boundaries
    # ddL, ddR are the second derivatives at the left and right boundaries
    def _boundary_from_sine(self, field, s_mid, der_order=0):
        xL, xR = s_mid[0] - self.ds, s_mid[-1] + self.ds
        fL = fR = dL = dR = ddL = ddR = 0.0

        params = self.fit_pars["sines"][der_order][field]
        for Ac, As, k in zip(params[::3], params[1::3], params[2::3]):
            fL += Ac * np.cos(k * xL) + As * np.sin(k * xL)
            fR += Ac * np.cos(k * xR) + As * np.sin(k * xR)
            dL += k * (As * np.cos(k * xL) - Ac * np.sin(k * xL))
            dR += k * (As * np.cos(k * xR) - Ac * np.sin(k * xR))

        if (len(params) % 3) == 1:
            c0 = params[-1]
            fL += c0
            fR += c0

        return np.array([fL, fR, dL, dR], dtype=float)

    # PRIVATE
    # This method computes the boundary conditions from a previously fitted polynomial.
    # xL, xR are the left and right boundaries of the new region.
    # dp and ddp are the first and second derivatives of the polynomial.
    def _boundary_from_poly(self, s_prev, poly):
        xL, xR = s_prev[0] - self.ds, s_prev[-1] + self.ds
        dp = poly.deriv()
        return np.array([poly(xL), poly(xR), dp(xL), dp(xR)], dtype=float)

    # PRIVATE
    # This slices a region into num_regions slices of (approximately) equal size.
    @staticmethod
    def _balanced_slices(n, num_regions):
        base, rem = n // num_regions, n % num_regions
        slices, start = [], 0
        for i in range(num_regions):
            end = start + base + (1 if i < rem else 0)
            if end > start:
                slices.append(slice(start, end))
            start = end
        return slices

    def _fit_poly_side(self, field, s_region, b_region, s_mid, num_slices, left_side, der_order):
        slices = self._balanced_slices(len(s_region), num_slices)
        # fit order: first piece next to the center, then outward
        slices_proc = list(reversed(slices)) if left_side else slices

        fit_reg = np.zeros_like(s_region, dtype=float)
        pieces = []
        prev_poly = None
        prev_s = None

        for ix, s in enumerate(slices_proc):
            s_this, b_this = s_region[s], b_region[s]
            integral_this = sc.integrate.trapezoid(b_this, s_this)
            if ix == 0:
                boundaries = self._boundary_from_sine(field, s_mid, der_order)
            else:
                boundaries = self._boundary_from_poly(prev_s, prev_poly)

            if left_side:
                dbL = (-3 * b_this[0] + 4 * b_this[1] - b_this[2]) / (2 * self.ds)
                coeffs = (b_this[0], dbL, boundaries[0], boundaries[2], integral_this)
            else:
                dbR = (3 * b_this[-1] - 4 * b_this[-2] + b_this[-3]) / (2 * self.ds)
                coeffs = (boundaries[1], boundaries[3], b_this[-1], dbR, integral_this)
            # ----------------------------------------------------------

            x0 = float(s_this[0])
            x1 = float(s_this[-1])
            poly = self._poly(x0, x1, coeffs)
            fit_reg[s] = poly(s_this)
            pieces.append((s.start, poly))
            prev_poly, prev_s = poly, s_this

        # borders in strictly increasing s on this side
        slices_ord = sorted(slices, key=lambda sl: sl.stop)
        borders = [float(s_region[0])] + [float(s_region[sl.stop - 1]) for sl in slices_ord]
        return fit_reg, pieces, borders

    def _fit_edges(self):
        if self.Bs_fit:
            fields = ["Bx", "By", "Bs"]
        else:
            fields = ["Bx", "By"]
        for field in fields:
            for der_order in range(self.deg+1):
                # center region slice and grid
                i0, i1 = self.borders_idx
                s_mid = self.s_full[i0:i1]

                # tails
                s_left = self.s_full[:i0]
                s_right = self.s_full[i1:]
                b_left = self.raw_data[der_order][field][:i0]
                b_right = self.raw_data[der_order][field][i1:]

                # degrees + number of chained pieces per side (configurable)
                nL = self.n_pieces_L
                nR = self.n_pieces_R
                fitL, piecesL, bordersL = self._fit_poly_side(field, s_left, b_left, s_mid, nL, left_side=True,
                                                              der_order=der_order)
                self.fit_data[der_order][field][:i0] = fitL
                fitR, piecesR, bordersR = self._fit_poly_side(field, s_right, b_right, s_mid, nR, left_side=False,
                                                              der_order=der_order)

                self.fit_data[der_order][field][i1:] = fitR

                # after computing fitL/piecesL and fitR/piecesR:
                self.fit_data[der_order][field][:i0] = fitL
                self.fit_data[der_order][field][i1:] = fitR

                # NEW: store coefficients (ascending-power) for exporter
                self.fit_pars["edge_L"][der_order][field] = [p[1].coef for p in piecesL]  # order: near-center -> far-left
                self.fit_pars["edge_R"][der_order][field] = [p[1].coef for p in piecesR]  # order: near-center -> far-right

                # existing border assembly (kept)
                self.poly_borders = bordersL[:-1] + [float(self.s_full[i0]), float(self.s_full[i1])] + bordersR[1:]

        # If Bs was not fitted, create matching zero entries for Bs that mirror Bx/By shapes
        # Only create Bs entries for the 0th derivative (Bs has no higher derivatives).
        if not self.Bs_fit:
            for der_order in range(self.deg + 1):
                # Ensure dicts exist
                self.fit_pars["edge_L"].setdefault(der_order, {})
                self.fit_pars["edge_R"].setdefault(der_order, {})

                if der_order != 0:
                    # leave higher-derivative entries jagged / absent for Bs
                    continue

                bx_L = self.fit_pars["edge_L"][der_order].get("Bx")
                if bx_L is None:
                    self.fit_pars["edge_L"][der_order]["Bs"] = []
                else:
                    if isinstance(bx_L, list):
                        self.fit_pars["edge_L"][der_order]["Bs"] = [np.zeros_like(coef) for coef in bx_L]
                    else:
                        self.fit_pars["edge_L"][der_order]["Bs"] = np.zeros_like(bx_L)

                bx_R = self.fit_pars["edge_R"][der_order].get("Bx")
                if bx_R is None:
                    self.fit_pars["edge_R"][der_order]["Bs"] = []
                else:
                    if isinstance(bx_R, list):
                        self.fit_pars["edge_R"][der_order]["Bs"] = [np.zeros_like(coef) for coef in bx_R]
                    else:
                        self.fit_pars["edge_R"][der_order]["Bs"] = np.zeros_like(bx_R)

    ####################################################################################################################
    # TRANSVERSE GRADIENTS
    ####################################################################################################################

    # PRIVATE
    # This method extracts the data at (x,y) = (-1,0), (0,0), (1,0) and fits parabolas to these points.
    # This is done because bpmeth needs the derivatives w.r.t. x at each point.
    # The first derivatives are zero, but can be extracted nevertheless.

    def _fit_transverse_polynomials(self, der=0):
        """
        Fits transverse polynomials of arbitrary degree through specified points and returns the der-th derivative at x=0.

        Parameters
        ----------
        points : list of int or float
            The transverse x positions (in multiples of dx) to use for fitting.
            Must have at least (degree + 1) entries.
        degree : int
            Degree of the polynomial to fit.
        der : int
            The order of the derivative to return (0 = value, 1 = first derivative, etc.)
        Returns
        -------
        dict
            Dictionary with keys "Bx", "By", "Bs" and values as arrays of the der-th derivative at x=0.
        """

        idx = self.df_raw_data.index
        ys = idx.get_level_values("Y")
        xs = idx.get_level_values("X")
        mask = ys == 0
        points = sorted(set(xs[mask]))

        subsets = {px: self.df_raw_data.xs((px, 0), level=["X", "Y"]).sort_index() for px in points}
        derivs = {"Bx": None, "By": None}

        for field in ["Bx", "By"]:
            x = [p * self.dx for p in points]
            n = len(subsets[points[0]][field])
            derivs[field] = np.zeros(n)

            for i in range(n):
                y = [subsets[px][field].to_numpy()[i] for px in points]
                coeffs = np.polyfit(x, y, self.deg)
                # Compute the der-th derivative at x=0
                d_coeffs = np.polyder(coeffs, m=der)
                # Evaluate at x=0
                derivs[field][i] = np.polyval(d_coeffs, 0)
            # Optionally store the result for later use
            col_name = f"{field}_{der}"
            # Ensure df_on_axis_raw exists and assign the derivative column
            self.df_on_axis_raw[col_name] = derivs[field]

        return derivs

    ####################################################################################################################
    # PLOTTING
    ####################################################################################################################

    @staticmethod
    def plot_integrated_fields(self):
        fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)

        Bx_int_raw = sc.integrate.cumulative_trapezoid(self.raw_data[0]["Bx"], dx=self.ds, initial=0)
        By_int_raw = sc.integrate.cumulative_trapezoid(self.raw_data[0]["By"], dx=self.ds, initial=0)
        Bs_int_raw = sc.integrate.cumulative_trapezoid(self.raw_data[0]["Bs"], dx=self.ds, initial=0)

        Bx_int_fit = sc.integrate.cumulative_trapezoid(self.fit_data[0]["Bx"], dx=self.ds, initial=0)
        By_int_fit = sc.integrate.cumulative_trapezoid(self.fit_data[0]["By"], dx=self.ds, initial=0)
        Bs_int_fit = sc.integrate.cumulative_trapezoid(self.fit_data[0]["Bs"], dx=self.ds, initial=0)

        ax1.plot(self.s_full, Bx_int_raw, label='Raw Data')
        ax1.plot(self.s_full, Bx_int_fit, label='Fit', linestyle='--')
        ax2.plot(self.s_full, By_int_raw, label='Raw Data')
        ax2.plot(self.s_full, By_int_fit, label='Fit', linestyle='--')
        ax3.plot(self.s_full, Bs_int_raw, label='Raw Data')
        ax3.plot(self.s_full, Bs_int_fit, label='Fit', linestyle='--')

        # Add vertical lines at different positions for each subplot
        for field in ["Bx", "By", "Bs"]:
            for idx in self.borders_idx:
                ax = {"Bx": ax1, "By": ax2, "Bs": ax3}[field]
                ax.axvline(x=self.s_full[idx], color='k', linestyle='--', linewidth=1)

        ax1.set_title(f"Integrated Magnetic Field at (X, Y) = {self.xy_point}")
        ax1.set_ylabel(r"Integrated Horizontal Field, $\int B_x \, ds$ [T·m]")
        ax2.set_ylabel(r"Integrated Vertical Field, $\int B_y \, ds$ [T·m]")
        ax3.set_ylabel(r"Integrated Longitudinal Field, $\int B_s \, ds$ [T·m]")
        ax3.set_xlabel(r"Longitudinal Position, $s$ [m]")

        ax1.legend(loc="lower right")
        ax2.legend(loc="lower right")
        ax3.legend(loc="upper right")

        # Turn on the grids.
        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.show()

    # PUBLIC
    # Plot the data against the fit.
    def plot_fields(self, der=0):
        fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)

        ax1.plot(self.s_full, self.raw_data[der]["Bx"])
        ax1.plot(self.s_full, self.fit_data[der]["Bx"])
        ax2.plot(self.s_full, self.raw_data[der]["By"])
        ax2.plot(self.s_full, self.fit_data[der]["By"])
        ax3.plot(self.s_full, self.raw_data[der]["Bs"])
        ax3.plot(self.s_full, self.fit_data[der]["Bs"])

        # Add vertical lines at different positions for each subplot
        for field in ["Bx", "By", "Bs"]:
            for idx in self.borders_idx:
                ax = {"Bx": ax1, "By": ax2, "Bs": ax3}[field]
                ax.axvline(x=self.s_full[idx], color='k', linestyle='--', linewidth=1)

        if der == 2:
            x_label = r"$\frac{d^2 B_x}{d x^2}$"
            y_label = r"$\frac{d^2 B_y}{d y^2}$"
            s_label = r"$\frac{d^2 B_s}{d x^2}$"
        elif der == 1:
            x_label = r"$\frac{d B_x}{d x}$"
            y_label = r"$\frac{d B_y}{d y}$"
            s_label = r"$\frac{d B_s}{d x}$"
        else:
            x_label = r"$B_x$"
            y_label = r"$B_y$"
            s_label = r"$B_s$"

        ax1.set_title(f"Magnetic Field at (X, Y) = {self.xy_point}")
        ax1.set_ylabel(f"Horizontal Field, {x_label} [T]")
        ax2.set_ylabel(f"Vertical Field, {y_label} [T]")
        ax3.set_ylabel(f"Longitudinal Field, {s_label} [T]")
        ax3.set_xlabel(r"Longitudinal Position, $s$ [m]")

        ax1.legend([f"{x_label} Data", f"{x_label} Fit"], loc="lower right")
        ax2.legend([f"{y_label} Data", f"{y_label} Fit"], loc="lower right")
        ax3.legend([f"{s_label} Data", f"{s_label} Fit"], loc="upper right")

        # Turn on the grids.
        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.show()

# This class takes the a_n, b_n and b_s coefficients from the WigglerFieldFitter
# It generates the magnetic field functions using bpmeth's GeneralVectorPotential, which it stores as attributes
# It is meant to only store one single segment

# Workflow for next step:
# - Let GeneralVectorPotential generate from a symbolic expression
# - Use GeneralVectorPotential.get_Bfield(lambdify=False) to get Bx, By, Bs functions
# - Use GeneralVectorPotential.get_A() to get Ax, Ay, As functions
#   - These functions are still symbolic expressions
#   - The shape of these functions can be either a sinusoid or a polynomial, depending on the parameters from WigglerFieldFitter
# - These expressions can be "recycled" by copying them and substituting the coefficients for a particular segment in them
# - After this whole process, we lambdify (or numbafy)? Is supposedly faster.

# TODO:
# - Generic symbolic expressions are part of WigglerFull
# -
class WigglerSegment:
    def __init__(self, s0=0, length=0, x0=0, y0=0):
        self.s0 = s0
        self.length = length
        self.x0 = x0
        self.y0 = y0
        self.scale = 1.0

        self.a_expr = None
        self.b_expr = None
        self.bs_expr= None
        self.a_fun  = None
        self.b_fun  = None
        self.bs_fun = None

        self.Bxexpr = None
        self.Byexpr = None
        self.Bsexpr = None
        self.Bxfun  = None
        self.Byfun  = None
        self.Bsfun  = None

        self.Axexpr = None
        self.Ayexpr = None
        self.Asexpr = None
        self.Axfun  = None
        self.Ayfun  = None
        self.Asfun  = None

    def get_field(self, x, y, s):
        return (self.scale * self.Bxfun(x - self.x0, y - self.y0, s),
                self.scale * self.Byfun(x - self.x0, y - self.y0, s),
                self.scale * self.Bsfun(x - self.x0, y - self.y0, s))

    # Order of arguments: (x, y, s). Vectorized.
    def get_vector_potential(self, x, y, s):
        return (self.scale * self.Axfun(x - self.x0, y - self.y0, s),
                self.scale * self.Ayfun(x - self.x0, y - self.y0, s),
                self.scale * self.Asfun(x - self.x0, y - self.y0, s))


class WigglerFull:
    def __init__(self, WigglerFieldFitter):
        self.field_fitter = WigglerFieldFitter
        self.segments = []
        self.integrator = []
        self.n_slices = 0
        self.env = None
        self.wiggler_line = None

        self._set_generic_expr()
        print(f"Generic Bx_poly(x, y, s) = {self.generic_poly_B[0]}")
        print(f"Generic By_poly(x, y, s) = {self.generic_poly_B[1]}")
        print(f"Generic Bs_poly(x, y, s) = {self.generic_poly_B[2]}")
        print(f"Generic Bx_sine(x, y, s) = {self.generic_sine_B[0]}")
        print(f"Generic By_sine(x, y, s) = {self.generic_sine_B[1]}")
        print(f"Generic Bs_sine(x, y, s) = {self.generic_sine_B[2]}")
        self.set_segments()

    # PRIVATE
    # This function defines generic symbolic expressions for the vector potential and magnetic field
    # - For the sines: Aa_ij corresponds to the i-th mode of the j-th derivative of Bx, cosine amplitude
    # - Similarly, Ba_ij is the sine amplitude, ka_ji is the wave number
    # - Ba_ij, Bb_ij, kb_ij are the corresponding parameters for By
    # - For the polynomials: a_j0, a_j1, ..., a_j4 are the coefficients of the j-th derivative of Bx
    # - Similarly, b_j0, b_j1, ..., b_j4 are the coefficients for By
    # Also added bs expressions.
    def _set_generic_expr(self):
        n_modes = self.field_fitter.n_modes
        n_ders = self.field_fitter.deg
        s = sp.symbols("s")
        curv = 0

        # symbols (mode m = 1..n_modes, derivative d = 1..n_ders+1)
        Aa = {(m, d): sp.Symbol(f"Aa{m}_{d}") for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}
        Ba = {(m, d): sp.Symbol(f"Ba{m}_{d}") for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}
        ka = {(m, d): sp.Symbol(f"ka{m}_{d}") for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}

        Ab = {(m, d): sp.Symbol(f"Ab{m}_{d}") for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}
        Bb = {(m, d): sp.Symbol(f"Bb{m}_{d}") for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}
        kb = {(m, d): sp.Symbol(f"kb{m}_{d}") for m in range(1, n_modes + 1) for d in range(1, n_ders + 2)}

        As = {m: sp.Symbol(f"As{m}") for m in range(1, n_modes + 1)}
        Bs = {m: sp.Symbol(f"Bs{m}") for m in range(1, n_modes + 1)}
        ks = {m: sp.Symbol(f"ks{m}") for m in range(1, n_modes + 1)}

        # DC per DERIVATIVE, not per mode
        Aa_dc = {d: sp.Symbol(f"Aa_dc_{d}") for d in range(1, n_ders + 2)}
        Ab_dc = {d: sp.Symbol(f"Ab_dc_{d}") for d in range(1, n_ders + 2)}
        As_dc = sp.Symbol(f"As_dc")

        # --- build expressions PER DERIVATIVE ---
        self.a_sine_exprs = tuple(
            Aa_dc[d] + sum(Aa[(m, d)] * sp.cos(ka[(m, d)] * s) + Ba[(m, d)] * sp.sin(ka[(m, d)] * s)
                           for m in range(1, n_modes + 1))
            for d in range(1, n_ders + 2)
        )
        self.b_sine_exprs = tuple(
            Ab_dc[d] + sum(Ab[(m, d)] * sp.cos(kb[(m, d)] * s) + Bb[(m, d)] * sp.sin(kb[(m, d)] * s)
                           for m in range(1, n_modes + 1))
            for d in range(1, n_ders + 2)
        )

        self.bs_sine_exprs = (As_dc + sum(As[m] * sp.cos(ks[m] * s) + Bs[m] * sp.sin(ks[m] * s) for m in range(1, n_modes + 1)))

        deg_poly = 4
        # create per-derivative polynomial coefficient symbols and expressions (degree 4 -> 5 terms)
        a_p_syms = {}
        b_p_syms = {}
        a_poly_exprs_list = []
        b_poly_exprs_list = []

        for j in range(1, n_ders + 2):
            a_syms = sp.symbols(f"a{j}_0:{deg_poly+1}")
            b_syms = sp.symbols(f"b{j}_0:{deg_poly+1}")
            a_p_syms[j] = a_syms
            b_p_syms[j] = b_syms
            a_poly_exprs_list.append(sum(coef * s**i for i, coef in enumerate(a_syms)))
            b_poly_exprs_list.append(sum(coef * s**i for i, coef in enumerate(b_syms)))
        bs_symbols = sp.symbols(f"bs_0:{deg_poly+1}")
        bs_poly_exprs_list = sum(coef * s**i for i, coef in enumerate(bs_symbols))

        self.a_poly_exprs = tuple(a_poly_exprs_list)
        self.b_poly_exprs = tuple(b_poly_exprs_list)
        self.bs_poly_exprs = (bs_poly_exprs_list)

        # Changed GeneralVectorPotential to accept expressions with free parameters such as Ac_i, As_i, k_i.
        # Pass the sympy expressions as strings (bpmeth/bp accepts string expressions)
        # self.generic_sine_A = (Ax, Ay, As)
        # self.generic_sine_B = (Bx, By, Bs)
        # Likewise for generic_poly_A and generic_poly_B
        a_sine_exprs_strings = tuple(f"{expr}" for expr in self.a_sine_exprs)
        b_sine_exprs_strings = tuple(f"{expr}" for expr in self.b_sine_exprs)
        bs_sine_exprs_string = f"{self.bs_sine_exprs}"
        generic_sine_bpmeth = bp.GeneralVectorPotential(hs=f"{curv}", a=a_sine_exprs_strings, b=b_sine_exprs_strings, bs=bs_sine_exprs_string)
        self.generic_sine_B = generic_sine_bpmeth.get_Bfield(lambdify=False)
        self.generic_sine_A = generic_sine_bpmeth.get_A()

        a_poly_exprs_strings = tuple(f"{expr}" for expr in self.a_poly_exprs)
        b_poly_exprs_strings = tuple(f"{expr}" for expr in self.b_poly_exprs)
        bs_poly_exprs_string = f"{self.bs_poly_exprs}"
        generic_poly_bpmeth = bp.GeneralVectorPotential(hs=f"{curv}", a=a_poly_exprs_strings, b=b_poly_exprs_strings, bs=bs_poly_exprs_string)
        self.generic_poly_B = generic_poly_bpmeth.get_Bfield(lambdify=False)
        self.generic_poly_A = generic_poly_bpmeth.get_A()

    def _extract_edge_fit_params(self, fit_type="edge_L"):
        pieces_all = self.field_fitter.fit_pars[fit_type]
        #print(f"Extracting {fit_type} fit parameters: {pieces_all}")
        base_map = {"Bx": "a", "By": "b", "Bs": "bs"}
        combined_pieces = []  # store as list of dicts, index = piece index

        for der_order, fields_dict in pieces_all.items():
            d = int(der_order) + 1  # e.g. 0 -> a1_, 1 -> a2_, ...
            for field, pieces in fields_dict.items():
                base = base_map[field]

                for p_idx, piece in enumerate(pieces):
                    # ensure the list is large enough
                    while len(combined_pieces) <= p_idx:
                        combined_pieces.append({})

                    # Normalize coefficients
                    if isinstance(piece, dict):
                        coef_arr = np.array(list(piece.values()), dtype=float)
                    elif hasattr(piece, "coef"):
                        coef_arr = np.asarray(piece.coef)
                    elif isinstance(piece, (list, tuple, np.ndarray)):
                        coef_arr = np.asarray(piece)
                    else:
                        coef_arr = np.asarray([piece])

                    coef_arr = coef_arr.ravel()

                    # Add coefficients for this derivative order and field
                    if field == "Bs":
                        for power, val in enumerate(coef_arr):
                            name = f"{base}_{power}"
                            combined_pieces[p_idx][sp.Symbol(name)] = float(val)
                    else:
                        for power, val in enumerate(coef_arr):
                            name = f"{base}{d}_{power}"
                            combined_pieces[p_idx][sp.Symbol(name)] = float(val)

        if fit_type == "edge_L":
            combined_pieces.reverse()  # left edge pieces need to be reversed

        return combined_pieces

    def _extract_sine_fit_params(self):
        fit_pars_all = self.field_fitter.fit_pars["sines"]
        n_modes = int(self.field_fitter.n_modes)
        combined_pieces = {}
        piece_idx = 0  # sines usually represent a single continuous region

        for der_order, fields_dict in fit_pars_all.items():
            d = int(der_order) + 1
            for field, sine_params in fields_dict.items():

                # Determine correct prefix mapping and whether names are per-derivative
                if field == "Bx":
                    pref_A, pref_B, pref_k = "Aa", "Ba", "ka"
                    dc_sym = sp.symbols(f"Aa_dc_{d}")
                    per_derivative = True
                elif field == "By":
                    pref_A, pref_B, pref_k = "Ab", "Bb", "kb"
                    dc_sym = sp.symbols(f"Ab_dc_{d}")
                    per_derivative = True
                else:  # "Bs" (longitudinal sines are not per-derivative)
                    pref_A, pref_B, pref_k = "As", "Bs", "ks"
                    dc_sym = sp.symbols("As_dc")
                    per_derivative = False

                if piece_idx not in combined_pieces:
                    combined_pieces[piece_idx] = {}

                # Loop over modes and assign amplitudes / wavenumbers
                for m in range(n_modes):
                    i = 3 * m
                    if i < len(sine_params):
                        name = f"{pref_A}{m + 1}_{d}" if per_derivative else f"{pref_A}{m + 1}"
                        combined_pieces[piece_idx][sp.symbols(name)] = float(sine_params[i])
                    if i + 1 < len(sine_params):
                        name = f"{pref_B}{m + 1}_{d}" if per_derivative else f"{pref_B}{m + 1}"
                        combined_pieces[piece_idx][sp.symbols(name)] = float(sine_params[i + 1])
                    if i + 2 < len(sine_params):
                        name = f"{pref_k}{m + 1}_{d}" if per_derivative else f"{pref_k}{m + 1}"
                        combined_pieces[piece_idx][sp.symbols(name)] = float(sine_params[i + 2])

                # If there’s a trailing DC term (mod 3 == 1)
                if (len(sine_params) % 3) == 1:
                    combined_pieces[piece_idx][dc_sym] = float(sine_params[-1])

        return [combined_pieces[idx] for idx in sorted(combined_pieces.keys())]

    def _extract_fit_params(self, fit_type="edge_L"):
        if fit_type in ("edge_L", "edge_R"):
            return self._extract_edge_fit_params(fit_type)
        elif fit_type == "sines":
            return self._extract_sine_fit_params()
        else:
            return []

    def set_segments(self):

        seg = []
        # determine fields (kept for selecting a reference for borders/shapes)
        fields = ["Bx", "By", "Bs"]

        # Use the first field as reference for shapes / poly_borders (parameters are global and substituted once per segment)
        poly_borders = self.field_fitter.poly_borders
        nL = self.field_fitter.n_pieces_L
        nR = self.field_fitter.n_pieces_R

        # ===================== LEFT EDGE =====================
        left_pieces = self._extract_fit_params(fit_type="edge_L")
        s_borders_L = poly_borders[: nL + 1]
        print(f"s_borders_L = {s_borders_L}")

        for i, params in enumerate(left_pieces):
            if i + 1 >= len(s_borders_L):
                break
            s0 = s_borders_L[i]
            length = s_borders_L[i + 1] - s_borders_L[i]
            segment = WigglerSegment(s0=s0, length=length)

            # Substitute parameters into generic expressions once for this segment
            exprs_B  = list(self.generic_poly_B)
            exprs_A  = list(self.generic_poly_A)
            exprs_a  = list(self.a_poly_exprs)
            exprs_b  = list(self.b_poly_exprs)
            exprs_bs = self.bs_poly_exprs

            segment.Bxexpr, segment.Byexpr, segment.Bsexpr = [expr.subs(params) for expr in exprs_B]
            segment.Axexpr, segment.Ayexpr, segment.Asexpr = [expr.subs(params) for expr in exprs_A]
            segment.a_expr = [exprs_a[j].subs(params) for j in range(len(exprs_a))]
            segment.b_expr = [exprs_b[j].subs(params) for j in range(len(exprs_b))]
            segment.bs_expr = [exprs_bs.subs(params)]

            for comp in ["Bx", "By", "Bs", "Ax", "Ay", "As"]:
                expr = getattr(segment, f"{comp}expr")
                setattr(segment, f"{comp}fun", sp.lambdify(("x", "y", "s"), expr, modules="numpy"))

            seg.append(segment)

        # ===================== CENTER REGION (SINES) =====================
        sine_pieces = self._extract_fit_params(fit_type="sines")
        i0, i1 = nL, nL + 1
        s_borders_center = poly_borders[i0 : i1 + 1]
        print(f"s_borders_center = {s_borders_center}")

        if len(sine_pieces) > 0 and len(s_borders_center) >= 2:
            for i, params in enumerate(sine_pieces):
                s0 = s_borders_center[i]
                length = s_borders_center[i + 1] - s_borders_center[i]
                segment = WigglerSegment(s0=s0, length=length)

                exprs_B = list(self.generic_sine_B)
                exprs_A = list(self.generic_sine_A)
                exprs_a = list(self.a_sine_exprs)
                exprs_b = list(self.b_sine_exprs)
                exprs_bs = self.bs_sine_exprs

                segment.Bxexpr, segment.Byexpr, segment.Bsexpr = [expr.subs(params) for expr in exprs_B]
                segment.Axexpr, segment.Ayexpr, segment.Asexpr = [expr.subs(params) for expr in exprs_A]
                segment.a_expr = [exprs_a[j].subs(params) for j in range(len(exprs_a))]
                segment.b_expr = [exprs_b[j].subs(params) for j in range(len(exprs_b))]
                segment.bs_expr = [exprs_bs.subs(params)]

                for comp in ["Bx", "By", "Bs", "Ax", "Ay", "As"]:
                    expr = getattr(segment, f"{comp}expr")
                    setattr(segment, f"{comp}fun", sp.lambdify(("x", "y", "s"), expr, modules="numpy"))

                seg.append(segment)

        # ===================== RIGHT EDGE =====================
        right_pieces = self._extract_fit_params(fit_type="edge_R")
        s_borders_R = poly_borders[-nR - 1 :]
        print(f"s_borders_R = {s_borders_R}")

        for i, params in enumerate(right_pieces):
            if i + 1 >= len(s_borders_R):
                break
            s0 = s_borders_R[i]
            length = s_borders_R[i + 1] - s_borders_R[i]
            segment = WigglerSegment(s0=s0, length=length)

            exprs_B = list(self.generic_poly_B)
            exprs_A = list(self.generic_poly_A)
            exprs_a = list(self.a_poly_exprs)
            exprs_b = list(self.b_poly_exprs)
            exprs_bs = self.bs_poly_exprs

            segment.Bxexpr, segment.Byexpr, segment.Bsexpr = [expr.subs(params) for expr in exprs_B]
            segment.Axexpr, segment.Ayexpr, segment.Asexpr = [expr.subs(params) for expr in exprs_A]
            segment.a_expr = [exprs_a[j].subs(params) for j in range(len(exprs_a))]
            segment.b_expr = [exprs_b[j].subs(params) for j in range(len(exprs_b))]
            segment.bs_expr = [exprs_bs.subs(params)]

            for comp in ["Bx", "By", "Bs", "Ax", "Ay", "As"]:
                expr = getattr(segment, f"{comp}expr")
                setattr(segment, f"{comp}fun", sp.lambdify(("x", "y", "s"), expr, modules="numpy"))

            seg.append(segment)

        # Store all field segments
        self.segments = seg

    def seg_selector(self, s):
        # Scalar path: return single segment
        if np.isscalar(s):
            seg = None
            for seg_candidate in self.segments:
                s0 = seg_candidate.s0
                s1 = s0 + seg_candidate.length
                if s0 <= s <= s1:
                    seg = seg_candidate
                    break
            if seg is None:
                seg = self.segments[-1]
            return seg

        # Array path: return array of indices (one index per s)
        s_arr = np.asarray(s)
        idxs = np.full(s_arr.shape, len(self.segments) - 1, dtype=int)  # default last segment (clamp)
        for i, seg_candidate in enumerate(self.segments):
            s0 = seg_candidate.s0
            s1 = s0 + seg_candidate.length
            mask = (s_arr >= s0) & (s_arr <= s1)
            idxs[mask] = i
        return idxs

    def get_field(self, x, y, s):

        # Scalar s -> keep existing behavior
        if np.isscalar(s):
            seg = self.seg_selector(s)
            return seg.get_field(x, y, s)

        # Vectorized s -> build outputs by grouping by segment
        s_arr = np.asarray(s)
        idxs = self.seg_selector(s_arr)  # array of indices

        Bx = np.empty_like(s_arr, dtype=float)
        By = np.empty_like(s_arr, dtype=float)
        Bs = np.empty_like(s_arr, dtype=float)

        for i, seg in enumerate(self.segments):
            mask = idxs == i
            if not np.any(mask):
                continue
            s_sub = s_arr[mask]
            bx_sub, by_sub, bs_sub = seg.get_field(x, y, s_sub)
            Bx[mask] = bx_sub
            By[mask] = by_sub
            Bs[mask] = bs_sub

        return Bx, By, Bs

    def get_vector_potential(self, x, y, s):
        import numpy as _np

        if _np.isscalar(s):
            seg = self.seg_selector(s)
            return seg.get_vector_potential(x, y, s)

        s_arr = _np.asarray(s)
        idxs = self.seg_selector(s_arr)

        Ax = _np.empty_like(s_arr, dtype=float)
        Ay = _np.empty_like(s_arr, dtype=float)
        As = _np.empty_like(s_arr, dtype=float)

        for i, seg in enumerate(self.segments):
            mask = idxs == i
            if not _np.any(mask):
                continue
            s_sub = s_arr[mask]
            ax_sub, ay_sub, as_sub = seg.get_vector_potential(x, y, s_sub)
            Ax[mask] = ax_sub
            Ay[mask] = ay_sub
            As[mask] = as_sub

        return Ax, Ay, As

    def plot_field(self, s_ends=(None,None), x0=0.0, y0=0.0, n_points=2000, plot_data=False):
        if s_ends == (None, None):
            s_start = self.field_fitter.s_full[0]
            s_end   = self.field_fitter.s_full[-1]
        else:
            s_start, s_end = s_ends

        s_vals = np.linspace(s_start, s_end, n_points)
        # If plot_data is true, then it tries to extract the corresponding data from field_fitter.
        # If that data is not available (because x0 and y0 are not in the dataframe index), then it skips plotting the data.
        if plot_data:
            x_int = int(x0 * 1000)
            y_int = int(y0 * 1000)
            import warnings
            # verify (X,Y) exists in the dataframe index
            try:
                self.field_fitter.df_raw_data.xs((x_int, y_int), level=["X", "Y"])
            except KeyError:
                xy_pairs = sorted(set(zip(self.field_fitter.df_raw_data.index.get_level_values("X"),
                                          self.field_fitter.df_raw_data.index.get_level_values("Y"))))
                warnings.warn(
                    f"Requested (X,Y)=({x_int},{y_int}) not found in `self.field_fitter.df`. "
                    f"Skipping data overlay. Available (X,Y) pairs (first 10 shown): {xy_pairs[:10]}"
                )
                plot_data = False
            else:
                self.field_fitter.xy_point = (x_int, y_int)
                s_full = self.field_fitter.s_full
                Bx_data = self.field_fitter.raw_data[0]["Bx"]
                By_data = self.field_fitter.raw_data[0]["By"]
                Bs_data = self.field_fitter.raw_data[0]["Bs"]

        Bx_vals, By_vals, Bs_vals = self.get_field(x0, y0, s_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(s_vals, Bx_vals, label='Bx')
        plt.plot(s_vals, By_vals, label='By')
        plt.plot(s_vals, Bs_vals, label='Bs')

        if plot_data:
            plt.scatter(s_full, Bx_data, label='Bx Data', color='C0', s=5, alpha=0.5)
            plt.scatter(s_full, By_data, label='By Data', color='C1', s=5, alpha=0.5)
            plt.scatter(s_full, Bs_data, label='Bs Data', color='C2', s=5, alpha=0.5)

        plt.xlabel('s [m]')
        plt.ylabel('Magnetic Field [T]')
        plt.title(f'Magnetic Field along Wiggler at (x={x0}, y={y0})')
        plt.legend()
        plt.grid()
        plt.show()

    def set_integrator(self, n_slices=1000, n_steps = 1000):
        if self.segments == []:
            self.set_segments()

        self.n_slices = n_slices
        l_wig = self.field_fitter.length
        s_start = self.field_fitter.s_full[0]
        s_end   = self.field_fitter.s_full[-1]

        s_cuts = np.linspace(s_start, s_end, n_slices + 1)
        #s_mid = 0.5 * (s_cuts[:-1] + s_cuts[1:])
        for ii in range(n_slices):
            wig = xt.BorisSpatialIntegrator(fieldmap_callable=self.get_field, s_start=s_cuts[ii], s_end=s_cuts[ii + 1],
                                            n_steps=np.round(n_steps / n_slices).astype(int),
                                            verbose=True)
            self.integrator.append(wig)

    def get_line(self):
        if self.integrator == []:
            self.set_integrator()

        self.env = xt.Environment()

        for ii in range(self.n_slices):
            self.env.elements[f'wigslice_{ii}'] = self.integrator[ii]
        self.wiggler_line = self.env.new_line(components=['wigslice_' + str(ii) for ii in range(self.n_slices)])
        return self.wiggler_line

    def correctors(self, particle_ref):
        start_time = time.time()
        if self.wiggler_line is None:
            self.get_line()

        self.wiggler_line.particle_ref = particle_ref

        self.env['k0l_corr1'] = 0.
        self.env['k0l_corr2'] = 0.
        self.env['k0l_corr3'] = 0.
        self.env['k0l_corr4'] = 0.
        self.env['k0sl_corr1'] = 0.
        self.env['k0sl_corr2'] = 0.
        self.env['k0sl_corr3'] = 0.
        self.env['k0sl_corr4'] = 0.
        self.env['on_wig_corr'] = 1.0

        self.env.new('corr1', xt.Multipole, knl=['on_wig_corr * k0l_corr1'], ksl=['on_wig_corr * k0sl_corr1'])
        self.env.new('corr2', xt.Multipole, knl=['on_wig_corr * k0l_corr2'], ksl=['on_wig_corr * k0sl_corr2'])
        self.env.new('corr3', xt.Multipole, knl=['on_wig_corr * k0l_corr3'], ksl=['on_wig_corr * k0sl_corr3'])
        self.env.new('corr4', xt.Multipole, knl=['on_wig_corr * k0l_corr4'], ksl=['on_wig_corr * k0sl_corr4'])

        l_wig = self.field_fitter.length

        self.wiggler_line.insert([
            self.env.place('corr1', at=0.02),
            self.env.place('corr2', at=0.1),
            self.env.place('corr3', at=l_wig - 0.1),
            self.env.place('corr4', at=l_wig - 0.02),
        ], s_tol=5e-3
        )

        # To compute the kicks
        opt = self.wiggler_line.match(
            solve=False,
            betx=0, bety=0,
            only_orbit=True,
            include_collective=True,
            vary=xt.VaryList(['k0l_corr1', 'k0sl_corr1',
                              'k0l_corr2', 'k0sl_corr2',
                              'k0l_corr3', 'k0sl_corr3',
                              'k0l_corr4', 'k0sl_corr4',
                              ], step=1e-6),
            targets=[
                xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
                xt.TargetSet(x=0., y=0, at='wigslice_167'),
                xt.TargetSet(x=0., y=0, at='wigslice_833')
                ],
        )
        opt.step(2)
        end_time = time.time()
        print(f"Wiggler correctors set in {end_time - start_time:.2f} seconds.")
        print("Corrector strengths [T]:")
        print(f"  k0l_corr1 = {self.env['k0l_corr1']:.6e}, k0sl_corr1 = {self.env['k0sl_corr1']:.6e}")
        print(f"  k0l_corr2 = {self.env['k0l_corr2']:.6e}, k0sl_corr2 = {self.env['k0sl_corr2']:.6e}")
        print(f"  k0l_corr3 = {self.env['k0l_corr3']:.6e}, k0sl_corr3 = {self.env['k0sl_corr3']:.6e}")
        print(f"  k0l_corr4 = {self.env['k0l_corr4']:.6e}, k0sl_corr4 = {self.env['k0sl_corr4']:.6e}")