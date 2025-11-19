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
            n_modes=6,
            poly_pieces=200,
            deg=0,
            filter_params=None,
    ):

        self.file_path = file_path
        self.xy_point = xy_point
        self.dx, self.dy, self.ds = dx, dy, ds
        self.poly_order = 4  # fixed at 4 for now (5 coefficients)

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
        self._set_derivative_df()
        self._find_regions()
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
    # Polynomials, which coefficients are determined by the boundary conditions and integral over the interval.
    # c1 = f(s0)
    # c2 = f'(s0)
    # c3 = f(s1)
    # c4 = f'(s1)
    # c5 = integral from s0 to s1 of f(s) ds
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



    ####################################################################################################################
    # IDENTIFYING REGIONS AND SETTING BORDERS IN DATA CLASSES
    ####################################################################################################################
    # PRIVATE
    # This method reads the data from the file and stores it in a pandas DataFrame.
    # It also extracts the on-axis data and checks if Bs is negligible compared to Bx and By.
    def _parse_to_dataframe(self) -> None:
        df = pd.read_csv(
            self.file_path, sep=r"\s+", header=None, names=["X", "Y", "Z", "Bx", "By", "Bs"]
        )
        df.set_index(["X", "Y", "Z"], inplace=True)
        self.df_raw_data = df
        self.s_full = np.sort(df.index.get_level_values("Z").unique()).astype(float)

        # Check if Bs is much smaller than Bx and By
        self.df_on_axis_raw = self.df_raw_data.xs(self.xy_point, level=("X", "Y")).sort_index().copy(deep=True)

        # use the on-axis subset extracted from self.df_raw_data above
        bs_vals = self.df_on_axis_raw["Bs"].abs().dropna()
        bx_vals = self.df_on_axis_raw["Bx"].abs().dropna()
        by_vals = self.df_on_axis_raw["By"].abs().dropna()

        if len(bs_vals) and len(bx_vals) and len(by_vals):
            bs_max = bs_vals.max()
            denom = min(bx_vals.max(), by_vals.max())
            if denom > 0 and (bs_max / denom) < self.Bs_tol:
                self.Bs_fit = False



    # PRIVATE
    # This method extracts on-axis data from the raw DataFrame and fits it to polynomials.
    # It computes the derivatives of said polynomials and stores them in the self.df_on_axis_raw DataFrame.
    # The data is not "raw" in the technical sense, but is used to fit a function of s to.
    def _set_derivative_df(self):
        # 0th derivative columns
        self.df_on_axis_raw.columns = pd.MultiIndex.from_product([self.df_on_axis_raw.columns, [0]])

        # compute transverse derivatives for der > 0 and add as columns (skip Bs derivatives)
        for der in range(1, self.deg + 1):
            derivs = self._fit_transverse_polynomials(der=der)
            self.df_on_axis_raw[('Bx', der)] = derivs['Bx']
            self.df_on_axis_raw[('By', der)] = derivs['By']
            # intentionally do not compute/store Bs_{der}

        # create a zeros-only DataFrame with the same index/columns as the on-axis raw data
        self.df_on_axis_fit = self.df_on_axis_raw.copy(deep=True)
        # set all values to 0.0 while preserving index and column structure
        self.df_on_axis_fit.loc[:, :] = 0.0



    # PRIVATE
    # This method loops over all fields and derivatives.
    # It finds peaks and valleys in the data within the peak_window, with specified width and prominence.
    def _find_regions(self):
        fields = ["Bx", "By"]

        for field in fields:
            for der in range(0, self.deg + 1):
                series = self.df_on_axis_raw[(field, der)].values

                # TODO: The filters in find_peaks are very useful for filtering noisy components in the data.
                # TODO: Still check if this adequately picks up small (but real) features in the data.
                # TODO: Chose width=15 and prominence=std_series as reasonable starting points.
                # TODO: With these settings, it correctly reduces the first derivative to one piece only.
                std_series = np.std(series)
                field_peaks = find_peaks(series, width=15, prominence=std_series)[0]
                field_valleys = find_peaks(-series, width=15, prominence=std_series)[0]
                field_extrema = np.sort(np.concatenate((field_peaks, field_valleys)))

                field_extrema = np.insert(field_extrema, 0, 0)
                field_extrema = np.append(field_extrema, len(series) - 1)

                # Set region starts in df_fit_pars
                n_pieces = len(field_extrema+1)  # number of pieces is number of extrema - 1
                print(n_pieces)
                self._set_df_fit_pars(der, n_pieces, field, field_extrema)

        # If Bs is to be fitted, do the same for Bs
        if self.Bs_fit:
            field = "Bs"
            der = 0
            series = self.df_raw_data[field].values

            field_peaks = find_peaks(series)[0]
            field_valleys = find_peaks(-series)[0]
            field_peaks = field_peaks[np.logical_and(field_peaks > w_left, field_peaks < w_right)]
            field_valleys = field_valleys[np.logical_and(field_valleys > w_left, field_valleys < w_right)]
            field_extrema = np.sort(np.concatenate((field_peaks, field_valleys)))

            field_extrema = np.insert(field_extrema, 0, 0)
            field_extrema = np.append(field_extrema, len(series) - 1)

            # Set region starts in df_fit_pars
            n_pieces = len(field_extrema)  # number of pieces is number of extrema
            self._set_df_fit_pars(der, n_pieces, field, field_extrema)

        else:
            self._set_df_fit_pars(0, 1, "Bs", [0])

        self.df_fit_pars.set_index(['field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start', 'idx_end', 'param_index'],
                                       inplace=True)



    # PRIVATE
    # This method initializes and appends rows to the df_fit_pars DataFrame.
    # Each row corresponds to a polynomial piece for a specific field and derivative.
    # It stores metadata about the piece, including parameter names and initial values.
    # This method is called by _find_regions to populate the DataFrame.
    # In case the set consists of only one piece, the parameters are initialized to 0.
    def _set_df_fit_pars(self, der_order, n_pieces, field, idx_extrema):
        rows = []
        for i in range(n_pieces-1):
            if field == "Bx":
                pars = [f"a_{der_order+1}_{k}" for k in range(self.poly_order + 1)]
            elif field == "By":
                pars = [f"b_{der_order+1}_{k}" for k in range(self.poly_order + 1)]
            else:  # Bs
                pars = [f"bs_{k}" for k in range(self.poly_order + 1)]

            idx_start = idx_extrema[i]
            idx_end = idx_extrema[i+1]
            s_start = self.s_full[idx_start]
            s_end = self.s_full[idx_end]

            for idx, name in enumerate(pars):
                rows.append({
                    "field_component": field,
                    "derivative_x": der_order,
                    "region_name": f"Poly_{i}",
                    "s_start": s_start,
                    "s_end": s_end,
                    "idx_start": idx_start,
                    "idx_end": idx_end,
                    "param_index": idx,
                    "param_name": name,
                    "param_symbol": sp.Symbol(name),
                    "param_value": 0 if n_pieces == 1 else None,
                })

        results = pd.DataFrame(rows)
        self.df_fit_pars = pd.concat([self.df_fit_pars, results])

        #with pd.option_context('display.max_columns', None, 'display.width', None):
        #    print(self.df_fit_pars)



    ####################################################################################################################
    # PIECEWISE POLYNOMIAL FITTING
    ####################################################################################################################

    # PRIVATE
    # This method computes the boundary conditions from a previously fitted polynomial.
    # xL, xR are the left and right boundaries of the new region.
    # dp and ddp are the first and second derivatives of the polynomial.
    def _boundary_from_poly(self, sL, poly):
        dp = poly.deriv()
        return np.array([poly(sL), dp(sL)], dtype=float)

    def _booundary_from_finite_differences(self, b_region, right_side=False):
        if right_side:
            dbL = (-3 * b_region[0] + 4 * b_region[1] - b_region[2]) / (2 * self.ds)
            return np.array([b_region[0], dbL], dtype=float)
        else:
            dbR = (3 * b_region[-1] - 4 * b_region[-2] + b_region[-3]) / (2 * self.ds)
            return np.array([b_region[-1], dbR], dtype=float)

    # To get a sub_df: sub_df = self.df_fit_pars.loc[
    #     (field, der_order)
    # ]

    # Desired logic:
    # - Loop over field and derivative order
    # - Within that, loop over regions
    # - For each region, extract the s-values and magnetic field values from the data
    # - Calculate the function value and derivative on the left side from the "previous" polynomial (pay mind to the left edge)
    # - Calculate the function value and derivative on the right side from the data using finite differences
    # - Calculate the integral over the region using trapezoidal rule
    # - Insert these coefficients into self._poly. This returns the polynomial in this region
    # - Assign the coefficients of this polynomial into the appropriate rows in self.df_fit_pars (a_1_0, a_1_1, ..., b_1_0, etc)

    # What this function needs to know:
    # - Raw data of the field/derivative combination in question, in the region, specified by start and end indices
    # - The s-values of the region in question, similarly defined by start and end indices
    # - The previous polynomial to get the left boundary conditions from
    # Need to make sure to calculate the right boundary conditions from the data itself, with finite differences around the correct index.

    # To Do:
    # We can do several things;
    # 1) Pass b_region, s_region and the previous polynomial to the function
    # 2) Extract them from self.df_on_axis_raw and self.s_full within the function, then we need to pass field, der_order, start_idx, end_idx
    # 3) Extract them from self.df_on_axis_raw and self.s_full within the function, but pass the sub_df already filtered for field, der_order and region
    # Think optino 3 makes the most sense.
    def _fit_single_poly(self, field, der_order, sub_df_this, sub_df_prev=None):
        idx_left = int(sub_df_this.index.get_level_values('idx_start')[0])
        idx_right = int(sub_df_this.index.get_level_values('idx_end')[0])
        s_left = int(sub_df_this.index.get_level_values('s_start')[0])
        s_right = int(sub_df_this.index.get_level_values('s_end')[0])

        s_region = self.s_full[idx_left:idx_right + 1]
        b_region = self.df_on_axis_raw[(field, der_order)].values[idx_left:idx_right + 1]
        integral = sc.integrate.trapezoid(b_region, s_region)

        if sub_df_prev is not None:
            coeff_prev = sub_df_prev['param_value'].iloc[:].values
            poly = self._poly(s_left, s_right, coeff_prev)
            left_bounds = self._boundary_from_poly(s_left, poly)
        else:
            left_bounds = self._booundary_from_finite_differences(b_region, right_side=True)

        right_bounds = self._booundary_from_finite_differences(b_region, right_side=False)
        coeffs = (left_bounds[0], left_bounds[1], right_bounds[0], right_bounds[1], integral)

        poly = self._poly(s_left, s_right, coeffs)

        # Assign coefficients into df_fit_pars
        for i in range(self.poly_order + 1):
            param_value = poly.convert().coef[i]  # get coefficient of s^i
            self.df_fit_pars.at[(field, der_order, sub_df_this['region_name'].iloc[0], s_left, s_right, idx_left, idx_right, i), 'param_value'] = param_value
            self.df_on_axis_fit[(field, der_order)].values[idx_left:idx_right + 1] = poly(s_region)


    # PRIVATE
    def _fit_edges(self):
        fields = ["Bx", "By"]
        if self.Bs_fit:
            fields.append("Bs")

        for field in fields:
            for der in range(0, self.deg + 1):
                sub_df = self.df_fit_pars.loc[(field, der)]
                sub_df.reset_index(level='region_name', inplace=True)
                n_regions = sub_df['region_name'].nunique()

                for i in range(n_regions):
                    sub_df_this = sub_df[sub_df['region_name'] == f"Poly_{i}"]
                    if i == 0:
                        sub_df_prev = None
                    else:
                        sub_df_prev = sub_df[sub_df['region_name'] == f"Poly_{i - 1}"]

                    self._fit_single_poly(field, der, sub_df_this, sub_df_prev)

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
            col_name = (f"{field}", der)
            # Ensure df_on_axis_raw exists and assign the derivative column
            self.df_on_axis_raw[col_name] = derivs[field]

        return derivs

    ####################################################################################################################
    # PLOTTING
    ####################################################################################################################

    def plot_integrated_fields(self):
        if self.df_on_axis_raw is None or self.df_on_axis_fit is None:
            raise RuntimeError("`df_on_axis_raw` and `df_on_axis_fit` must be set before plotting.")

        s = self.s_full

        Bx_raw = self.df_on_axis_raw[('Bx', 0)].to_numpy()
        By_raw = self.df_on_axis_raw[('By', 0)].to_numpy()
        try:
            Bs_raw = self.df_on_axis_raw[('Bs', 0)].to_numpy()
        except KeyError:
            Bs_raw = np.zeros_like(Bx_raw)

        Bx_fit = self.df_on_axis_fit[('Bx', 0)].to_numpy()
        By_fit = self.df_on_axis_fit[('By', 0)].to_numpy()
        try:
            Bs_fit = self.df_on_axis_fit[('Bs', 0)].to_numpy()
        except KeyError:
            Bs_fit = np.zeros_like(Bx_fit)

        fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)

        Bx_int_raw = sc.integrate.cumulative_trapezoid(Bx_raw, x=s, initial=0)
        By_int_raw = sc.integrate.cumulative_trapezoid(By_raw, x=s, initial=0)
        Bs_int_raw = sc.integrate.cumulative_trapezoid(Bs_raw, x=s, initial=0)

        Bx_int_fit = sc.integrate.cumulative_trapezoid(Bx_fit, x=s, initial=0)
        By_int_fit = sc.integrate.cumulative_trapezoid(By_fit, x=s, initial=0)
        Bs_int_fit = sc.integrate.cumulative_trapezoid(Bs_fit, x=s, initial=0)

        ax1.plot(s, Bx_int_raw, label='Raw Data')
        ax1.plot(s, Bx_int_fit, label='Fit', linestyle='--')
        ax2.plot(s, By_int_raw, label='Raw Data')
        ax2.plot(s, By_int_fit, label='Fit', linestyle='--')
        ax3.plot(s, Bs_int_raw, label='Raw Data')
        ax3.plot(s, Bs_int_fit, label='Fit', linestyle='--')

        for field_ax in ["Bx", "By", "Bs"]:
            ax = {"Bx": ax1, "By": ax2, "Bs": ax3}[field_ax]
            for idx in getattr(self, "borders_idx", []):
                if 0 <= idx < len(s):
                    ax.axvline(x=s[idx], color='k', linestyle='--', linewidth=1)

        ax1.set_title(f"Integrated Magnetic Field at (X, Y) = {self.xy_point}")
        ax1.set_ylabel(r"Integrated Horizontal Field, $\int B_x \, ds$ [T·m]")
        ax2.set_ylabel(r"Integrated Vertical Field, $\int B_y \, ds$ [T·m]")
        ax3.set_ylabel(r"Integrated Longitudinal Field, $\int B_s \, ds$ [T·m]")
        ax3.set_xlabel(r"Longitudinal Position, $s$ [m]")

        ax1.legend(loc="lower right")
        ax2.legend(loc="lower right")
        ax3.legend(loc="upper right")

        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.show()

    # PUBLIC
    # Plot the data against the fit.
    def plot_fields(self, der=0):
        if self.df_on_axis_raw is None or self.df_on_axis_fit is None:
            raise RuntimeError("`df_on_axis_raw` and `df_on_axis_fit` must be set before plotting.")

        s = self.s_full
        fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)

        def get_series(df, field, der):
            try:
                return df[(field, der)].to_numpy()
            except KeyError:
                # fallback to zeros if Bs not present or derivative missing
                ref = df.iloc[:, 0].to_numpy()
                return np.zeros_like(ref)

        ax1.plot(s, get_series(self.df_on_axis_raw, "Bx", der), label='Raw Data')
        ax1.plot(s, get_series(self.df_on_axis_fit, "Bx", der), label='Fit')
        ax2.plot(s, get_series(self.df_on_axis_raw, "By", der), label='Raw Data')
        ax2.plot(s, get_series(self.df_on_axis_fit, "By", der), label='Fit')
        ax3.plot(s, get_series(self.df_on_axis_raw, "Bs", der), label='Raw Data')
        ax3.plot(s, get_series(self.df_on_axis_fit, "Bs", der), label='Fit')

        for field_ax in ["Bx", "By", "Bs"]:
            ax = {"Bx": ax1, "By": ax2, "Bs": ax3}[field_ax]
            for idx in getattr(self, "borders_idx", []):
                if 0 <= idx < len(s):
                    ax.axvline(x=s[idx], color='k', linestyle='--', linewidth=1)

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