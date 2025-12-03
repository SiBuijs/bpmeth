from __future__ import annotations

import numpy as np
import pandas as pd
import scipy as sc
import sympy as sp
import xtrack as xt
import bisect
import math
from functools import partial

import bpmeth as bp
import time
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import cProfile

from sympy.utilities.codegen import codegen



class FieldFitter:

    def __init__(
            self,
            file_path,
            xy_point=(0, 0),
            dx=0.001,
            dy=0.001,
            ds=0.001,
            min_region_size=10,
            deg=2,
    ):

        # Parameters
        self.file_path = file_path
        self.xy_point = xy_point
        self.dx, self.dy, self.ds = dx, dy, ds
        self.poly_order = 4  # fixed at 4 for now (5 coefficients)
        self.min_region_size = min_region_size
        self.s_full = None
        self.length = None
        self.deg = deg
        self.field_tol = 1e-3

        # DataFrames
        self.df_raw_data = None
        self.df_on_axis_raw  = None
        self.df_on_axis_fit = None
        self.df_fit_pars = None

    # PUBLIC
    # Setter method that calls all the other methods to arrive at a fit.
    def set(self):
        self._parse_to_dataframe()
        self._set_df_on_axis()
        self._find_regions()
        self._fit_slices()



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
    # TODO: Consider making this dynamic in self.poly_order.
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
    # This method reads the data from the file and stores it in a pandas DataFrame..
    def _parse_to_dataframe(self):
        df = pd.read_csv(
            self.file_path, sep=r"\s+", header=None, names=["X", "Y", "Z", "Bx", "By", "Bs"]
        )
        df.set_index(["X", "Y", "Z"], inplace=True)
        self.df_raw_data = df
        self.s_full = np.sort(df.index.get_level_values("Z").unique()).astype(float) * self.ds

        # Check if Bs is much smaller than Bx and By
        # Sets an additional index der = 0.
        der = 0
        df_on = self.df_raw_data.xs(self.xy_point, level=("X", "Y")).sort_index().copy(deep=True)
        # convert columns to MultiIndex (field, derivative)
        df_on.columns = pd.MultiIndex.from_tuples([(col, der) for col in df_on.columns])
        self.df_on_axis_raw = df_on

    def save_fit_pars(self, file_path):
        self.df_fit_pars.to_csv(file_path, index=True)

    # PRIVATE
    # This method extracts on-axis data from the raw DataFrame and fits it to polynomials.
    # It computes the derivatives of said polynomials and stores them in the self.df_on_axis_raw DataFrame.
    # The data is not "raw" in the technical sense, but is used to fit a function of s to.
    def _set_df_on_axis(self):
        # 0th derivative columns
        self.df_on_axis_raw.columns = pd.MultiIndex.from_tuples([
                    (col[0] if isinstance(col, tuple) else col, 0) for col in self.df_on_axis_raw.columns
                ])
        # compute transverse derivatives for der > 0 and add as columns (skip Bs derivatives)
        for der in range(1, self.deg + 1):
            derivs = self._fit_transverse_polynomials(der=der)
            self.df_on_axis_raw[('Bx', der)] = derivs['Bx']
            self.df_on_axis_raw[('By', der)] = derivs['By']
            # intentionally do not compute/store Bs_{der}, as we are not using them.

        # create a zeros-only DataFrame with the same index/columns as the on-axis raw data
        self.df_on_axis_fit = self.df_on_axis_raw.copy(deep=True)
        # set all values to 0.0 while preserving index and column structure
        self.df_on_axis_fit.loc[:, :] = 0.0

    # PRIVATE
    # This method loops over all fields and derivatives.
    # It first checks if a field/derivative needs fitting based on its maximum value compared to the maximum of the main field.
    # It finds peaks and valleys in the data within the peak_window, with specified width and prominence.
    # It uses these extrema to define regions for polynomial fitting.
    # Then, it cuts regions if they span too wide a range.
    # Finally, it stores the regions in the df_fit_pars DataFrame.
    def _find_regions(self):
        fields = ["Bx", "By", "Bs"]

        abs_max = 0
        for field in fields:
            series = self.df_on_axis_raw[(field, 0)].values
            field_max = np.max(np.abs(series))
            if field_max > abs_max:
                abs_max = field_max

        for field in fields:
            # Bs only has der = 0; other fields range 0..deg
            ders = [0] if field == "Bs" else range(0, self.deg + 1)

            for der in ders:
                series = self.df_on_axis_raw[(field, der)].values

                # FIELD TOLERANCE AREA: check if this field/derivative needs fitting
                field_der_max = np.max(np.abs(series))
                relative_max = 1 / math.factorial(der) * field_der_max * (self.dx ** der)
                if relative_max < self.field_tol * abs_max:
                    # set to single region with zero parameters and skip expensive processing
                    field_extrema = np.array([0, len(series) - 1], dtype=int)
                    to_fit = False
                else:
                    # SPLIT REGIONS AREA
                    # choose prominence: more permissive for Bs
                    std_series = np.std(series)
                    prominence = 0.5 * std_series if field == "Bs" else std_series

                    field_peaks = find_peaks(series, width=15, prominence=prominence)[0]
                    field_valleys = find_peaks(-series, width=15, prominence=prominence)[0]
                    field_extrema = np.sort(np.concatenate((field_peaks, field_valleys)))

                    # include endpoints
                    field_extrema = np.insert(field_extrema, 0, 0)
                    field_extrema = np.append(field_extrema, len(series) - 1)

                    # split long regions while ensuring each part has at least `min_region_size` points
                    this_min_region_size = self.min_region_size if der == 0 else self.min_region_size // 2
                    new_extrema = [int(field_extrema[0])]
                    for left, right in zip(field_extrema[:-1], field_extrema[1:]):
                        length = int(right - left)
                        if length < 2 * this_min_region_size:
                            new_extrema.append(int(right))
                            continue
                        n_parts = int(np.floor(length / this_min_region_size))
                        if n_parts <= 1:
                            new_extrema.append(int(right))
                            continue
                        splits = np.round(np.linspace(left, right, n_parts + 1)).astype(int)
                        for sp in splits[1:]:
                            if sp > new_extrema[-1]:
                                new_extrema.append(int(sp))

                    field_extrema = np.unique(np.asarray(new_extrema, dtype=int))
                    to_fit = True

                # number of pieces is number of extrema - 1 (ensure at least 1)
                n_pieces = max(1, len(field_extrema) - 1)
                print(f"{field} der={der} -> n_pieces={n_pieces}")
                self._set_df_fit_pars(der, n_pieces, field, field_extrema, to_fit)


        self.df_fit_pars.set_index(['field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start', 'idx_end', 'param_index'],
                                       inplace=True)

        # ensure MultiIndex is lexsorted so partial-key .loc lookups (e.g. .loc[(field, der)]) are fast and avoid PerformanceWarning
        if not self.df_fit_pars.empty:
            self.df_fit_pars.sort_index(inplace=True)

        #with pd.option_context('display.max_columns', None, 'display.max_rows', None, 'display.width', None):
            #print(self.df_fit_pars)

    # PRIVATE
    # This method initializes and appends rows to the df_fit_pars DataFrame.
    # Each row corresponds to a polynomial piece for a specific field and derivative.
    # It stores metadata about the piece, including parameter names and initial values.
    # This method is called by _find_regions to populate the DataFrame.
    # In case the set consists of only one piece, the parameters are initialized to 0.
    def _set_df_fit_pars(self, der_order, n_pieces, field, idx_extrema, to_fit=True):
        rows = []
        for i in range(n_pieces):
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
                    "param_value": 0 if not to_fit else None,
                    "to_fit": to_fit,
                })

        results = pd.DataFrame(rows)
        self.df_fit_pars = pd.concat([self.df_fit_pars, results])



    ####################################################################################################################
    # PIECEWISE POLYNOMIAL FITTING
    ####################################################################################################################

    # PRIVATE
    # This method computes the boundary conditions from a previously fitted polynomial.
    # Accepts the polynomial and the position sL where to evaluate it (we fit from left to right, so always the leftmost point).
    def _boundary_from_poly(self, sL, poly):
        dp = poly.deriv()
        return np.array([poly(sL), dp(sL)], dtype=float)

    # PRIVATE
    # This method computes the boundary conditions from finite differences in the specified region.
    def _boundary_from_finite_differences(self, b_region, get_right_point=True):
        if get_right_point:
            dbR = (3 * b_region[-1] - 4 * b_region[-2] + b_region[-3]) / (2 * self.ds)
            return np.array([b_region[-1], dbR], dtype=float)
        else:
            dbL = (-3 * b_region[0] + 4 * b_region[1] - b_region[2]) / (2 * self.ds)
            return np.array([b_region[0], dbL], dtype=float)

    # PRIVATE
    # This method fits a single polynomial piece to the specified region of data.
    def _fit_single_poly(self, field, der_order, sub_df_this, sub_df_prev=None):
        idx_left = int(sub_df_this.index.get_level_values('idx_start')[0])
        idx_right = int(sub_df_this.index.get_level_values('idx_end')[0])
        s_left = float(sub_df_this.index.get_level_values('s_start')[0])
        s_right = float(sub_df_this.index.get_level_values('s_end')[0])

        s_region = self.s_full[idx_left:idx_right + 1]
        b_region = self.df_on_axis_raw[(field, der_order)].values[idx_left:idx_right + 1]
        integral = sc.integrate.trapezoid(b_region, s_region)

        if sub_df_prev is not None:
            coeff_prev = sub_df_prev['param_value'].iloc[:].values
            poly = np.polynomial.Polynomial(coeff_prev)
            left_bounds = self._boundary_from_poly(s_left, poly)
        else:
            left_bounds = self._boundary_from_finite_differences(b_region, get_right_point=False)

        right_bounds = self._boundary_from_finite_differences(b_region, get_right_point=True)
        coeffs = (left_bounds[0], left_bounds[1], right_bounds[0], right_bounds[1], integral)

        poly = self._poly(s_left, s_right, coeffs)

        # Assign coefficients into df_fit_pars
        for i in range(self.poly_order + 1):
            param_value = poly.convert().coef[i]  # get coefficient of s^i
            self.df_fit_pars.at[(field, der_order, sub_df_this['region_name'].iloc[0], s_left, s_right, idx_left, idx_right, i), 'param_value'] = param_value
            self.df_on_axis_fit[(field, der_order)].values[idx_left:idx_right + 1] = poly(s_region)

    # PRIVATE
    # This method loops over all fields and derivatives and fits polynomials to each region.
    def _fit_slices(self):

        for field in ["Bx", "By", "Bs"]:
            for der in range(0, self.deg + 1):
                if field == "Bs" and der > 0:
                    continue

                print(f"Fitting field {field} derivative {der}")
                sub_df = self.df_fit_pars.loc[(field, der)]

                if not sub_df['to_fit'].any():
                    continue

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

    # PUBLIC
    # Plot the integrated fields.
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

            # Vertical border lines removed

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
    # der: derivative order to plot (0 = field, 1 = first derivative, etc.)
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
        ax1.plot(s, get_series(self.df_on_axis_fit, "Bx", der), label='Fit', linestyle='--')
        ax2.plot(s, get_series(self.df_on_axis_raw, "By", der), label='Raw Data')
        ax2.plot(s, get_series(self.df_on_axis_fit, "By", der), label='Fit', linestyle='--')
        ax3.plot(s, get_series(self.df_on_axis_raw, "Bs", der), label='Raw Data')
        ax3.plot(s, get_series(self.df_on_axis_fit, "Bs", der), label='Fit', linestyle='--')

        # compute border indices per field/derivative (fall back to existing attribute if absent)
        def _borders_for_field(field_ax):
            if getattr(self, "df_fit_pars", None) is None:
                return getattr(self, "borders_idx", []) or []
            try:
                lvl_field = np.asarray(self.df_fit_pars.index.get_level_values('field_component'))
                lvl_der = np.asarray(self.df_fit_pars.index.get_level_values('derivative_x')).astype(int)
                mask = (lvl_field == field_ax) & (lvl_der == int(der))
                if not np.any(mask):
                    return []
                s_start_vals = np.asarray(self.df_fit_pars.index.get_level_values('s_start'))[mask].astype(float)
                s_end_vals = np.asarray(self.df_fit_pars.index.get_level_values('s_end'))[mask].astype(float)
                s_borders = np.unique(np.concatenate((s_start_vals, s_end_vals)))
                s_arr = np.asarray(s)
                return sorted({int(np.argmin(np.abs(s_arr - float(sb)))) for sb in s_borders})
            except Exception:
                return getattr(self, "borders_idx", []) or []

        for field_ax in ["Bx", "By", "Bs"]:
            ax = {"Bx": ax1, "By": ax2, "Bs": ax3}[field_ax]
            borders_idx_field = _borders_for_field(field_ax)
            for idx in borders_idx_field or []:
                if 0 <= idx < len(s):
                    ax.axvline(x=s[idx], color='k', linestyle='--', linewidth=1, alpha=0.3)

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


# TODO: Investigate how lamdified functions work. Do they accept arguments based on name or something?
# TODO: Build a good slice detector, that takes the correct magnetic field for each component depending on s0.
# TODO: Look at difference between lambdify, or C-code generation.

# LOGIC:
# 1) We already have the symbolic and lambdified expressions.
# 2) We extract the s_start and s_end for each segment from FieldFitter.df_fit_pars.
# 3) We apply the mask to get all rows where s_start <= s < s_end.
# 4) When looping, we must somehow remember the previous mask.
# 5) If the new mask is the same as the previous one, we are in the same segment.
#    We can reuse the previous coefficients, which we do using from functools import partial.
# 6) If the new mask is different, we extract the new coefficients from df_fit_pars.
#    We also create a partial function, which is used in the next iteration if the mask does not change again.

# Workflow:
# 1) A get_Bfield function, which receives an (x, y, s).
# 2) Passes these to a slice or segment selector. This chooses the correct coefficients for each field component, depending on where the segment starts.
#    This function returns a dict of the coefficients for each field component.
# 3) These dicts are passed to the generic lamdified function, as well as the coordinates.
# 4) The lamdified function returns the field components at that point.
class SymbolicGenerator:
    def __init__(self, FieldFitter, curv=0):
        self.FieldFitter = FieldFitter
        self.poly_order = FieldFitter.poly_order
        self.deg = FieldFitter.deg
        self.curv = curv
        self.coord_symbols = sp.symbols("x y s")
        self.param_symbols = None
        self.param_names   = None
        self.param_map     = None

        self.a_dict = {}
        self.b_dict = {}
        self.bs_dict = {}

        # Symbolic expressions
        self.symbolic_Ax = None
        self.symbolic_Ay = None
        self.symbolic_As = None
        self.symbolic_Bx = None
        self.symbolic_By = None
        self.symbolic_Bs = None

        # Common sub-expressions
        self.A_cse_subs = None
        self.B_cse_subs = None

        # Reduced expressions
        self.A_reduced_exprs = None
        self.B_reduced_exprs = None

        self._get_parameters()
        self._set_symbolic_exprs()
        self._get_reduced_expressions()

    # PRIVATE
    # Get the list of parameter symbols/names from FieldFitter.df_fit_pars.
    def _get_parameters(self):
        all_symbols = list(self.FieldFitter.df_fit_pars["param_symbol"].to_list())
        param_map = {}
        for sym in all_symbols:
            if sym.name not in param_map:
                param_map[sym.name] = sym
        # store mapping and ordered tuples for later use
        self.param_map = param_map
        self.param_symbols = tuple(param_map.values())
        self.param_names = tuple(param_map.keys())

        # Set the dicts per parameter type
        # a_dict <=> a_{d}_{k}, which are the Bx coefficients
        # b_dict <=> b_{d}_{k}, which are the By coefficients
        # bs_dict <=> bs_{k}, which are the Bs coefficients
        for name in self.param_names:
            if name.startswith('a'):
                self.a_dict[name] = self.param_map[name]
            elif name.startswith('bs'):
                self.bs_dict[name] = self.param_map[name]
            elif name.startswith('b'):
                self.b_dict[name] = self.param_map[name]
            else:
                pass

    # PRIVATE
    # Works as desired.
    def _set_symbolic_exprs(self):
        # create per-derivative polynomial coefficient symbols and expressions (degree 4 -> 5 terms)
        a_poly_exprs_list = []
        b_poly_exprs_list = []
        s = sp.Symbol("s")

        a_syms  = list(self.a_dict.values())
        b_syms  = list(self.b_dict.values())
        bs_syms = list(self.bs_dict.values())

        a_keys  = list(self.a_dict.keys())
        b_keys  = list(self.b_dict.keys())
        bs_keys = list(self.bs_dict.keys())

        # group coefficients by derivative index d (names expected like 'a_d_k')
        a_groups = {}
        for i, name in enumerate(a_keys):
            parts = name.split('_')
            if len(parts) >= 3:
                d = int(parts[1])
                k = int(parts[2])
                a_groups.setdefault(d, {})[k] = a_syms[i]

        # build one polynomial per derivative (ordered by d)
        for d in sorted(a_groups.keys()):
            terms = [a_groups[d][k] * s ** k for k in sorted(a_groups[d].keys())]
            a_poly_exprs_list.append(sum(terms))

        # same for b (names expected like 'b_d_k')
        b_groups = {}
        for i, name in enumerate(b_keys):
            parts = name.split('_')
            if len(parts) >= 3:
                d = int(parts[1])
                k = int(parts[2])
                b_groups.setdefault(d, {})[k] = b_syms[i]

        for d in sorted(b_groups.keys()):
            terms = [b_groups[d][k] * s ** k for k in sorted(b_groups[d].keys())]
            b_poly_exprs_list.append(sum(terms))

        # bs is single-set (names like 'bs_k'); build single polynomial
        bs_terms = []
        for i, name in enumerate(bs_keys):
            parts = name.split('_')
            k = int(parts[-1]) if len(parts) >= 2 else i
            bs_terms.append(bs_syms[i] * s ** k)

        bs_poly_exprs_list = sum(bs_terms) if bs_terms else sp.Integer(0)

        a_poly_exprs_strings = tuple(f"{expr}" for expr in a_poly_exprs_list)
        b_poly_exprs_strings = tuple(f"{expr}" for expr in b_poly_exprs_list)
        bs_poly_exprs_string = f"{bs_poly_exprs_list}"

        generic_poly_bpmeth = bp.GeneralVectorPotential(hs=f"{self.curv}", a=a_poly_exprs_strings,
                                                        b=b_poly_exprs_strings, bs=bs_poly_exprs_string)
        self.symbolic_Bx, self.symbolic_By, self.symbolic_Bs = generic_poly_bpmeth.get_Bfield(lambdify=False)
        self.symbolic_Ax, self.symbolic_Ay, self.symbolic_As = generic_poly_bpmeth.get_A()

    # PRIVATE
    # The resulting expressions typically have many common sub-expressions.
    # This method uses sympy.cse to identify and extract these common sub-expressions.
    # It is expected to be a small optimization step.
    def _get_reduced_expressions(self):
        A_exprs = [self.symbolic_Ax, self.symbolic_Ay, self.symbolic_As]
        self.A_cse_subs, self.A_reduced_exprs = sp.cse(A_exprs)
        B_exprs = [self.symbolic_Bx, self.symbolic_By, self.symbolic_Bs]
        self.B_cse_subs, self.B_reduced_exprs = sp.cse(B_exprs)

    # PUBLIC
    # This method writes the symbolic expressions to a Python file.
    # The generated file contains a function that evaluates the field components given (x, y, s) and parameters.
    # It utilizes common sub-expressions for efficiency.
    def write_to_python(self, field='B'):
        if field == 'A':
            cse_subs = self.A_cse_subs
            reduced_exprs = self.A_reduced_exprs
            filename = 'A_field_eval.py'
        else:
            cse_subs = self.B_cse_subs
            reduced_exprs = self.B_reduced_exprs
            filename = 'B_field_eval.py'

        with open(filename, 'w') as f:
            f.write(f"# Auto-generated symbolic field expressions for {field}\n")

            f.write(f"from operator import itemgetter\n")
            f.write(f"def evaluate_{field}(x, y, s, **params):\n")

            f.write("    # Parameter assignments\n")
            arg_list = ", ".join(f"'{n}'" for n in self.param_names)
            names = ", ".join(self.param_names)
            f.write(f"    {names} = itemgetter({arg_list})(params)\n")
            f.write("\n")

            f.write("    # Common sub-expressions\n")
            for lhs, rhs in cse_subs:
                f.write(f"    {lhs} = {rhs}\n")
            f.write("\n")
            f.write("    # Reduced expressions\n")
            names = [f'{field}x', f'{field}y', f'{field}s']
            for name, expr in zip(names, reduced_exprs):
                f.write(f"    {name} = {expr}\n")
            f.write("\n")

            f.write(f"    return {field}x, {field}y, {field}s\n")

    def write_to_c(self, field='B'):
        from sympy.printing.c import C99CodePrinter

        from sympy import Pow

        class MulPowerPrinter(C99CodePrinter):
            def _print_Pow(self, expr):
                base, exp = expr.as_base_exp()

                # Only rewrite integer positive powers
                if exp.is_integer and exp.is_positive:
                    n = int(exp)
                    return "*".join([self._print(base)] * n)

                # Fallback to default handling
                return super()._print_Pow(expr)

        printer = MulPowerPrinter()

        if field == 'A':
            cse_subs = self.A_cse_subs
            reduced_exprs = self.A_reduced_exprs
            filename = 'A_field_eval.c'
        else:
            cse_subs = self.B_cse_subs
            reduced_exprs = self.B_reduced_exprs
            filename = 'B_field_eval.c'

        with open(filename, 'w') as f:
            f.write(f"#include <math.h>\n")
            f.write(f"#include <stddef.h>\n\n")

            f.write(f"// Auto-generated symbolic field expressions for {field}\n")
            f.write(f"void evaluate_{field}(const double x_array[], const double y_array[], const double s_array[], size_t n, const double **params, double {field}x_out[], double {field}y_out[], double {field}s_out[]){{\n")
            names = [f'{field}x_out', f'{field}y_out', f'{field}s_out']
            f.write("\tfor (size_t ii = 0; ii < n; ++ii) {\n")
            f.write("\t\tconst double x = x_array[ii];\n")
            f.write("\t\tconst double y = y_array[ii];\n")
            f.write("\t\tconst double s = s_array[ii];\n\n")

            f.write("\t\t// Parameter List\n")
            for j, name in enumerate(self.param_names):
                f.write(f"\t\tconst double {name} = params[ii][{j}];\n")
            f.write("\n")

            f.write("\t\t// Common sub-expressions\n")
            for lhs, rhs in cse_subs:
                f.write(f"\t\tconst double {lhs} = {printer.doprint(rhs)};\n")
            f.write("\n")

            f.write("\t\t// Reduced expressions\n")
            for i, expr in enumerate(reduced_exprs):
                f.write(f"\t\t{names[i]}[ii] = {expr};\n")
            f.write("}\n")
            f.write("}")

    def compile_C_code(self, field='B'):
        from cffi import FFI
        import sys

        ffibuilder = FFI()

        ffibuilder.cdef(f"void evaluate_{field}(const double x_array[], const double y_array[], const double s_array[], size_t n, const double **params, double {field}x_out[], double {field}y_out[], double {field}s_out[]);")
        # Read the C file
        with open(f"{field}_field_eval.c") as f:
            c_source = f.read()

        # --- Platform-specific optimizer flags ---
        common_flags = ["-O3", "-fomit-frame-pointer"]

        # Fast math optimizations (enable only if safe for your physics model)
        fast_math = ["-ffast-math"]

        # CPU-specific optimizations (best performance, but not portable across machines)
        march_native = ["-march=native"]

        extra_compile_args = common_flags + fast_math + march_native

        # Windows uses MSVC; flags are different
        #if sys.platform == "win32":
            # MSVC equivalents:
            # /O2 = optimize, /fp:fast = fast math
            # extra_compile_args = ["/O2", "/fp:fast"]

        ffibuilder.set_source(
            f"_field_{field}",  # Name of the generated Python extension module
            c_source,  # Include the C code
            libraries=[],  # No external libraries needed
            extra_compile_args=extra_compile_args,
        )

        ffibuilder.compile(verbose=True)

class FieldCalculator:
    def __init__(self, df_fit_pars=None, filepath=None):
        import importlib
        import B_field_eval
        import A_field_eval
        try:
            from _field_B import ffi as _ffi_B, lib as _lib_B
        except Exception:
            _ffi_B = None
            _lib_B = None
        try:
            from _field import ffi as _ffi_A, lib as _lib_A
        except Exception:
            _ffi_A = None
            _lib_A = None
        importlib.reload(B_field_eval)
        importlib.reload(A_field_eval)
        self.B_field_eval = B_field_eval
        self.A_field_eval = A_field_eval
        self._ffi_B = _ffi_B
        self._lib_B = _lib_B
        self._ffi_A = _ffi_A
        self._lib_A = _lib_A

        # For the integrator
        self.integrator = []
        self.n_slices = 1000
        self.n_steps = 1000

        if df_fit_pars is not None:
            self.df = self._rework_dataframe(df_fit_pars)
        if filepath is not None:
            self.df = pd.read_csv(filepath, index_col=['field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start', 'idx_end', 'param_index'])
            self.df = self._rework_dataframe(self.df)

        self.s_start = self.df['s_start'].to_numpy()
        self.s_end = self.df['s_end'].to_numpy()
        self.s_boundaries = np.unique(np.concatenate((self.s_start, self.s_end)))
        self.s_mid = (self.s_boundaries[:-1] + self.s_boundaries[1:]) / 2

        # TODO: Remove this dict, can just use lists/tuples.
        self._np_par_cache = {
            "param_name": self.df["param_name"].to_numpy(),
            "param_value": self.df["param_value"].to_numpy(),
        }

        self._contruct_par_table()

    def _contruct_par_table(self):
        par_dicts = []
        par_table = []
        for i in range(len(self.s_boundaries) - 1):
            s_mid_i = self.s_mid[i]
            mask = (self.s_start <= s_mid_i) & (self.s_end > s_mid_i)
            names = self._np_par_cache["param_name"][mask]
            vals = self._np_par_cache["param_value"][mask]
            par_dicts.append(dict(zip(names, vals)))
            par_table.append(list(vals))
        self.par_dicts = par_dicts
        self.par_table = np.ascontiguousarray(par_table, dtype=np.double)

    @staticmethod
    def _rework_dataframe(df_fit_pars):
        # reset index to make s_start and s_end columns
        df = df_fit_pars.reset_index()
        # create a MultiIndex with field_component, derivative_x, region_name
        df.set_index(['field_component', 'derivative_x', 'region_name'], inplace=True)
        return df

    #@profile
    def _select_region(self, s_val):
        import numpy as _np
        sb = self.s_boundaries
        # Use numpy.searchsorted which accepts scalars and arrays
        idxs = _np.searchsorted(sb, s_val, side='right') - 1
        # Clamp to valid region indices [0, n_regions-1]
        max_idx = len(sb) - 2
        idxs = _np.clip(idxs, 0, max_idx)
        if _np.isscalar(s_val):
            return int(idxs)
        return idxs

    #@profile
    def get_Bfield(self, x_arr, y_arr, s_arr, python=False):
        idxs = self._select_region(s_arr)
        #param_dict = self._par_dicts[int(idxs)]
        #param_table = self.par_table[i] for i in idxs

        # TODO: This is probably not correct right now, need to change it later. Focus on C first.
        if python:
            Bx_out, By_out, Bs_out = self.B_field_eval.evaluate_B(x_arr, y_arr, s_arr, **self.par_table)

        else:
            # Ensure contiguous double arrays
            x = np.ascontiguousarray(x_arr, dtype=np.double)
            y = np.ascontiguousarray(y_arr, dtype=np.double)
            s = np.ascontiguousarray(s_arr, dtype=np.double)

            params = np.ascontiguousarray(self.par_table[idxs])

            ffi = self._ffi_B

            n = s.size

            # Build array of pointers
            params_ptrs = [None] * n
            for i in range(n):
                row = self.par_table[idxs[i]]
                params_ptrs[i] = ffi.cast("const double *", ffi.from_buffer(row))

            # Now make a real C array of pointers:
            params_c = ffi.new("const double*[]", params_ptrs)

            # Output buffer
            Bx_out = np.empty(n, dtype=np.double)
            By_out = np.empty(n, dtype=np.double)
            Bs_out = np.empty(n, dtype=np.double)

            # Get C pointers (use const for input arrays)
            x_c = self._ffi_B.cast("const double *", self._ffi_B.from_buffer(x))
            y_c = self._ffi_B.cast("const double *", self._ffi_B.from_buffer(y))
            s_c = self._ffi_B.cast("const double *", self._ffi_B.from_buffer(s))

            #params_c = self._ffi_B.cast(f"const double (*)[{n_params}]", self._ffi_B.from_buffer(params))

            bx_c = self._ffi_B.cast("double *", self._ffi_B.from_buffer(Bx_out))
            by_c = self._ffi_B.cast("double *", self._ffi_B.from_buffer(By_out))
            bs_c = self._ffi_B.cast("double *", self._ffi_B.from_buffer(Bs_out))

            # Call the C function
            self._lib_B.evaluate_B(x_c, y_c, s_c, n, params_c, bx_c, by_c, bs_c)

        return Bx_out, By_out, Bs_out

    def get_vector_potential(self, x, y, s, python=True):
        import numpy as _np
        # Detect scalar vs array mode
        scalar_mode = _np.isscalar(s) and _np.isscalar(x) and _np.isscalar(y)

        if scalar_mode:
            idx = self._select_region(s)
            param_dict = self.par_dicts[int(idx)]

            if python:
                Ax, Ay, As = self.A_field_eval.evaluate_A(x, y, s, **param_dict)
            else:
                ffi = self._ffi_A or __import__('_field').ffi
                lib = self._lib_A or __import__('_field').lib

                params = _np.array(list(param_dict.values()), dtype=_np.double)
                params_c = ffi.cast("double*", params.ctypes.data)

                result = lib.evaluate_A(x, y, s, params_c)
                Ax = result.Ax
                Ay = result.Ay
                As = result.As

            return Ax, Ay, As

        # Array mode: ensure contiguous numpy arrays
        x_arr = _np.ascontiguousarray(_np.atleast_1d(x), dtype=_np.double)
        y_arr = _np.ascontiguousarray(_np.atleast_1d(y), dtype=_np.double)
        s_arr = _np.ascontiguousarray(_np.atleast_1d(s), dtype=_np.double)

        idxs = self._select_region(s_arr)

        if python:
            n = s_arr.size
            Ax_out = _np.empty(n, dtype=_np.double)
            Ay_out = _np.empty(n, dtype=_np.double)
            As_out = _np.empty(n, dtype=_np.double)
            for i in range(n):
                pd = self.par_dicts[int(idxs[i])]
                Ax_out[i], Ay_out[i], As_out[i] = self.A_field_eval.evaluate_A(x_arr[i], y_arr[i], s_arr[i], **pd)
            return Ax_out, Ay_out, As_out

        # C path (array)
        ffi = self._ffi_A or __import__('_field_A').ffi
        lib = self._lib_A or __import__('_field_A').lib

        n = s_arr.size
        params_ptrs = [None] * n
        for i in range(n):
            row = self.par_table[idxs[i]]
            params_ptrs[i] = ffi.cast("const double *", ffi.from_buffer(row))

        params_c = ffi.new("const double*[]", params_ptrs)

        Ax_out = _np.empty(n, dtype=_np.double)
        Ay_out = _np.empty(n, dtype=_np.double)
        As_out = _np.empty(n, dtype=_np.double)

        x_c = ffi.cast("const double *", ffi.from_buffer(x_arr))
        y_c = ffi.cast("const double *", ffi.from_buffer(y_arr))
        s_c = ffi.cast("const double *", ffi.from_buffer(s_arr))

        ax_c = ffi.cast("double *", ffi.from_buffer(Ax_out))
        ay_c = ffi.cast("double *", ffi.from_buffer(Ay_out))
        as_c = ffi.cast("double *", ffi.from_buffer(As_out))

        lib.evaluate_A(x_c, y_c, s_c, n, params_c, ax_c, ay_c, as_c)
        return Ax_out, Ay_out, As_out

    # TODO: Removed the for loop, I believe it's no longer necessary; lookup is done in get_Bfield.
    # TODO: But not sure if it's really correct this way.
    def set_integrator(self):
         self.integrator = xt.BorisSpatialIntegrator(fieldmap_callable=self.get_Bfield, s_start=self.s_start[0], s_end=self.s_end[-1],
                                             n_steps=np.round(self.n_steps / self.n_slices).astype(int),
                                             verbose=True)

    # TODO: Removed the for loop, I believe it's no longer necessary; lookup is done in get_Bfield.
    # TODO: But not sure if it's really correct this way.
    def get_line(self):

        if self.integrator == []:
            self.set_integrator()

        self.env = xt.Environment()

        self.env.elements[f'wig'] = self.integrator

        self.wiggler_line = self.env.new_line(components=['wig'])

        return self.wiggler_line

    # PUBLIC
    # Plot the B field along s at a given (x, y) point.
    # Seems to function as desired and the plotted data seems correct.
    def plot_B_field(self, x_arr=0, y_arr=0, n_points=2000, plot_data=False, python=False):
        s_min = self.s_boundaries[0]
        s_max = self.s_boundaries[-2]
        s_vals = np.linspace(s_min, s_max, n_points)

        if type(x_arr) is not np.ndarray or list:
            x_arr = np.full(n_points, x_arr)
        if type(y_arr) is not np.ndarray or list:
            y_arr = np.full(n_points, y_arr)

        Bx_vals, By_vals, Bs_vals = self.get_Bfield(x_arr, y_arr, s_vals, python=python)

        plt.figure(figsize=(10, 6))
        plt.plot(s_vals, Bx_vals, label='Bx')
        plt.plot(s_vals, By_vals, label='By')
        plt.plot(s_vals, Bs_vals, label='Bs')
        plt.xlabel('s [m]')
        plt.ylabel('Magnetic Field [T]')
        plt.title(f'Magnetic Field at (x={x_arr[0]}, y={y_arr[0]})')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_A_field(self, x_arr=0, y_arr=0, n_points=2000, python=False):
        s_min = self.s_boundaries[0]
        s_max = self.s_boundaries[-2]
        s_vals = np.linspace(s_min, s_max, n_points)

        if not isinstance(x_arr, (np.ndarray, list)):
            x_arr = np.full(n_points, x_arr)
        if not isinstance(y_arr, (np.ndarray, list)):
            y_arr = np.full(n_points, y_arr)

        Ax_vals, Ay_vals, As_vals = self.get_vector_potential(x_arr, y_arr, s_vals, python=python)

        plt.figure(figsize=(10, 6))
        plt.plot(s_vals, Ax_vals, label='Ax')
        plt.plot(s_vals, Ay_vals, label='Ay')
        plt.plot(s_vals, As_vals, label='As')
        plt.xlabel('s [m]')
        plt.ylabel('Vector Potential [T·m]')
        plt.title(f'Vector Potential at (x={x_arr[0]}, y={y_arr[0]})')
        plt.legend()
        plt.grid()
        plt.show()

#
#     def set_integrator(self, n_slices=1000, n_steps = 1000):
#         if self.segments == []:
#             self.set_segments()
#
#         self.n_slices = n_slices
#         l_wig = self.field_fitter.length
#         s_start = self.field_fitter.s_full[0]
#         s_end   = self.field_fitter.s_full[-1]
#
#         s_cuts = np.linspace(s_start, s_end, n_slices + 1)
#         #s_mid = 0.5 * (s_cuts[:-1] + s_cuts[1:])
#         for ii in range(n_slices):
#             wig = xt.BorisSpatialIntegrator(fieldmap_callable=self.get_field, s_start=s_cuts[ii], s_end=s_cuts[ii + 1],
#                                             n_steps=np.round(n_steps / n_slices).astype(int),
#                                             verbose=True)
#             self.integrator.append(wig)
#
#     def get_line(self):
#         if self.integrator == []:
#             self.set_integrator()
#
#         self.env = xt.Environment()
#
#         for ii in range(self.n_slices):
#             self.env.elements[f'wigslice_{ii}'] = self.integrator[ii]
#         self.wiggler_line = self.env.new_line(components=['wigslice_' + str(ii) for ii in range(self.n_slices)])
#         return self.wiggler_line
#
#     def correctors(self, particle_ref):
#         start_time = time.time()
#         if self.wiggler_line is None:
#             self.get_line()
#
#         self.wiggler_line.particle_ref = particle_ref
#
#         self.env['k0l_corr1'] = 0.
#         self.env['k0l_corr2'] = 0.
#         self.env['k0l_corr3'] = 0.
#         self.env['k0l_corr4'] = 0.
#         self.env['k0sl_corr1'] = 0.
#         self.env['k0sl_corr2'] = 0.
#         self.env['k0sl_corr3'] = 0.
#         self.env['k0sl_corr4'] = 0.
#         self.env['on_wig_corr'] = 1.0
#
#         self.env.new('corr1', xt.Multipole, knl=['on_wig_corr * k0l_corr1'], ksl=['on_wig_corr * k0sl_corr1'])
#         self.env.new('corr2', xt.Multipole, knl=['on_wig_corr * k0l_corr2'], ksl=['on_wig_corr * k0sl_corr2'])
#         self.env.new('corr3', xt.Multipole, knl=['on_wig_corr * k0l_corr3'], ksl=['on_wig_corr * k0sl_corr3'])
#         self.env.new('corr4', xt.Multipole, knl=['on_wig_corr * k0l_corr4'], ksl=['on_wig_corr * k0sl_corr4'])
#
#         l_wig = self.field_fitter.length
#
#         self.wiggler_line.insert([
#             self.env.place('corr1', at=0.02),
#             self.env.place('corr2', at=0.1),
#             self.env.place('corr3', at=l_wig - 0.1),
#             self.env.place('corr4', at=l_wig - 0.02),
#         ], s_tol=5e-3
#         )
#
#         # To compute the kicks
#         opt = self.wiggler_line.match(
#             solve=False,
#             betx=0, bety=0,
#             only_orbit=True,
#             include_collective=True,
#             vary=xt.VaryList(['k0l_corr1', 'k0sl_corr1',
#                               'k0l_corr2', 'k0sl_corr2',
#                               'k0l_corr3', 'k0sl_corr3',
#                               'k0l_corr4', 'k0sl_corr4',
#                               ], step=1e-6),
#             targets=[
#                 xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
#                 xt.TargetSet(x=0., y=0, at='wigslice_167'),
#                 xt.TargetSet(x=0., y=0, at='wigslice_833')
#                 ],
#         )
#         opt.step(2)
#         end_time = time.time()
#         print(f"Wiggler correctors set in {end_time - start_time:.2f} seconds.")
#         print("Corrector strengths [T]:")
#         print(f"  k0l_corr1 = {self.env['k0l_corr1']:.6e}, k0sl_corr1 = {self.env['k0sl_corr1']:.6e}")
#         print(f"  k0l_corr2 = {self.env['k0l_corr2']:.6e}, k0sl_corr2 = {self.env['k0sl_corr2']:.6e}")
#         print(f"  k0l_corr3 = {self.env['k0l_corr3']:.6e}, k0sl_corr3 = {self.env['k0sl_corr3']:.6e}")
#         print(f"  k0l_corr4 = {self.env['k0l_corr4']:.6e}, k0sl_corr4 = {self.env['k0sl_corr4']:.6e}")