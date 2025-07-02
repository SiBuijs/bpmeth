import numpy.polynomial
import scipy.special

import bpmeth
import xtrack as xt
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from bpmeth import poly_fit

plt.close("all")

########################################################################################################################
# IMPORTING AND PREPARING THE DATA
########################################################################################################################
# Function to parse the data from a csv.
def parse_to_dataframe(file_path):
    """Parse directly into a multi-indexed DataFrame."""
    # Read the entire file into a DataFrame
    df = pd.read_csv(file_path, sep=r'\s+', header=None,
                     names=['X', 'Y', 'Z', 'Bx', 'By', 'Bz'])

    # Set multi-index
    df.set_index(['X', 'Y', 'Z'], inplace=True)

    return df

# Parse the data from knot_map_text.txt
file_path = 'example_data/knot_map_test.txt'
df = parse_to_dataframe(file_path)

# Select at which transverse coordinates (x, y) we want to evaluate the field (usually (0, 0)).
# The subsetz indicates that it takes the z-axis as the independent coordinate.
xy_point = (0, 0)
subsetz = df.xs(xy_point, level=['X', 'Y'])

# Extract the transverse fields and the longitudinal axis as numpy arrays.
# The B_z is on the order of 10⁻⁷, compared to B_x and B_y so can be neglected.
z_values = subsetz.index.to_numpy()
dz = 0.001
z_values = z_values * dz               # Convert to meters, more stable for the polynomials.
bx_values = subsetz['Bx'].to_numpy()
by_values = subsetz['By'].to_numpy()

bx_der    = np.gradient(bx_values, dz, edge_order=2)
by_der    = np.gradient(by_values, dz, edge_order=2)

#bz_values = subsetz['Bz'].to_numpy()
bt_values = np.sqrt(bx_values**2 + by_values**2)

########################################################################################################################
# DETERMINING BORDERS
########################################################################################################################

# Finds the peaks in B_x and B_y
bx_peaks    = find_peaks(bx_values)
by_peaks    = find_peaks(by_values)
bx_valleys  = find_peaks(-bx_values)
by_valleys  = find_peaks(-by_values)

# Reassign to only include our regions of interest.
# The find_peaks also picks up peaks in the flat regions, so this is to exclude those.
# The 99's are to actually include one peak which is exactly at 100.
bx_peaks   = bx_peaks[0][np.logical_and(bx_peaks[0] > 99, bx_peaks[0] < 2100)]
by_peaks   = by_peaks[0][np.logical_and(by_peaks[0] > 99, by_peaks[0] < 2100)]
bx_valleys = bx_valleys[0][np.logical_and(bx_valleys[0] > 99, bx_valleys[0] < 2100)]
by_valleys = by_valleys[0][np.logical_and(by_valleys[0] > 99, by_valleys[0] < 2100)]

# Splits the magnetic field into five regions. The regions are decided based on the peaks and valleys of B_x.
# The region between -1100 and xborderleft goes from the start until the first peak. We fit a series of polynomials.
# The region between xborderleft and xborderright encompasses the sinusoidal region in the middle. We fit a sinusoid.
# The region between xborderright adn 1100 goes from the last peak until the end. We fit a series of polynomials.
xborderleft  = bx_peaks[0]      # z = -974
xborderright = bx_peaks[-1]     # z =  970

# Splits the magnetic field into five regions. The regions are decided based on the peaks and valleys of B_y.
# The region between -1100 and yborderleft goes from the start until the first peak. We fit a series of polynomials.
# The region between yborderleft and yborderright encompasses the sinusoidal region in the middle. We fit a sinusoid.
# The region between yborderright adn 1100 goes from the last peak until the end. We fit a series of polynomials.
yborderleft  = by_valleys[0]    # z = -984
yborderright = by_valleys[-1]   # z =  961

# Assign the z-arrays for each region.
zx_regionleft  = z_values[:xborderleft].copy()
zx_regionsines = z_values[xborderleft:xborderright].copy()
zx_regionright = z_values[xborderright:].copy()

# Same for y.
zy_regionleft  = z_values[:yborderleft].copy()
zy_regionsines = z_values[yborderleft:yborderright].copy()
zy_regionright = z_values[yborderright:].copy()

# And the B_x arrays
bx_regionleft  = bx_values[:xborderleft].copy()
bx_regionsines = bx_values[xborderleft:xborderright].copy()
bx_regionright = bx_values[xborderright:].copy()

# And for B_y
by_regionleft  = by_values[:yborderleft].copy()
by_regionsines = by_values[yborderleft:yborderright].copy()
by_regionright = by_values[yborderright:].copy()

########################################################################################################################
# SINUSOID FITTING
########################################################################################################################

# Define the function to be fitted, a sinusoid superimposed on an enge function.
def sinusoid(x, *params):
    # Define the output array
    y = np.zeros_like(x, dtype=np.float64)

    # The range depends on the number of sinusoids there are present in the function.
    # B_x is best approximated with two, whereas B_y only needs one.
    # Note the integer division by 3, because each mode has three parameters.
    for A1, A2, k in zip(params[::3], params[1::3], params[2::3]):
        # General sinusoidal part is a linear combination of cosine and sine.
        # This is equivalent to a single function with a phase-offset, but more numerically stable.
        y += A1 * np.cos(k * x) + A2 * np.sin(k * x)

    return y

# These frequencies have been found before using a Fourier Transform.
# As stated before, B_x contains two modes and B_y only one.
# The factor of 2pi used to be in the sinusoid function, but this is a bit inconvenient, so I moved here.
x_freqs = [2*np.pi*19, 2*np.pi*37]#[0.019, 0.037]
y_freqs = [2*np.pi*28]#[0.028]

# Initial parameter guesses [A1.1, A1.2, freq1, A2.1, A2.2, freq2]
# These parameters were found from looking at the data for the amplitudes
# and a Fourier transform of the data for the frequencies.
x_initial_guess = np.array([0.29, 0.031, x_freqs[0], 0.083, -0.1, x_freqs[1]])
y_initial_guess = np.array([0.3, 0.8, y_freqs[0]])

# Fit the curve.
xpoptregsines, xpcovregsines = curve_fit(sinusoid, zx_regionsines, bx_regionsines, p0=x_initial_guess)
ypoptregsines, ypcovregsines = curve_fit(sinusoid, zy_regionsines, by_regionsines, p0=y_initial_guess)

# Calculates the output of the fit.
xfit_regsines = sinusoid(zx_regionsines, *xpoptregsines)
yfit_regsines = sinusoid(zy_regionsines, *ypoptregsines)


########################################################################################################################
# EDGE FITTING
########################################################################################################################

# This function fits a polynomial to data (b_region) of given degree over a region (z_region).
# It expects the boundary values and on which side the fit takes place (left=True for left, left=False for right).
# The latter is important to match the polynomial to the sinusoidal region, or the
def fit_polynomial_matching(degree, z_region, b_region, boundaries, left):
    """
    Fits a polynomial to any region that matches the middle sinusoidal region.

    Args:
        z_region (np.array): Array of z-coordinates for the region of interest.
        b_region (np.array): Array of B-values for the region of interest.
        sinparams (dict): Dictionary of boundary parameters.

    Returns:
        list: Polynomial coefficients for a polynomial of order N.
    """

    if degree >= 5:
        # Fit polynomial with boundary conditions
        if left:
            db_dz_left = (-3*b_region[0] + 4*b_region[1] - b_region[2]) / (2 * 0.001)
            d2b_dz2_left = (2 * b_region[0] - 5 * b_region[1] + 4 * b_region[2] - b_region[3]) / 0.001 ** 2
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[b_region[0], boundaries[0]],
                xp0=[z_region[0], z_region[-1]],
                yp0=[db_dz_left, boundaries[2]],
                xpp0=[z_region[0], z_region[-1]],
                ypp0=[d2b_dz2_left, boundaries[4]]
            )

        else:
            db_dz_right = (3*b_region[-1] - 4*b_region[-2] + b_region[-3]) / (2 * 0.001)
            d2b_dz2_right = (2*b_region[-1] - 5*b_region[-2] + 4*b_region[-3] - b_region[-4]) / 0.001**2
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[boundaries[1], b_region[-1]],
                xp0=[z_region[0], z_region[-1]],
                yp0=[boundaries[3], db_dz_right],
                xpp0=[z_region[0], z_region[-1]],
                ypp0=[boundaries[5], d2b_dz2_right]
            )
    elif degree >=3:
        # Fit polynomial with boundary conditions
        if left:
            db_dz_left = (-3*b_region[0] + 4*b_region[1] - b_region[2]) / (2 * 0.001)
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[b_region[0], boundaries[0]],
                xp0=[z_region[0], z_region[-1]],
                yp0=[db_dz_left, boundaries[2]]
            )

        else:
            db_dz_right = (3*b_region[-1] - 4*b_region[-2] + b_region[-3]) / (2 * 0.001)
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[boundaries[1], b_region[-1]],
                xp0=[z_region[0], z_region[-1]],
                yp0=[boundaries[3], db_dz_right]
            )

    elif degree >=1:
        # Fit polynomial with boundary conditions
        if left:
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[b_region[0], boundaries[0]]
            )

        else:
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[boundaries[1], b_region[-1]]
            )

    return bpol_reg

# This function gets the function value, its derivative and its second derivative at the end points.
# If sinusoid=True, the parameters in "object" are interpreted as the outcomes of the sinusoidal fit.
# Then an analytical expression for the sinusoid and its derivatives is used to find the boundary values.
# If sinusoid=False, the parameters in "object" are interpreted as the coefficients of a polynomial.
# The derivatives of the polynomial is computed analytically and evaluated at the boundaries.
def get_boundary_conditions(x, object, sinusoid):
    # Boundaries contains the values, derivatives and second derivatives at the boundary points.
    # Even entries contain the boundary values at the left side of the region in question.
    # Odd entries contain the boundary values at the right side of the region in question.
    boundaries = np.zeros(6)

    # The point that needs to match is actually one point to the left of the region in question.
    xleft  = x[0]  - 0.001
    xright = x[-1] + 0.001

    if sinusoid:
        # If the object are the coefficients of a sinusoid, then calculate the derivatives analytically.
        # The derivatives of the sinusoid are known and simple to evaluate.
        for i in range(len(object) // 3):
            A1 = object[3 * i]
            A2 = object[3 * i + 1]
            k  = object[3 * i + 2]

            boundaries[0] += A1 * np.cos(k*xleft)  + A2 * np.sin(k*xleft)
            boundaries[1] += A1 * np.cos(k*xright) + A2 * np.sin(k*xright)
            boundaries[2] += k * (A2 * np.cos(k*xleft)  - A1 * np.sin(k*xleft))
            boundaries[3] += k * (A2 * np.cos(k*xright) - A1 * np.sin(k*xright))
            boundaries[4] += -k * k * boundaries[0]
            boundaries[5] += -k * k * boundaries[1]
    else:
        # If object is a polynomial, then calculate the derivative using polyder.
        # The boundaries are evaluated by evaluating the polynomial and its derivatives at the boundaries.
        dxpoly = object.deriv()
        ddxpoly = object.deriv(2)

        boundaries[0] = object(xleft)
        boundaries[1] = object(xright)
        boundaries[2] = dxpoly(xleft)
        boundaries[3] = dxpoly(xright)
        boundaries[4] = ddxpoly(xleft)
        boundaries[5] = ddxpoly(xright)

    return boundaries

# This function splits an array in a number of regions specified by num_regions.
# It ensures that the sub-arrays have more-or less the same number of entries.
# It returns a slices object that can immediately be used to iterate over a region.
def get_balanced_slices(arr, num_regions):
    n = len(arr)
    base_size = n // num_regions  # Minimum elements per region
    remainder = n % num_regions  # Extra elements to distribute

    slices = []
    start = 0
    for i in range(num_regions):
        # Add 1 extra element to the first 'remainder' regions
        end = start + base_size + (1 if i < remainder else 0)
        slices.append(slice(start, end))  # Using `slice` for direct indexing
        start = end
    return slices

# This function expects z_region and b_region (the x- and y-values over which is to be fitted).
# It requires the number of slices, as it calls get_balanced_slices to slice z_region and b_region.
# It requires knowledge of if the fit is over the x-values or not and if it's on the left side or not.
# It calls get_boundary_conditions to get the boundary values, taking into account on which side these are to be found.
# It first fits a polynomial to the sinusoidal boundary condition. Then it continues to fit successive polynomials.
def fit_poly_regions(z_region, b_region, num_slices, left, x):
    slices = get_balanced_slices(z_region, num_slices)
    fit_reg = np.zeros_like(z_region)

    # If left, it needs to iterate in the reverse direction.
    if left:
        slices.reverse()

    # If x, it takes the values for the sinusoidal region of B_x.
    if x:
        z_sines = zx_regionsines
        poptsines = xpoptregsines

    # If !x, it takes the values for the sinusoidal region of B_y.
    else:
        z_sines = zy_regionsines
        poptsines = ypoptregsines

    # Iterates over the slices.
    for ii in slices:
        # Match a polynomial to the sinusoidal region first.
        if ii == slices[0]:
            # The x- and y-values of the current slice.
            z_this = z_region[ii]
            b_this = b_region[ii]

            # Find boundary values and fit.
            boundaries = get_boundary_conditions(z_sines, poptsines, sinusoid=True)
            fit = fit_polynomial_matching(degree, z_this, b_this, boundaries, left)
            poly = np.polynomial.Polynomial(fit)
            fit_reg[ii] = poly(z_this)

        # Then match a polynomial to the previous one.
        # Note that it uses "poly" and "z_this" from the previous iteration when calling "boundaries()".
        # Afterwards, the new "z_this" is declared and a new "poly" is found.
        # This is because the slices object cannot easily be indexed using [ii-1] or something like that.
        else:
            boundaries = get_boundary_conditions(z_this, poly, sinusoid=False)

            # The x- and y-values of the current slice.
            z_this = z_region[ii]
            b_this = b_region[ii]

            # Fitting procedure.
            fit = fit_polynomial_matching(degree, z_this, b_this, boundaries, left)
            poly = np.polynomial.Polynomial(fit)
            fit_reg[ii] = poly(z_this)

    return fit_reg

# The degree of the fitted polynomials.
degree=3

# The number slices.
# Also the number of separate polynomials fitted to a region.
num_slices = 15

# Get the data of the fit.
xfit_regleft  = fit_poly_regions(zx_regionleft,  bx_regionleft,  num_slices, left=True,  x=True)
xfit_regright = fit_poly_regions(zx_regionright, bx_regionright, num_slices, left=False, x=True)
yfit_regleft  = fit_poly_regions(zy_regionleft,  by_regionleft,  num_slices, left=True,  x=False)
yfit_regright = fit_poly_regions(zy_regionright, by_regionright, num_slices, left=False, x=False)

########################################################################################################################
# MERGE IT ALL TOGETHER
########################################################################################################################

# Concatenate the polynomial regions at the ends of the sinusoidal region.
bx_fit = np.concatenate((xfit_regleft, xfit_regsines, xfit_regright))
by_fit = np.concatenate((yfit_regleft, yfit_regsines, yfit_regright))

# Calculate the magnitude of the magnetic field.
bt_fit = np.sqrt(bx_fit**2 + by_fit**2)

########################################################################################################################
# PLOTS
########################################################################################################################

# Plot the data against the fit.
fig1, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(z_values, bx_values)
ax1.plot(z_values, bx_fit)
ax2.plot(z_values, by_values)
ax2.plot(z_values, by_fit)
ax3.plot(z_values, bt_values)
ax3.plot(z_values, bt_fit)

# Label the graphs.
ax1.set_title(f"Magnetic Field at (X, Y) = {xy_point}")
ax3.set_xlabel("Longitudinal Position, $s$, [m]")
ax1.set_ylabel("Horizontal Field, $B_x$, [T]")
ax2.set_ylabel("Vertical Field, $B_y$, [T]")
ax3.set_ylabel("Magnitude, $|B|$, [T]")

# Make a legend.
ax1.legend(["$B_x$ Data", "$B_x$ Fit"], loc="upper right")
ax2.legend(["$B_y$ Data", "$B_y$ Fit"], loc="upper right")
ax3.legend(["$|B|$ Data", "$|B|$ Fit"], loc="upper right")

# Turn on the grids.
ax1.grid()
ax2.grid()
ax3.grid()
plt.show()