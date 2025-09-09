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

# Convert from millimeters to meters, a bit more stable for polynomial fitting.
dz = 0.001
z_values = z_values * dz               # Convert to meters, more stable for the polynomials.
bx_values = subsetz['Bx'].to_numpy()
by_values = subsetz['By'].to_numpy()
bz_values = subsetz['Bz'].to_numpy()

#bz_values = subsetz['Bz'].to_numpy()
bt_values = np.sqrt(bx_values**2 + by_values**2)

########################################################################################################################
# DETERMINING BORDERS
########################################################################################################################

# Finds the peaks in B_x and B_y.
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
xborderleft  = bx_valleys[1]      # z = -938
xborderright = bx_valleys[-2]     # z =  952

# Splits the magnetic field into five regions. The regions are decided based on the peaks and valleys of B_y.
# The region between -1100 and yborderleft goes from the start until the first peak. We fit a series of polynomials.
# The region between yborderleft and yborderright encompasses the sinusoidal region in the middle. We fit a sinusoid.
# The region between yborderright adn 1100 goes from the last peak until the end. We fit a series of polynomials.
yborderleft  = by_peaks[2]    # z = -929
yborderright = by_peaks[-3]   # z =  871

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

# And for B_y.
by_regionleft  = by_values[:yborderleft].copy()
by_regionsines = by_values[yborderleft:yborderright].copy()
by_regionright = by_values[yborderright:].copy()


########################################################################################################################
# SINUSOID FITTING
########################################################################################################################

# Define the function to be fitted, a sinusoid.
# If multiple sinusoids are present, this is reflected in the length of "params".
# All modes present will be iterated over.
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

# This function finds the amplitudes and frequencies of the (co)sines present in the data.
def find_frequencies(x, y, dx, threshold):
    # Calculate the Fourier Transform of the data and find the frequency spectrum.
    fft = sc.fft.fft(y)
    fftfreq = sc.fft.fftfreq(len(x), dx)

    # Only keep the positive part.
    fft = fft[fftfreq > 0]
    fftfreq = fftfreq[fftfreq > 0]

    # Find the peaks in the Fourier Transform applying the threshold.
    fftpeaks = find_peaks(np.abs(fft), threshold=threshold)

    # The corresponding amplitudes and k-values are extracted from the location of the peaks.
    amplitudes = fft[fftpeaks[0]]
    k_values = fftfreq[fftpeaks[0]]

    return amplitudes, k_values

# Get the amplitudes and frequencies of the (co)sines present in the data.
# Played a bit with the threshold. Threshold=1 gives four sinusoids for B_x and three for B_y.
# Threshold=2 gives three sinusoids for B_x and one for B_y and still a good fit.
bx_fft_amps, bx_fft_k = find_frequencies(zx_regionsines, bx_regionsines, dz, threshold=2)
by_fft_amps, by_fft_k = find_frequencies(zy_regionsines, by_regionsines, dz, threshold=2)

# The amplitudes of the (co)sine terms are related to the Fourier Transform of the data.
xcos_amp_guesses =  2 * dz * bx_fft_amps.real
xsin_amp_guesses = -2 * dz * bx_fft_amps.imag
ycos_amp_guesses =  2 * dz * by_fft_amps.real
ysin_amp_guesses = -2 * dz * by_fft_amps.imag

# An empty array. It will be arranged as follows:
# [A1.1, A1.2, freq1, A2.1, A2.2, freq2, ...]
# So the amplitude of the cosines are the 0, 3, 6, ... entries,
# The amplitude of the sines are the 1, 4, 7, ... entries
# The frequencies are the 2, 5, 8, ... entries.
xparams = np.zeros(len(bx_fft_k)*3)

for ii in range(len(bx_fft_k)):
    xparams[3*ii]     = xcos_amp_guesses[ii]
    xparams[3*ii + 1] = xsin_amp_guesses[ii]
    xparams[3*ii + 2] = 2 * np.pi * bx_fft_k[ii]

# Defined the same way as above, but for B_y.
yparams = np.zeros(len(by_fft_k)*3)

for ii in range(len(by_fft_k)):
    yparams[3*ii]     = ycos_amp_guesses[ii]
    yparams[3*ii + 1] = ysin_amp_guesses[ii]
    yparams[3*ii + 2] = 2 * np.pi * by_fft_k[ii]

# Fit the curve.
xpoptregsines, xpcovregsines = curve_fit(sinusoid, zx_regionsines, bx_regionsines, p0=xparams)
ypoptregsines, ypcovregsines = curve_fit(sinusoid, zy_regionsines, by_regionsines, p0=yparams)

# Calculates the output of the fit.
bxfit_regsines = sinusoid(zx_regionsines, *xpoptregsines)
byfit_regsines = sinusoid(zy_regionsines, *ypoptregsines)

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
    polys = []

    # If left, it needs to iterate in the reverse direction.
    if left:
        slices.reverse()

    # Iterates over the slices.
    for ii in slices:
        # Match a polynomial to the sinusoidal region first.
        if ii == slices[0]:
            # If x, it takes the values for the sinusoidal region of B_x.
            if x:
                z_sines = zx_regionsines
                poptsines = xpoptregsines

                # If !x, it takes the values for the sinusoidal region of B_y.
            else:
                z_sines = zy_regionsines
                poptsines = ypoptregsines

            # The x- and y-values of the current slice.
            z_this = z_region[ii]
            b_this = b_region[ii]

            # Find boundary values and fit.
            boundaries = get_boundary_conditions(z_sines, poptsines, sinusoid=True)
            fit = fit_polynomial_matching(degree, z_this, b_this, boundaries, left)
            poly = np.polynomial.Polynomial(fit)
            polys.append(poly)
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
            polys.append(poly)
            fit_reg[ii] = poly(z_this)

    return fit_reg, polys

# The degree of the fitted polynomials.
degree=3

xleftslices  = 23 # Good: 23
xrightslices = 21 # Good: 21
yleftslices  = 20 # Good: 20
yrightslices = 30 # Good: 30

# Get the data of the fit.
bxfit_regleft, bxfit_polysleft   = fit_poly_regions(zx_regionleft, bx_regionleft, xleftslices, left=True, x=True)
bxfit_regright, bxfit_polysright = fit_poly_regions(zx_regionright, bx_regionright, xrightslices, left=False, x=True)
byfit_regleft, byfit_polysleft   = fit_poly_regions(zy_regionleft, by_regionleft, yleftslices, left=True, x=False)
byfit_regright, byfit_polysright = fit_poly_regions(zy_regionright, by_regionright, yrightslices, left=False, x=False)

# The following is not used, but I will keep it here in case we want to use it later anyway.
'''
########################################################################################################################
# FIT EVERYTHING TO POLYNOMIALS
########################################################################################################################
# This function expects z_region and b_region (the x- and y-values over which is to be fitted).
# It requires the number of slices, as it calls get_balanced_slices to slice z_region and b_region.
# It requires knowledge of if the fit is over the x-values or not and if it's on the left side or not.
# It calls get_boundary_conditions to get the boundary values, taking into account on which side these are to be found.
# It first fits a polynomial to the sinusoidal boundary condition. Then it continues to fit successive polynomials.
def modified_fit_poly_regions(z_region, b_region, num_slices, left):
    slices = get_balanced_slices(z_region, num_slices)
    fit_reg = np.zeros_like(z_region)

    # If left, it needs to iterate in the reverse direction.
    if left:
        slices.reverse()

    # Iterates over the slices.
    for ii in slices:
        # Match a polynomial to the sinusoidal region first.
        if ii == slices[0]:
            # The x- and y-values of the current slice.
            z_this = z_region[ii]
            b_this = b_region[ii]

            # Find boundary values and fit.
            # If we fit everything with polynomials, all left boundaries are zero.
            boundaries = [0, 0, 0, 0, 0, 0]
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

degree = 3

# This is just to try what effect it has if I choose the total number of slices for the entire region.
# Then I redistribute the slices over the three regions according to their ratios.
xnum_slices = 300
ynum_slices = 160

bx_polyfit = modified_fit_poly_regions(z_values, bx_values, xnum_slices, left=True)
by_polyfit = modified_fit_poly_regions(z_values, by_values, ynum_slices, left=True)
bt_polyfit = np.sqrt(bx_polyfit**2 + by_polyfit**2)
'''

########################################################################################################################
# DEFINE FIELD AS FUNCTIONS
########################################################################################################################

def b_field_func(z, *params):
    # params = [borders, left_poly, right_poly, sinusoids]
    borders = params[0]
    left_poly = params[1]
    right_poly = params[2]
    sinusoids = params[3]

########################################################################################################################
# MERGE IT ALL TOGETHER
########################################################################################################################

# Concatenate the polynomial regions at the ends of the sinusoidal region.
bx_fit = np.concatenate((bxfit_regleft, bxfit_regsines, bxfit_regright))
by_fit = np.concatenate((byfit_regleft, byfit_regsines, byfit_regright))

# Calculate the magnitude of the magnetic field.
bt_fit = np.sqrt(bx_fit**2 + by_fit**2)

# Calculates the primitive function of bx_fit and by_fit and the data.
bx_trapz = sc.integrate.cumulative_trapezoid(bx_values,  dx=dz)
by_trapz = sc.integrate.cumulative_trapezoid(by_values,  dx=dz)
bx_fit_trapz = sc.integrate.cumulative_trapezoid(bx_fit, dx=dz)
by_fit_trapz = sc.integrate.cumulative_trapezoid(by_fit, dx=dz)


########################################################################################################################
# PLOTS
########################################################################################################################

# Functions
#########################################################################
# Plot the data against the fit.
fig1, (ax1, ax2) = plt.subplots(2, figsize=(10, 4), constrained_layout=True)
ax1.plot(z_values, bx_values)
ax1.plot(zx_regionleft, bx_regionleft)
ax1.plot(zx_regionsines, bx_regionsines)
ax1.plot(zx_regionright, bx_regionright)
ax2.plot(z_values, by_values)
ax2.plot(zy_regionleft, by_regionleft)
ax2.plot(zy_regionsines, by_regionsines)
ax2.plot(zy_regionright, by_regionright)
#ax3.plot(z_values, bt_values)
#ax3.plot(z_values, bt_fit)

# Add vertical lines at different positions for each subplot
ax1.axvline(x=z_values[xborderleft],  color='k', linestyle='--', linewidth=1)
ax1.axvline(x=z_values[xborderright], color='k', linestyle='--', linewidth=1)
ax2.axvline(x=z_values[yborderleft],  color='k', linestyle='--', linewidth=1)
ax2.axvline(x=z_values[yborderright], color='k', linestyle='--', linewidth=1)

# Label the graphs.
ax1.set_title(f"Magnetic Field at (X, Y) = {xy_point}")
ax2.set_xlabel("Longitudinal Position, $s$, [m]")
ax1.set_ylabel("Horizontal Field, $B_x$, [T]")
ax2.set_ylabel("Vertical Field, $B_y$, [T]")
#ax3.set_ylabel("Magnitude, $|B|$, [T]")

# Make a legend.
ax1.legend(["$B_x$ Data", "$B_x$ Left Fit", "$B_x$ Sine Fit", "$B_x$ Right Fit"], loc="lower right")
ax2.legend(["$B_y$ Data", "$B_y$ Left Fit", "$B_y$ Sine Fit", "$B_y$ Right Fit"], loc="lower right")
#ax3.legend(["$|B|$ Data", "$|B|$ Section Fit"], loc="upper right")

# Turn on the grids.
ax1.grid()
ax2.grid()
#ax3.grid()

# Primitives
#########################################################################
# Plot the data against the fit.
fig2, (ax4, ax5) = plt.subplots(2)
ax4.plot(z_values[:-1], bx_trapz)
ax4.plot(z_values[:-1], bx_fit_trapz)
ax5.plot(z_values[:-1], by_trapz)
ax5.plot(z_values[:-1], by_fit_trapz)

# Add vertical lines at different positions for each subplot
ax4.axvline(x=z_values[xborderleft],  color='k', linestyle='--', linewidth=1)
ax4.axvline(x=z_values[xborderright], color='k', linestyle='--', linewidth=1)
ax5.axvline(x=z_values[yborderleft],  color='k', linestyle='--', linewidth=1)
ax5.axvline(x=z_values[yborderright], color='k', linestyle='--', linewidth=1)

# Label the graphs.
ax4.set_title(f"Integral of Magnetic Field at (X, Y) = {xy_point}")
ax5.set_xlabel("Longitudinal Position, $s$, [m]")
ax4.set_ylabel("Horizontal Field, $∫ B_x ds$, [Tm]")
ax5.set_ylabel("Vertical Field, $∫ B_y ds$, [Tm]")

# Make a legend.
ax4.legend(["$∫ B_x ds$ Data", "$∫ B_x ds$ Region Fit"], loc="lower right")
ax5.legend(["$∫ B_y ds$ Data", "$∫ B_y ds$ Region Fit"], loc="upper right")

# Turn on the grids.
ax4.grid()
ax5.grid()
plt.show()