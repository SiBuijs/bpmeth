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
dbx_peaks   = find_peaks(bx_der)
dby_peaks   = find_peaks(by_der)
dbx_valleys = find_peaks(-bx_der)
dby_valleys = find_peaks(-by_der)

# Reassign to only include our regions of interest.
# The find_peaks also picks up peaks in the flat regions, so this is to exclude those.
# The 99's are to actually include one peak which is exactly at 100.
bx_peaks   = bx_peaks[0][np.logical_and(bx_peaks[0] > 99, bx_peaks[0] < 2100)]
by_peaks   = by_peaks[0][np.logical_and(by_peaks[0] > 99, by_peaks[0] < 2100)]
bx_valleys = bx_valleys[0][np.logical_and(bx_valleys[0] > 99, bx_valleys[0] < 2100)]
by_valleys = by_valleys[0][np.logical_and(by_valleys[0] > 99, by_valleys[0] < 2100)]
dbx_peaks   = dbx_peaks[0][np.logical_and(dbx_peaks[0] > 95, dbx_peaks[0] < 2100)]
dby_peaks   = dby_peaks[0][np.logical_and(dby_peaks[0] > 95, dby_peaks[0] < 2100)]
dbx_valleys = dbx_valleys[0][np.logical_and(dbx_valleys[0] > 95, dbx_valleys[0] < 2100)]
dby_valleys = dby_valleys[0][np.logical_and(dby_valleys[0] > 95, dby_valleys[0] < 2100)]

# Splits the magnetic field into five regions. The regions are decided based on the peaks and valleys of B_x.
# The region between -1100 and xborder1 goes up to the point where the first derivative is zero.
# The region between xborder1 and xborder2 goes from where the first derivative is zero to the first valley.
# The region between xborder2 and xborder3 goes from the first valley to the first peak.
# The region between xborder3 and xborder4 is the sinusoidal area.
# The region between xborder4 and xborder5 goes from the last peak to the last valley and will be fitted with a polynomial.
# The region between xborder5 and xborder6 goes from the last valley to where the first derivative is zero.
# The region between xborder6 and 1100 goes from point where the first derivative is zero to the end and will be fitted with a polynomial.
#xborder1 = dbx_valleys[0]   # z = -998
#xborder2 = bx_valleys[0]    # z = -992
xborderleft  = bx_peaks[0]      # z = -974
xborderright = bx_peaks[-1]     # z =  970
#xborder5 = bx_valleys[-1]   # z =  990
#xborder6 = dbx_peaks[-1]    # z =  997

# Splits the magnetic field into five regions. The regions are decided based on the peaks and valleys of B_y.
# The region between -1100 and yborder1 goes up to the point where the first derivative is zero.
# The region between yborder1 and yborder2 goes from where the first derivative is zero to the first peak.
# The region between yborder2 and yborder3 goes from the first peak to the first valley.
# The region between yborder3 and yborder4 is the sinusoidal area.
# The region between yborder4 and yborder5 goes from the last peak to the last valley and will be fitted with a polynomial.
# The region between yborder5 and yborder6 goes from the last valley to where the first derivative is zero.
# The region between yborder6 and 1100 goes from point where the first derivative is zero to the end and will be fitted with a polynomial.
#yborder1 = dby_peaks[0]     # z = -1004
#yborder2 = by_peaks[0]      # z = -1000
yborderleft  = by_valleys[0]    # z = -984
yborderright = by_valleys[-1]   # z =  961
#yborder5 = by_peaks[-1]     # z =  978
#yborder6 = dby_valleys[-1]  # z =  983

# Assign the z-arrays for each region.
zx_regionleft  = z_values[:xborderleft].copy()
zx_regionsines = z_values[xborderleft:xborderright].copy()
zx_regionright = z_values[xborderright:].copy()
#zx_region4 = z_values[xborderleft:xborderright].copy()
#zx_region5 = z_values[xborderright:xborder5].copy()
#zx_region6 = z_values[xborder5:xborder6].copy()
#zx_region7 = z_values[xborder6:].copy()

# Same for y.
zy_regionleft  = z_values[:yborderleft].copy()
zy_regionsines = z_values[yborderleft:yborderright].copy()
zy_regionright = z_values[yborderright:].copy()
#zy_region4 = z_values[yborderleft:yborderright].copy()
#zy_region5 = z_values[yborderright:yborder5].copy()
#zy_region6 = z_values[yborder5:yborder6].copy()
#zy_region7 = z_values[yborder6:].copy()

# And the B_x arrays
bx_regionleft  = bx_values[:xborderleft].copy()
bx_regionsines = bx_values[xborderleft:xborderright].copy()
bx_regionright = bx_values[xborderright].copy()
#bx_region4 = bx_values[xborderleft:xborderright].copy()
#bx_region5 = bx_values[xborderright:xborder5].copy()
#bx_region6 = bx_values[xborder5:xborder6].copy()
#bx_region7 = bx_values[xborder6:].copy()

# And for B_y
by_regionleft  = by_values[:yborderleft].copy()
by_regionsines = by_values[yborderleft:yborderright].copy()
by_regionright = by_values[yborderright:].copy()
#by_region4 = by_values[yborderleft:yborderright].copy()
#by_region5 = by_values[yborderright:yborder5].copy()
#by_region6 = by_values[yborder5:yborder6].copy()
#by_region7 = by_values[yborder6:].copy()

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
x_initial_guess = np.array([0.29, 0.031, x_freqs[0], 0.083, -0.1, x_freqs[1]])
y_initial_guess = np.array([0.3, 0.8, y_freqs[0]])

# Fit the curve.
xpoptregsines, xpcovregsines = curve_fit(sinusoid, zx_regionsines, bx_regionsines, p0=x_initial_guess)
ypoptregsines, ypcovregsines = curve_fit(sinusoid, zy_regionsines, by_regionsines, p0=y_initial_guess)

print(xpoptregsines)
print(ypoptregsines)

# Calculates the output of the fit.
xfit_regsines = sinusoid(zx_regionsines, *xpoptregsines)
yfit_regsines = sinusoid(zy_regionsines, *ypoptregsines)

# Find the pole length from k_y, the wavenumber of B_y
# The frequency already has a 2pi factored out, so wavelength = 1/freq.
# The pole length is 1/4 of the wavelength.
# This pole length includes any spacing between the poles.
# Returns 9 mm.
#polelength = 1 / 4 / ypoptreg4[2]
#print(f"Pole length [mm] = {polelength}")

########################################################################################################################
# EDGE FITTING
########################################################################################################################


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

    # Derivatives of region
    # Calculate the left/right derivatives with forward/backward differentiation.
    #db_dz_left = (-3*b_region[0] + 4*b_region[1] - b_region[2]) / (2 * 0.001)
    #db_dz_right = (3*b_region[-1] - 4*b_region[-2] + b_region[-3]) / (2 * 0.001)
    #d2b_dz2_left = (2*b_region[0] - 5*b_region[1] + 4*b_region[2] - b_region[3]) / 0.001**2
    #d2b_dz2_right = (2*b_region[-1] - 5*b_region[-2] + 4*b_region[-3] - b_region[-4]) / 0.001**2

    #print("----------")
    #print("db_dz_region[0]    = ", db_dz_left)
    #print("d2b_dz2_region[0]  = ", db_dz_right)
    #print("db_dz_region[-1]   = ", d2b_dz2_left)
    #print("d2b_dz2_region[-1] = ", d2b_dz2_right)

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

xsinesbounds = get_boundary_conditions(zx_regionsines, xpoptregsines, sinusoid=True)
ysinesbounds = get_boundary_conditions(zy_regionsines, ypoptregsines, sinusoid=True)

# Print the boundary values of region 4.
print("Region 4 x boundary values: ", xsinesbounds)
print("Region 4 y boundary values: ", ysinesbounds)

degree=3

num_polys = 10

xslicesleft  = get_balanced_slices(zx_regionleft,  num_polys)
xslicesright = get_balanced_slices(zx_regionright, num_polys)
yslicesleft  = get_balanced_slices(zy_regionleft,  num_polys)
yslicesright = get_balanced_slices(zy_regionright, num_polys)

# For left, we iterate backwards, because we want to start with the sinusoids.
# Need to find a way to effectively deal with the parameters of the polynomials.
for ii in xslicesleft[::-1]:
    xpopt, xpcov = [], []
    z_region = zx_regionleft[ii]
    bx_region = bx_regionleft[ii]
    # Treat the sinusoidal case first.
    if ii == xslicesleft[-1]:
        boundaries = get_boundary_conditions(zx_regionsines, xpoptregsines, sinusoid=True)
        fit, cov = fit_polynomial_matching(degree, z_region, bx_region, boundaries, left=True)
        xpopt.append(fit)
        xpcov.append(cov)

    else:
        boundaries = get_boundary_conditions(zx_regionleft[ii-1], xpoptregsines[ii-1], sinusoid=False)
        fit, cov = fit_polynomial_matching(degree, z_region, bx_region, boundaries, left=True)
        xpopt.append(fit)
        xpcov.append(cov)

'''
########################################################################################################################
# MERGE IT ALL TOGETHER
########################################################################################################################

bx_fit = np.concatenate((xfit_regleft, xfit_regsines, xfitregright))
by_fit = np.concatenate((yfit_regleft, yfit_regsines, yfitregright))
bt_fit = np.sqrt(bx_fit**2 + by_fit**2)

print(sc.integrate.cumulative_trapezoid(bx_fit, dx=dz))
print(sc.integrate.cumulative_trapezoid(by_fit, dx=dz))

########################################################################################################################
# PLOTS
########################################################################################################################
fig1, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(z_values, bx_values)
ax1.plot(z_values, bx_fit)
#ax1.plot(zx_region1, xfit_reg1)
#ax1.plot(zx_region2, xfit_reg2)
#ax1.plot(zx_region3, xfit_reg3)
#ax1.plot(zx_region4, xfit_reg4)
#ax1.plot(zx_region5, xfit_reg5)
ax2.plot(z_values, by_values)
ax2.plot(z_values, by_fit)
#ax2.plot(zy_region1, yfit_reg1)
#ax2.plot(zy_region2, yfit_reg2)
#ax2.plot(zy_region3, yfit_reg3)
#ax2.plot(zy_region4, yfit_reg4)
#ax2.plot(zy_region5, yfit_reg5)
ax3.plot(z_values, bt_values)
ax3.plot(z_values, bt_fit)

ax1.set_title(f"Magnetic Field at (X, Y) = {xy_point}")
ax3.set_xlabel("Longitudinal Position, $s$, [m]")
ax1.set_ylabel("Horizontal Field, $B_x$, [T]")
ax2.set_ylabel("Vertical Field, $B_y$, [T]")
ax3.set_ylabel("Magnitude, $|B|$, [T]")

ax1.legend(["$B_x$ Data", "$B_x$ Fit"], loc="upper right")
ax2.legend(["$B_y$ Data", "$B_y$ Fit"], loc="upper right")
ax3.legend(["$|B|$ Data", "$|B|$ Fit"], loc="upper right")

ax1.grid()
ax2.grid()
ax3.grid()
plt.show()
'''