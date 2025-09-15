import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import pandas as pd
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial
from sympy import preorder_traversal


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
xborderleft  = bx_peaks[0]      # z = -938
xborderright = bx_peaks[-1]     # z =  952

# Splits the magnetic field into five regions. The regions are decided based on the peaks and valleys of B_y.
# The region between -1100 and yborderleft goes from the start until the first peak. We fit a series of polynomials.
# The region between yborderleft and yborderright encompasses the sinusoidal region in the middle. We fit a sinusoid.
# The region between yborderright adn 1100 goes from the last peak until the end. We fit a series of polynomials.
yborderleft  = by_valleys[0]    # z = -929
yborderright = by_valleys[-1]   # z =  871

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
# FIT ENGES
########################################################################################################################

x = zy_regionright
y = by_regionright

x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

print(f"y_min = {y_min}, y_max = {y_max}")

if y_min <= 0:
    y_rescaled = y - y_min + 0.1  # Shift to positive values for fitting.


# Convert to u in [-1, 1]
c_u0 = -(x_min + x_max) / (x_max - x_min)
c_u1 = 2 / (x_max - x_min)
poly_u = Polynomial([c_u0, c_u1])
u = poly_u(x)

degree = 15  # degree of polynomial in exponent. 15 seems good fot one peak and one valley.

# --- Helper functions ---
def fit_polynomial(x, y, A, deg=2):
    t = np.log(A / y - 1)  # transformed targets
    coeffs = Polynomial.fit(x, t, deg)  # monomial basis
    return coeffs.convert(kind=Polynomial)  # convert to standard basis


def enge_eval(x, poly, A):
    """Evaluate Enge function with fitted polynomial poly and scale A."""
    return A * sp.special.expit(-poly(x))


def error_for_A(A, x, y, deg=2):
    """Compute squared error for a candidate A."""
    poly = fit_polynomial(x, y, A, deg=deg)
    y_hat = enge_eval(x, poly, A)
    return np.sum((y - y_hat) ** 2)

# --- Step 2: manual scan over A ---
A_candidates = np.linspace(max(y_rescaled) * 1.01, 2, 50)
errors = [error_for_A(Ac, u, y_rescaled, deg=degree) for Ac in A_candidates]
best_A_scan = A_candidates[np.argmin(errors)]
print("Best A from scan:", best_A_scan)

"""
# --- Step 3: automatic optimization (bracket) ---
result = minimize_scalar(error_for_A, bounds=(max(y_rescaled) * 1.01, 5),
                         args=(u, y_rescaled, degree), method="bounded")
best_A_opt = result.x
print("Best A from optimizer:", best_A_opt)
"""

# --- Step 4: fit polynomial at best A ---

# Rescale back to original x
poly_fit = fit_polynomial(u, y_rescaled, best_A_scan, deg=degree)
y_fit = enge_eval(u, poly_fit, best_A_scan)

# Rescale back to original x
poly_x = poly_fit.convert(domain=[x_min, x_max])
y_fit_rescaled = enge_eval(u, poly_x, best_A_scan)

print("Poly_x coefficients:")
print(poly_x.coef)

print("Poly_u coefficients:")
print(poly_fit.coef)

if y_min <= 0:
    y_fit += y_min - 0.1
    y_fit_rescaled += y_min - 0.1

# --- Plot ---
plt.scatter(x, y, label="data", color="black")
plt.plot(x, y_fit_rescaled, label="fit", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
