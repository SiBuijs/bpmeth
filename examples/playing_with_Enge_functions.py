import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import pandas as pd
from scipy.signal import find_peaks

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

x = zy_regionleft
y = by_regionleft

x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

x = 2 * (x - x_min) / (x_max - x_min) - 1  # scale to [-1,1]

if y_min <= 0:
    y -= y_min - 0.1

degree = 15  # degree of polynomial in exponent. 15 seems good fot one peak and one valley.

# --- Helper functions ---
def fit_polynomial(x, y, A, deg=2):
    """
    Given data (x,y) and a candidate A, transform y -> t, fit polynomial.
    Returns Polynomial object (coefficients in ascending order).
    """
    if np.any(y <= 0) or np.any(y >= A):
        return None  # invalid for log transform

    t = np.log(A / y - 1)  # transformed targets
    coeffs = np.polyfit(x, t, deg)  # monomial basis
    return np.poly1d(coeffs)


def enge_predict(x, poly, A):
    """Evaluate Enge function with fitted polynomial poly and scale A."""
    return A / (1 + np.exp(poly(x)))


def error_for_A(A, x, y, deg=2):
    """Compute squared error for a candidate A."""
    poly = fit_polynomial(x, y, A, deg=deg)
    if poly is None:
        return np.inf
    y_hat = enge_predict(x, poly, A)
    return np.sum((y - y_hat) ** 2)


# --- Step 2: manual scan over A ---

A_candidates = np.linspace(max(y) * 1.01, 10, 50)
errors = [error_for_A(Ac, x, y, deg=degree) for Ac in A_candidates]

best_A_scan = A_candidates[np.argmin(errors)]
print("Best A from scan:", best_A_scan)

# --- Step 3: automatic optimization (bracket) ---
result = minimize_scalar(error_for_A, bounds=(max(y) * 1.01, 10),
                         args=(x, y, degree), method="bounded")
best_A_opt = result.x
print("Best A from optimizer:", best_A_opt)

# --- Step 4: fit polynomial at best A ---
poly_fit = fit_polynomial(x, y, best_A_opt, deg=degree)
y_fit = enge_predict(x, poly_fit, best_A_opt)

# --- Plot ---
plt.scatter(x, y, label="data", color="black")
plt.plot(x, y_fit, label="fit", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
