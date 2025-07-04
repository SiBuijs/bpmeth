import numpy as np
import pandas as pd
import scipy as sc
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from bpmeth import poly_fit

class Wiggler:
    def __init__(self, file_path, xy_point=(0, 0), dz=0.001):
        self.file_path = file_path
        self.xy_point = xy_point
        self.dz = dz
        self.df = None
        self.z_full = None
        self.field_data = {
            'Bx': {
                'data': None,               # Data points from the field map for B_x.
                'fit': None,                # Data points from the fit for B_x.
                'sine_fit_params': None,    # Fit parameters for the sinusoidal region.
                'poly_fit_params': {        # Fit parameters for the polynomial regions (edges).
                    'left': [],
                    'right': []
                },
                'borders': []               # Indices at which the sinusoidal region starts and ends.
            },
            'By': {
                'data': None,               # Data points from the field map for B_y.
                'fit': None,                # Data points from the fit for B_y.
                'sine_fit_params': None,    # Fit parameters for the sinusoidal region.
                'poly_fit_params': {        # Fit parameters for the polynomial regions (edges).
                    'left': [],
                    'right': []
                },
                'borders': []               # Indices at which the sinusoidal region starts and ends.
            }
        }

    def set_wiggler(self, print=False, border_indices=None, border_threshold=0.01, sine_threshold=2):
        self.parse_to_dataframe()
        self.load_data()
        self.find_regions(border_indices=border_indices, print=print)
        self.fit_sinusoidal_region(threshold=sine_threshold)

    # CHECK
    # A function that consists of a sum of sinusoidal functions.
    # The functions have the form: A1 * cos(k * x) + A2 * sin(k * x)
    def sinusoid(self, x, *params):
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

    # CHECK
    # This function finds the amplitudes and frequencies of the (co)sines present in the data.
    def find_frequencies(self, x, y, threshold=2):
        # Calculate the Fourier Transform of the data and find the frequency spectrum.
        fft = sc.fft.fft(y)
        fftfreq = sc.fft.fftfreq(len(x), self.dz)

        # Only keep the nonnegative part.
        fft = fft[fftfreq >= 0]
        fftfreq = fftfreq[fftfreq >= 0]

        # Find the peaks in the Fourier Transform applying the threshold.
        fftpeaks = find_peaks(np.abs(fft), threshold=threshold)

        # The corresponding amplitudes and k-values are extracted from the location of the peaks.
        amplitudes = fft[fftpeaks[0]]
        k_values = fftfreq[fftpeaks[0]]

        return amplitudes, k_values

    # CHECK
    def parse_to_dataframe(self):
        """Parse directly into a multi-indexed DataFrame."""
        # Read the entire file into a DataFrame
        df = pd.read_csv(self.file_path, sep=r'\s+', header=None,
                         names=['X', 'Y', 'Z', 'Bx', 'By', 'Bz'])

        # Set multi-index
        df.set_index(['X', 'Y', 'Z'], inplace=True)

        self.df = df

        return

    # CHECK
    def load_data(self):
        # The subsetz indicates that it takes the z-axis as the independent coordinate.
        subsetz = self.df.xs(self.xy_point, level=['X', 'Y'])

        # Extract the transverse fields and the longitudinal axis as numpy arrays.
        # The B_z is on the order of 10⁻⁷, compared to B_x and B_y so will be neglected.
        self.z_full = subsetz.index.to_numpy() * self.dz

        self.field_data['Bx']['data'] = subsetz['Bx'].to_numpy()
        self.field_data['By']['data'] = subsetz['By'].to_numpy()

        # Already initialize the fit arrays to zeros.
        self.field_data['Bx']['fit'] = np.zeros_like(self.field_data['Bx']['data'])
        self.field_data['By']['fit'] = np.zeros_like(self.field_data['By']['data'])

        return

    # CHECK
    def find_regions(self, threshold=0.01, print=False, border_indices=None):
        """Identify field regions using peak detection"""
        bx_peaks = find_peaks(self.field_data['Bx']['data'], threshold)
        by_peaks = find_peaks(self.field_data['By']['data'], threshold)
        bx_valleys = find_peaks(-self.field_data['Bx']['data'], threshold)
        by_valleys = find_peaks(-self.field_data['By']['data'], threshold)

        if print:
            print("B_x peaks:   ", bx_peaks[0])
            print("B_y peaks:   ", by_peaks[0])
            print("B_x valleys: ", bx_valleys[0])
            print("B_y valleys: ", by_valleys[0])

        # Splits the magnetic field into five regions. The regions are decided based on the peaks and valleys of B_x.
        # Usually, the region between the second valley and the second-last valley is close to sinusoidal,
        # so that is taken as the default. Using the print statements, other borders can be chosen through
        # border_indices. This is just about as modular as I can make it.
        if border_indices is None:
            self.field_data['Bx']['borders'] = [bx_valleys[0][1], bx_valleys[0][-2]]
            self.field_data['By']['borders'] = [by_peaks[0][1], by_valleys[0][-2]]

        else:
            self.field_data['Bx']['borders'] = [bx_valleys[0][border_indices[0]], bx_valleys[0][border_indices[1]]]
            self.field_data['By']['borders'] = [by_peaks[0][border_indices[2]], by_valleys[0][border_indices[4]]]

        return

    # CHECK
    def fit_sinusoidal_region(self, threshold=2):
        """Fit sinusoidal region using FFT-based approach"""# Get the amplitudes and frequencies of the (co)sines present in the data.
        # Played a bit with the threshold. Threshold=1 gives four sinusoids for B_x and three for B_y.
        # Threshold=2 gives three sinusoids for B_x and one for B_y and still a good fit.

        xslice = slice(self.field_data['Bx']['borders'][0], self.field_data['Bx']['borders'][1])
        yslice = slice(self.field_data['By']['borders'][0], self.field_data['By']['borders'][1])

        z_region = self.z_full[xslice]
        bx_region = self.field_data['Bx']['data'][xslice]
        by_region = self.field_data['By']['data'][yslice]

        bx_fft_amps, bx_fft_k = self.find_frequencies(z_region, bx_region, threshold=2)
        by_fft_amps, by_fft_k = self.find_frequencies(z_region, by_region, threshold=2)

        # The amplitudes of the (co)sine terms are related to the Fourier Transform of the data.
        xcos_amp_guesses =  2 * self.dz * bx_fft_amps.real
        xsin_amp_guesses = -2 * self.dz * bx_fft_amps.imag
        ycos_amp_guesses =  2 * self.dz * by_fft_amps.real
        ysin_amp_guesses = -2 * self.dz * by_fft_amps.imag

        # An empty array. It will be arranged as follows:
        # [A1.1, A1.2, freq1, A2.1, A2.2, freq2, ...]
        # So the amplitude of the cosines are the 0, 3, 6, ... entries,
        # The amplitude of the sines are the 1, 4, 7, ... entries
        # The frequencies are the 2, 5, 8, ... entries.
        xguesses = np.zeros(len(bx_fft_k) * 3)

        for ii in range(len(bx_fft_k)):
            xguesses[3 * ii]     = xcos_amp_guesses[ii]
            xguesses[3 * ii + 1] = xsin_amp_guesses[ii]
            xguesses[3 * ii + 2] = 2 * np.pi * bx_fft_k[ii]

        # Defined the same way as above, but for B_y.
        yguesses = np.zeros(len(by_fft_k) * 3)

        for ii in range(len(by_fft_k)):
            yguesses[3 * ii]     = ycos_amp_guesses[ii]
            yguesses[3 * ii + 1] = ysin_amp_guesses[ii]
            yguesses[3 * ii + 2] = 2 * np.pi * by_fft_k[ii]

        # Fit the curve.
        self.field_data['Bx']['sine_fit_params'], xpcovregsines = curve_fit(self.sinusoid, z_region, bx_region, p0=xguesses)
        self.field_data['By']['sine_fit_params'], ypcovregsines = curve_fit(self.sinusoid, z_region, by_region, p0=yguesses)

        return

    def get_boundary_conditions(self, x, sinusoid=False):
        # Boundaries contains the values, derivatives and second derivatives at the boundary points.
        # Even entries contain the boundary values at the left side of the region in question.
        # Odd entries contain the boundary values at the right side of the region in question.
        boundary_conds = np.zeros(6)

        # The point that needs to match is actually one point to the left of the region in question.
        xleft = x[0] - self.dz
        xright = x[-1] + self.dz

        if sinusoid:
            # If the object are the coefficients of a sinusoid, then calculate the derivatives analytically.
            # The derivatives of the sinusoid are known and simple to evaluate.
            for i in range(len(object) // 3):
                A1 = object[3 * i]
                A2 = object[3 * i + 1]
                k  = object[3 * i + 2]

                boundary_conds[0] += A1 * np.cos(k * xleft) + A2 * np.sin(k * xleft)
                boundary_conds[1] += A1 * np.cos(k * xright) + A2 * np.sin(k * xright)
                boundary_conds[2] += k * (A2 * np.cos(k * xleft) - A1 * np.sin(k * xleft))
                boundary_conds[3] += k * (A2 * np.cos(k * xright) - A1 * np.sin(k * xright))
                boundary_conds[4] += -k * k * boundary_conds[0]
                boundary_conds[5] += -k * k * boundary_conds[1]
        else:
            # If object is a polynomial, then calculate the derivative using polyder.
            # The boundaries are evaluated by evaluating the polynomial and its derivatives at the boundaries.
            dxpoly = object.deriv()
            ddxpoly = object.deriv(2)

            boundary_conds[0] = object(xleft)
            boundary_conds[1] = object(xright)
            boundary_conds[2] = dxpoly(xleft)
            boundary_conds[3] = dxpoly(xright)
            boundary_conds[4] = ddxpoly(xleft)
            boundary_conds[5] = ddxpoly(xright)

        return boundary_conds

    def get_balanced_slices(self, arr, num_regions):
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

    def fit_edge_regions(self, degree=3, num_slices=None):
        for field in self.field_data:  # 'Bx' and 'By'
            borders = self.field_data[field]['borders']
            z_region = self.z_full[borders[0]:borders[1]]
            b_region = self.field_data[field]['data'][borders[0]:borders[1]]

            for side in self.field_data[field]['poly_fit_params']:  # 'left' and 'right'
                slices = self.get_balanced_slices(z_region, num_slices)
                fit_reg = np.zeros_like(z_region)
                polys = []

                # Reverse slices for 'left', keep order for 'right'
                if side == 'left':
                    slices.reverse()

                for idx, slc in enumerate(slices):
                    z_this = z_region[slc]
                    b_this = b_region[slc]

                    if idx == 0:
                        # Use the sine fit parameters for the boundary
                        z_sines = z_region
                        poptsines = self.field_data[field]['sine_fit_params']
                        boundaries = self.get_boundary_conditions(z_sines, poptsines, sinusoid=True)
                    else:
                        # Use the previous polynomial for the boundary
                        boundaries = self.get_boundary_conditions(z_this, polys[-1], sinusoid=False)

                    fit = poly_fit(degree, z_this, b_this, boundaries, side == 'left')
                    poly = np.polynomial.Polynomial(fit)
                    polys.append((slc.start, poly))  # Store start index and polynomial
                    fit_reg[slc] = poly(z_this)

                self.field_data[field]['poly_fit_params'][side] = polys

                # Add fit_reg to the correct region in the full fit array
                if side == 'left':
                    start = borders[0]
                    end = borders[0] + len(fit_reg)
                else:  # 'right'
                    start = borders[1] - len(fit_reg)
                    end = borders[1]
                self.field_data[field]['fit'][start:end] = fit_reg

        return

    def evaluate_field(self, z):
        """
        Evaluate fitted field at point z.
        Returns (Bx, By) using analytic expressions for the polynomial and sinusoidal regions.
        """
        result = []
        for field in ['Bx', 'By']:
            borders = self.field_data[field]['borders']
            # Check which region z is in
            if z < self.z_full[borders[0]]:
                # Left polynomial region
                polys = self.field_data[field]['poly_fit_params']['left']
                # Find the correct polynomial for z
                for start_idx, poly in reversed(polys):
                    if z >= self.z_full[start_idx]:
                        val = poly(z)
                        break
                else:
                    val = polys[0][1](z)
            elif z > self.z_full[borders[1] - 1]:
                # Right polynomial region
                polys = self.field_data[field]['poly_fit_params']['right']
                for start_idx, poly in polys:
                    if z >= self.z_full[start_idx]:
                        val = poly(z)
                val = polys[-1][1](z) if polys else 0.0
            else:
                # Sinusoidal region
                params = self.field_data[field]['sine_fit_params']
                val = self.sinusoid(np.array([z]), *params)[0]
            result.append(val)
        return tuple(result)

    def plot_results(self):
        """Generate diagnostic plots"""
        import matplotlib.pyplot as plt
        z_values = self.z_full
        bx_values = self.field_data['Bx']['data']
        by_values = self.field_data['By']['data']
        bx_fit = self.field_data['Bx']['fit']
        by_fit = self.field_data['By']['fit']
        xborderleft, xborderright = self.field_data['Bx']['borders']
        yborderleft, yborderright = self.field_data['By']['borders']
        xy_point = self.xy_point

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(z_values, bx_values, label="$B_x$ Data")
        ax1.plot(z_values, bx_fit, label="$B_x$ Section Fit")
        ax2.plot(z_values, by_values, label="$B_y$ Data")
        ax2.plot(z_values, by_fit, label="$B_y$ Section Fit")

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

        # Make a legend.
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")

        # Turn on the grids.
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        plt.show()

    def save_fit_parameters(self, output_file):
        """Save fit parameters for later use"""
        # Save polynomial coefficients and sine parameters