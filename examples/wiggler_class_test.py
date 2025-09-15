from wiggler_class import Wiggler
from scipy import signal
import bpmeth as bp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt

########################################################################################################################
# TEST THE CLASS
########################################################################################################################

# Class has been successfully tested for the sinusoidal region in the middle.
# The procedure is as follows: We fit the field on-axis (0,0).
# We fit the transverse second derivative at (0,0).
# This is a function of s, which is in turn fitted to a sinusoid.
# We pass the resulting function parameters to bpmeth.
# bpmeth correctly reproduces the data off-axis.

dz = 0.001  # Step size in the z direction for numerical differentiation.

# Create a Wiggler with parameters:
# file_path: Path to the field map file.
# xy_point: (x,y) coordinates of the axis where the field is evaluated.
# dz: Step size in the z direction for numerical differentiation.
#       In this case, dz=0.001 rescales the distances to m instead of mm.
# x_left_slices: Number of slices to the left in the x direction for fitting.
# x_right_slices: Number of slices to the right in the x direction for fitting.
# y_left_slices: Number of slices to the left in the y direction for fitting.
# y_right_slices: Number of slices to the right in the y direction for fitting.
# n_modes_x: Number of modes in the x direction for fitting the sinusoid.
# n_modes_y: Number of modes in the y direction for fitting the sinusoid.
print("FIELDS:")
Test_Wiggler = Wiggler(file_path='example_data/knot_map_test.txt',
                       xy_point=(0, 0),
                       dx=dz,
                       dy=dz,
                       ds=dz,
                       n_modes=[3, 3, 1],
                       enge_deg=[[22, 15], [23, 20], [15, 15]],
                       der=False
)

#Test_Wiggler.set()
#Test_Wiggler.plot_fields()


print("DERIVATIVES:")

Test_Wiggler_Der = Wiggler(file_path='example_data/knot_map_test.txt',
                           xy_point=(0, 0),
                           dx=dz,
                           dy=dz,
                           ds=dz,
                           n_modes=[6, 4, 1],
                           enge_deg=[[22, 15], [23, 20], [15, 15]],
                           peak_window=(100, 2085),
                           der=True
)

Test_Wiggler_Der.set()
#Test_Wiggler_Der.plot_fields()

from scipy.signal import lfilter

x = Test_Wiggler_Der.s_full[Test_Wiggler_Der.borders_idx["Bx"][1]:]
y = Test_Wiggler_Der.raw_data["Bx"][Test_Wiggler_Der.borders_idx["Bx"][1]:]

n = 7  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b, a, y)
plt.plot(x, yy, linewidth=2, linestyle="-", c="b")  # smooth by filter
plt.plot(x, y, linewidth=1, linestyle="-", c="r")  # original data
plt.show()