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
                       peak_window=(99, 2100),
                       data_cut=(None, None),
                       n_modes=[3, 3, 1],
                       enge_deg=[[23, 23], [22, 23], [11, 11]],
                       der=False
)

Test_Wiggler.set()
#Test_Wiggler.plot_fields()
print("DERIVATIVES:")

Test_Wiggler_Der = Wiggler(file_path='example_data/knot_map_test.txt',
                           xy_point=(0, 0),
                           dx=dz,
                           dy=dz,
                           ds=dz,
                           n_modes=[6, 4, 1],
                           data_cut=(50, -50),
                           enge_deg=[[22, 23], [21, 23], [15, 15]],
                           peak_window=(99, 2100),
                           der=True,
                           filter_params=(85, 2015, 15, 21, 2)
)

Test_Wiggler_Der.set()
#Test_Wiggler_Der.plot_fields()

Bx_string = Test_Wiggler.export_piecewise_string(component="Bx")
Bx_der_string = Test_Wiggler_Der.export_piecewise_string(component="Bx")
By_string = Test_Wiggler.export_piecewise_string(component="By")
By_der_string = Test_Wiggler_Der.export_piecewise_string(component="By")
Bs_string = Test_Wiggler.export_piecewise_string(component="Bs")


a1 = Bx_string
b1 = By_string
bs = Bs_string

a2 = 0
b2 = 0

a3 = Bx_der_string
b3 = By_der_string

print(f"Bx string: {a1}")
print(f"By string: {b1}")
print(f"Bs string: {bs}")
print(f"Bx der string: {a3}")
print(f"By der string: {b3}")

curv=0

wiggler = bp.GeneralVectorPotential(hs=f"{curv}",a=(f"{a1}", f"{a2}", f"{a3}"),b=(f"{b1}", f"{b2}", f"{b3}"), bs=f"{bs}")

# NOTE: Investigate how bpmeth makes these functions, there might be something that causes discrepancies.
Bxfun, Byfun, Bsfun = wiggler.get_Bfield()

leftidx = Test_Wiggler.borders_idx["Bx"][0]
rightidx = Test_Wiggler.borders_idx["Bx"][1]

# ----------------------------------------------------------------------------------------------------------------------
# Show a comparison of the fitted field and the original field at (0, 0).
xoffset = 0.0
yoffset = 0.0

Test_Wiggler.xy_point = (xoffset, yoffset)
Test_Wiggler.select_xy()
Bx00 = Test_Wiggler.raw_data["Bx"]
By00 = Test_Wiggler.raw_data["By"]
Bz00 = Test_Wiggler.raw_data["Bs"]

Z  = Test_Wiggler.s_full

# NOTE: The offsets are: -0.007 for x, and 0.02 for y.
# NOTE: Bs is way off.

fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)
ax1.plot(Z, Bx00, label=f"Bx Data  ({xoffset}, {yoffset})")
ax1.plot(Z, Bxfun(dz*xoffset, dz*yoffset, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax2.plot(Z, By00, label=f"Bx Data  ({xoffset}, {yoffset})")
ax2.plot(Z, Byfun(dz*xoffset, dz*yoffset, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax3.plot(Z, Bz00, label=f"Bz Data  ({xoffset}, {yoffset})")
ax3.plot(Z, Bsfun(dz*xoffset, dz*yoffset, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax1.set_title(f'Field Comparison at {xoffset, yoffset}')
ax1.set_ylabel('Bx [T]')
ax2.set_ylabel('By [T]')
ax3.set_ylabel('Bz [T]')
ax3.set_xlabel('Z [m]')
ax1.grid()
ax2.grid()
ax3.grid()

# ----------------------------------------------------------------------------------------------------------------------
# Show a comparison of the fitted field and the original field at (1, 1).
xoffset = 1.0
yoffset = 0.0

Test_Wiggler.xy_point = (xoffset, yoffset)
Test_Wiggler.select_xy()
Bx10 = Test_Wiggler.raw_data["Bx"]
By10 = Test_Wiggler.raw_data["By"]
Bz10 = Test_Wiggler.raw_data["Bs"]


ax1.plot(Z, Bx10, label=f"Bx Data  ({xoffset}, {yoffset})")
ax1.plot(Z, Bxfun(dz*xoffset, dz*yoffset, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax2.plot(Z, By10, label=f"By Data  ({xoffset}, {yoffset})")
ax2.plot(Z, Byfun(dz*xoffset, dz*yoffset, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax3.plot(Z, Bz10, label=f"Bz Data  ({xoffset}, {yoffset})")
ax3.plot(Z, Bsfun(dz*xoffset, dz*yoffset, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax1.legend()
ax2.legend()
ax3.legend()

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Show a comparison of the fitted field and the original field at (1, 0).

fig2, (ax4, ax5, ax6) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)
ax4.plot(Z, Bx10 - Bx00, label=f"Bx Data  ({xoffset}, {yoffset})")
ax4.plot(Z, Bxfun(dz, 0, Z) - Bxfun(0, 0, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax5.plot(Z, By10 - By00, label=f"Bx Data  ({xoffset}, {yoffset})")
ax5.plot(Z, Byfun(dz, 0, Z) - Byfun(0, 0, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax6.plot(Z, Bz10 - Bz00, label=f"Bz Data  ({xoffset}, {yoffset})")
ax6.plot(Z, Bsfun(dz, 0, Z) - Bsfun(0, 0, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax4.set_title(f'Field Comparison at {xoffset, yoffset}')
ax4.set_ylabel('Bx(1, 0) - Bx(0, 0) [T]')
ax5.set_ylabel('By(1, 0) - By(0, 0) [T]')
ax6.set_ylabel('Bz(1, 0) - Bz(0, 0) [T]')
ax6.set_xlabel('Z [m]')
ax4.grid()
ax5.grid()
ax6.grid()

plt.show()
