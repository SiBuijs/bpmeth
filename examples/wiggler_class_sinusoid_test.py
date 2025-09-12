from wiggler_class import Wiggler
import bpmeth as bp
import matplotlib.pyplot as plt
import numpy as np

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
Test_Wiggler = Wiggler(file_path='example_data/knot_map_test.txt',
                       xy_point=(0, 0),
                       dx=dz,
                       dy=dz,
                       dz=dz,
                       x_left_slices=23,
                       x_right_slices=21,
                       y_left_slices=20,
                       y_right_slices=30,
                       n_modes_x=4,
                       n_modes_y=4,
                       n_modes_z=1)

# The Wiggler.fit() method fits the field with sinusoids in the middle and polynomials on the edges.
# The Wiggler.tune_slices_for_zero_integral() method tunes the number of slices for the polynomials
# to achieve zero integral of the field.
Test_Wiggler.fit(
    curvature=True,
    curvature_axes=("x","y"),       # choose which axes to fit curvature on
    curvature_fields=("Bx","By"),   # which components
    n_modes={"Bx":6,"By":3},        # optional per-field mode caps
    reuse_borders=True,             # reuse each field’s mid window
    center_xy=(0,0),                # transverse center to evaluate parabolas at
)

Test_Wiggler.tune_slices_for_zero_integral(field="both", tradeoff_mse=0.0,
                                           left_candidates=range(8, 24),
                                           right_candidates=range(8, 24),
                                           verbose=True)

# Field piecewise for Bx:
expr_bx = Test_Wiggler.to_piecewise_string("Bx", coeff_fmt=".12g")
print(expr_bx)
# -> "Piecewise((...), (...), ...)"

# Curvature piecewise for d²Bx/dx²:
curv_bx = Test_Wiggler.to_piecewise_curvature_string("Bx", axis="x", coeff_fmt=".12g")
print(curv_bx)

Test_Wiggler.plot_second_derivatives()
Test_Wiggler.plot_second_derivative_fit()
Test_Wiggler.plot_second_derivative_fit(field="By")
plt.show()

Test_Wiggler.plot_fields()
Test_Wiggler.plot_integral()
#Test_Wiggler.plot_fields()
#Test_Wiggler.plot_integral()

Bx_Sinusoid_String = Test_Wiggler.fields['Bx'].sine.series_string()
By_Sinusoid_String = Test_Wiggler.fields['By'].sine.series_string()
Bz_Sinusoid_String = Test_Wiggler.fields['Bz'].sine.series_string()

Bxpp_Sinusoid_String = Test_Wiggler.curvature_series_string(in_terms_of="s")
Bypp_Sinusoid_String = Test_Wiggler.curvature_series_string(in_terms_of="s", field="By")

z0x, z1x = Test_Wiggler.fields["Bx"].borders_z
z0y, z1y = Test_Wiggler.fields["By"].borders_z

a1 = Bx_Sinusoid_String
b1 = By_Sinusoid_String
bs = Bz_Sinusoid_String

a2 = 0
b2 = 0

a3 = Bxpp_Sinusoid_String
b3 = Bypp_Sinusoid_String

curv=0

wiggler = bp.GeneralVectorPotential(hs=f"{curv}",a=(f"{a1}", f"{a2}", f"{a3}"),b=(f"{b1}", f"{b2}", f"{b3}"), bs=f"{bs}") # Creates a wiggler from the sinusoid strings
#wiggler = bp.FringeVectorPotential(b1=b1) # Does the same as above
#wiggler.plotfield_yz(zmin=z0x, zmax=z1x, ymin=-0.001, ymax=0.001, ystep=0.00005, zstep=0.0005)
#wiggler.plotfield_xz(zmin=z0x, zmax=z1x, xmin=-0.001, xmax=0.001, xstep=0.00005, zstep=0.0005)
#wiggler.plotfield_z(X=0.001, Y=0.001, zmin=z0x, zmax=z1x, zstep=0.001)

# NOTE: Investigate how bpmeth makes these functions, there might be something that causes discrepancies.
Bxfun, Byfun, Bsfun = wiggler.get_Bfield()

leftidx = Test_Wiggler.fields["Bx"].borders_idx[0]
rightidx = Test_Wiggler.fields["Bx"].borders_idx[1]

# ----------------------------------------------------------------------------------------------------------------------
# Show a comparison of the fitted field and the original field at (0, 0).
xoffset = 0.0
yoffset = 0.0

Test_Wiggler.xy_point = (xoffset, yoffset)
Test_Wiggler._select_xy()
Bx00 = Test_Wiggler.fields["Bx"].data[leftidx:rightidx]
By00 = Test_Wiggler.fields["By"].data[leftidx:rightidx]
Bz00 = Test_Wiggler.fields["Bz"].data[leftidx:rightidx]

Z  = Test_Wiggler.z_full[leftidx:rightidx]

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
Test_Wiggler._select_xy()
Bx10 = Test_Wiggler.fields["Bx"].data[leftidx:rightidx]
By10 = Test_Wiggler.fields["By"].data[leftidx:rightidx]
Bz10 = Test_Wiggler.fields["Bz"].data[leftidx:rightidx]


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