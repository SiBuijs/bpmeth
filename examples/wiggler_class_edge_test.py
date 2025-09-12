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

Test_Wiggler.plot_fields()
Test_Wiggler.plot_integral()

prrr

print(curv_bx)

a1 = expr_bx
b1 = expr_by
bs = expr_bz

a2 = 0
b2 = 0

a3 = curv_bx
b3 = curv_by

wiggler = bp.GeneralVectorPotential(hs=f"{0}",a=(f"{a1}", f"{a2}", f"{a3}"),b=(f"{b1}", f"{b2}", f"{b3}"), bs=f"{bs}")

Bxfun, Byfun, Bsfun = wiggler.get_Bfield()

leftidx = Test_Wiggler.fields["Bx"].borders_idx[0]
rightidx = Test_Wiggler.fields["Bx"].borders_idx[1]

# ----------------------------------------------------------------------------------------------------------------------
# First do the left edge
# ----------------------------------------------------------------------------------------------------------------------
xoffset = 0.0
yoffset = 0.0

Test_Wiggler.xy_point = (xoffset, yoffset)
Test_Wiggler._select_xy()
Bx00 = Test_Wiggler.fields["Bx"].data[:leftidx]
By00 = Test_Wiggler.fields["By"].data[:leftidx]
Bz00 = Test_Wiggler.fields["Bz"].data[:leftidx]

Z  = Test_Wiggler.z_full[:leftidx]

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
yoffset = 1.0

Test_Wiggler.xy_point = (xoffset, yoffset)
Test_Wiggler._select_xy()
Bx11 = Test_Wiggler.fields["Bx"].data[:leftidx]
By11 = Test_Wiggler.fields["By"].data[:leftidx]
Bs11 = Test_Wiggler.fields["Bz"].data[:leftidx]

ax1.plot(Z, Bx11, label=f"Bx Data  ({xoffset}, {yoffset})")
ax1.plot(Z, Bxfun(dz*xoffset, dz*yoffset, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax2.plot(Z, By11, label=f"By Data  ({xoffset}, {yoffset})")
ax2.plot(Z, Byfun(dz*xoffset, dz*yoffset, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax3.plot(Z, Bs11, label=f"Bz Data  ({xoffset}, {yoffset})")
ax3.plot(Z, Bsfun(dz*xoffset, dz*yoffset, Z), label=f"Bx bpmeth  ({xoffset}, {yoffset})", linestyle='dashed')
ax1.legend()
ax2.legend()
ax3.legend()

plt.show()