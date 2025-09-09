from wiggler_class import Wiggler
import bpmeth as bp
import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
# TEST THE CLASS
########################################################################################################################

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

Test_Wiggler.plot_second_derivatives()
Test_Wiggler.plot_second_derivative_fit()
Test_Wiggler.plot_second_derivative_fit(field="By")
plt.show()

print(Test_Wiggler.curvature_series_string(in_terms_of="s"))
print(Test_Wiggler.curvature_series_string(in_terms_of="s", field="By"))\

# The Wiggler.plot_fields() method plots the original field and the fitted field.
# The Wiggler.plot_integral() method plots the integral of the field.
#Test_Wiggler.plot_fields()
#Test_Wiggler.plot_integral()

Bx_Sinusoid_String = Test_Wiggler.fields['Bx'].sine.series_string()
By_Sinusoid_String = Test_Wiggler.fields['By'].sine.series_string()
Bz_Sinusoid_String = Test_Wiggler.fields['Bz'].sine.series_string()

Bxpp_Sinusoid_String = Test_Wiggler.curvature_series_string(in_terms_of="s")
Bypp_Sinusoid_String = Test_Wiggler.curvature_series_string(in_terms_of="s", field="By")

z0x, z1x = Test_Wiggler.fields["Bx"].borders_z
z0y, z1y = Test_Wiggler.fields["By"].borders_z

print("Bx borders: ", z0x, z1x)
print("By borders: ", z0y, z1y)

print("Sinusoid Bx: ", Bx_Sinusoid_String)
print("Sinusoid By: ", By_Sinusoid_String)

a0 = 0.0
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
#wiggler.plotfield_yz(zmin=z0x, zmax=z1x, ymin=-0.001, ymax=0.001, ystep=0.0001, zstep=0.001)
#wiggler.plotfield_xz(zmin=z0x, zmax=z1x, xmin=-0.001, xmax=0.001, xstep=0.0001, zstep=0.001)

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
Bx = Test_Wiggler.fields["Bx"].data[leftidx:rightidx]
By = Test_Wiggler.fields["By"].data[leftidx:rightidx]
Bz = Test_Wiggler.fields["Bz"].data[leftidx:rightidx]

Z  = Test_Wiggler.z_full[leftidx:rightidx]

# NOTE: The offsets are: -0.007 for x, and 0.02 for y.
# NOTE: Bs is way off.

fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)
ax1.plot(Z, Bx, label="Data")
ax1.plot(Z, Bxfun(dz*xoffset, dz*yoffset, Z), label="Bx bpmeth", linestyle='dashed')
ax2.plot(Z, By, label="Data")
ax2.plot(Z, Byfun(dz*xoffset, dz*yoffset, Z), label="By bpmeth", linestyle='dashed')
ax3.plot(Z, Bz, label="Data")
ax3.plot(Z, Bsfun(dz*xoffset, dz*yoffset, Z), label="Bs bpmeth", linestyle='dashed')
ax1.set_title(f'Field Comparison at {xoffset, yoffset}')
ax1.set_ylabel('Bx (T)')
ax2.set_ylabel('By (T)')
ax3.set_ylabel('Bz (T)')
ax3.set_xlabel('Z (m)')
ax1.grid()
ax2.grid()
ax3.grid()

# ----------------------------------------------------------------------------------------------------------------------
# Show a comparison of the fitted field and the original field at (1, 1).
xoffset = 1.0
yoffset = -1.0

Test_Wiggler.xy_point = (xoffset, yoffset)
Test_Wiggler._select_xy()
Bx = Test_Wiggler.fields["Bx"].data[leftidx:rightidx]
By = Test_Wiggler.fields["By"].data[leftidx:rightidx]
Bz = Test_Wiggler.fields["Bz"].data[leftidx:rightidx]

kx1 = Test_Wiggler.fields["Bx"].sine.k[0]
ky1 = Test_Wiggler.fields["By"].sine.k[0]
lambdax1 = 2 * np.pi / kx1
lambday1 = 2 * np.pi / ky1


ax1.plot(Z, Bx, label="Data")
ax1.plot(Z, Bxfun(dz*xoffset, dz*yoffset, Z), label="Bx bpmeth", linestyle='dashed')
ax2.plot(Z, By, label="Data")
ax2.plot(Z, Byfun(dz*xoffset, dz*yoffset, Z), label="By bpmeth", linestyle='dashed')
ax3.plot(Z, Bz, label="Data")
ax3.plot(Z, Bsfun(dz*xoffset, dz*yoffset, Z), label="Bs bpmeth", linestyle='dashed')
ax1.legend()
ax2.legend()
ax3.legend()

plt.show()