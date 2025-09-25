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
                       n_modes=[3, 3, 1],
                       enge_deg=[[8, 5], [5, 5], [5, 5]],
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
                           enge_deg=[[8, 5], [8, 10], [5, 5]],
                           peak_window=(99, 2100),
                           der=True,
                           filter_params=(None, 2090, 7, 11, 3)
)

Test_Wiggler_Der.set()
#Test_Wiggler_Der.plot_fields()

Bx_string, Cx = Test_Wiggler.export_piecewise_string(component="Bx")
Bx_der_string, Cx_der = Test_Wiggler_Der.export_piecewise_string(component="Bx")
By_string, Cy = Test_Wiggler.export_piecewise_string(component="By")
By_der_string, Cy_der = Test_Wiggler_Der.export_piecewise_string(component="By")
Bs_string, Cs = Test_Wiggler.export_piecewise_string(component="Bs")

#print(Test_Wiggler.debug_piece_counts("Bs"))
#print(Test_Wiggler_Der.debug_piece_counts("Bs"))
#print(By_string)
#print(Bs_string)
#print(Bx_der_string)
#print(By_der_string)

a1 = Bx_string
b1 = By_string
bs = Bs_string

a2 = 0
b2 = 0

a3 = Bx_der_string
b3 = By_der_string

#print(f"Bx string: {a1}")
#print(f"By string: {b1}")
#print(f"Bs string: {bs}")
#print(f"Bx der string: {a3}")
#print(f"By der string: {b3}")

# NOTE:
# a1 = "0" gives cut between first/last extremum in By and first/last extremum in Bx
# b1 = "0" gives cut between first extremum in By and second extremum in Bx
# b1 = "0" gives cut between second last and last extremum in Bx
# Both a1 and b1 = "0" gives cut between first extremum in By and first extremum in Bx
#   Hard to explain gaps on the right



curv=0

wiggler = bp.GeneralVectorPotential(hs=f"{curv}",a=(f"{a1}", f"{a2}", f"{a3}"),b=(f"{b1}", f"{b2}", f"{b3}"), bs=f"{bs}")

print("Made the Vector Potential.\nNow making the functions...")

import time
start_time = time.time()
# NOTE: Investigate how bpmeth makes these functions, there might be something that causes discrepancies.
Bxfun, Byfun, Bsfun = wiggler.get_Bfield()
end_time = time.time()
print(f"Time to make the functions: {end_time - start_time} seconds")

print("Made the functions.\nNow plotting...")


# ----------------------------------------------------------------------------------------------------------------------
# Show a comparison of the fitted field and the original field at (0, 0).
xoffset = 0.0
yoffset = 0.0
cut_idx_L = 4
cut_idx_R = -5
slice = slice(cut_idx_L, cut_idx_R)

Test_Wiggler.xy_point = (xoffset, yoffset)
Test_Wiggler.select_xy()
Bx00 = Test_Wiggler.raw_data["Bx"][slice]
By00 = Test_Wiggler.raw_data["By"][slice]
Bz00 = Test_Wiggler.raw_data["Bs"][slice]

Z  = Test_Wiggler.s_full[slice]

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
# Show a comparison of the fitted field and the original field at (1, 0).
xoffset = 1.0
yoffset = 1.0

Test_Wiggler.xy_point = (xoffset, yoffset)
Test_Wiggler.select_xy()
Bx10 = Test_Wiggler.raw_data["Bx"][slice]
By10 = Test_Wiggler.raw_data["By"][slice]
Bz10 = Test_Wiggler.raw_data["Bs"][slice]


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

print("Plotted the fields.\nNow making the Hamiltonian...")

qp0 = [0,0,0,0,0,0]
length = Test_Wiggler.s_full[-1] - Test_Wiggler.s_full[0]
start_time = time.time()
H_wiggler = bp.Hamiltonian(length, curv, wiggler)
end_time = time.time()
print(f"Time to make the Hamiltonian: {end_time - start_time} seconds")

"""
print("Made the Hamiltonian.\nNow solving the equations of motion...")

start_time = time.time()
ivp_opt={"rtol":1e-4, "atol":1e-7}
sol_wiggler = H_wiggler.solve(qp0, ivp_opt=ivp_opt)
end_time = time.time()
print(f"Time to solve the equations of motion: {end_time - start_time} seconds")

H_wiggler.plotsol(qp0, ivp_opt=ivp_opt)
plt.show()
"""

print("Finished first track")

import xtrack as xt

start_time = time.time()
p = xt.Particles(x = np.linspace(-1e-3, 1e-3, 1), energy0=10e9, mass0=xt.ELECTRON_MASS_EV)
sol = H_wiggler.track(p, return_sol=True)
end_time = time.time()
print(f"Time to track with xtrack: {end_time - start_time} seconds")

print(sol[0].keys())
t = sol[0]['t']
y = sol[0]['y'][0]

plt.plot(t,y)
plt.show()

print("Finished second track")