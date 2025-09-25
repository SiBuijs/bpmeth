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
Test_Wiggler.plot_fields()
Test_Wiggler.plot_integrated_fields()

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

Bx_string, separatex = Test_Wiggler.export_piecewise_string(component="Bx")
Bx_der_string, separatex_der = Test_Wiggler_Der.export_piecewise_string(component="Bx")
By_string, separatey = Test_Wiggler.export_piecewise_string(component="By")
By_der_string, separatey_der = Test_Wiggler_Der.export_piecewise_string(component="By")
Bs_string, separates = Test_Wiggler.export_piecewise_string(component="Bs")

#print(Test_Wiggler.debug_piece_counts("Bs"))
#print(Test_Wiggler_Der.debug_piece_counts("Bs"))
#print(By_string)
#print(Bs_string)
#print(Bx_der_string)
#print(By_der_string)

a1 = Bx_string
b1 = By_string
bs = "0"#Bs_string

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

# Screw the Hamiltonian. Let's try some manual tracking.

def lorentz_force(x, y, z, px, py, pz, q, m, gamma, Bxfun, Byfun, Bsfun):
    # Calculate the magnetic field at the given position
    Bx = Bxfun(x, y, z)
    By = Byfun(x, y, z)
    Bz = Bsfun(x, y, z)

    # Calculate the velocity components
    vx = px / (m * gamma)
    vy = py / (m * gamma)
    vz = pz / (m * gamma)

    # Calculate the derivatives of momentum using the Lorentz force equation
    dpx_dt = q * (vy * Bz - vz * By)
    dpy_dt = q * (vz * Bx - vx * Bz)
    dpz_dt = q * (vx * By - vy * Bx)

    return dpx_dt, dpy_dt, dpz_dt

def runge_kutta_step(x, y, z, px, py, pz, q, m, gamma, Bxfun, Byfun, Bsfun, dt):
    # Calculate k1 values
    dpx1, dpy1, dpz1 = lorentz_force(x, y, z, px, py, pz, q, m, gamma, Bxfun, Byfun, Bsfun)
    k1_px = dpx1 * dt
    k1_py = dpy1 * dt
    k1_pz = dpz1 * dt
    k1_x = (px / (m * gamma)) * dt
    k1_y = (py / (m * gamma)) * dt
    k1_z = (pz / (m * gamma)) * dt

    # Calculate k2 values
    dpx2, dpy2, dpz2 = lorentz_force(x + 0.5 * k1_x, y + 0.5 * k1_y, z + 0.5 * k1_z,
                                     px + 0.5 * k1_px, py + 0.5 * k1_py, pz + 0.5 * k1_pz,
                                     q, m, gamma, Bxfun, Byfun, Bsfun)
    k2_px = dpx2 * dt
    k2_py = dpy2 * dt
    k2_pz = dpz2 * dt
    k2_x = ((px + 0.5 * k1_px) / (m * gamma)) * dt
    k2_y = ((py + 0.5 * k1_py) / (m * gamma)) * dt
    k2_z = ((pz + 0.5 * k1_pz) / (m * gamma)) * dt

    # Calculate k3 values
    dpx3, dpy3, dpz3 = lorentz_force(x + 0.5 * k2_x, y + 0.5 * k2_y, z + 0.5 * k2_z,
                                     px + 0.5 * k2_px, py + 0.5 * k2_py, pz + 0.5 * k2_pz,
                                     q, m, gamma, Bxfun, Byfun, Bsfun)
    k3_px = dpx3 * dt
    k3_py = dpy3 * dt
    k3_pz = dpz3 * dt
    k3_x = ((px + 0.5 * k2_px) / (m * gamma)) * dt
    k3_y = ((py + 0.5 * k2_py) / (m * gamma)) * dt
    k3_z = ((pz + 0.5 * k2_pz) / (m * gamma)) * dt

    # Calculate k4 values
    dpx4, dpy4, dpz4 = lorentz_force(x + k3_x, y + k3_y, z + k3_z,
                                     px + k3_px, py + k3_py, pz + k3_pz,
                                     q, m, gamma, Bxfun, Byfun, Bsfun)
    k4_px = dpx4 * dt
    k4_py = dpy4 * dt
    k4_pz = dpz4 * dt
    k4_x = ((px + k3_px) / (m * gamma)) * dt
    k4_y = ((py + k3_py) / (m * gamma)) * dt
    k4_z = ((pz + k3_pz) / (m * gamma)) * dt

    # Update momentum and position
    px_new = px + (k1_px + 2 * k2_px + 2 * k3_px + k4_px) / 6
    py_new = py + (k1_py + 2 * k2_py + 2 * k3_py + k4_py) / 6
    pz_new = pz + (k1_pz + 2 * k2_pz + 2 * k3_pz + k4_pz) / 6
    x_new = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    y_new = y + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
    z_new = z + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

    return x_new, y_new, z_new, px_new, py_new, pz_new

# Particle properties
q = -1.602e-19  # Charge of the electron in Coulombs
m = 9.109e-31   # Mass of the electron in kg
E = 2.4e9 * 1.602e-19  # Energy in Joules (3 GeV)
c = 299792458  # Speed of light in m/s
gamma = E / (m * c**2)  # Lorentz factor

# Initial conditions
x0 = 0.0  # Initial x position in meters
y0 = 0.0    # Initial y position in meters
z0 = Test_Wiggler.s_full[5]    # Initial z position in meters
px0 = 0   # Initial x momentum in kg*m/s
py0 = 0   # Initial y momentum in kg*m/s
pz0 = np.sqrt((E/c)**2 - px0**2 - py0**2)  # Initial z momentum in kg*m/s

# Time parameters
length = Test_Wiggler.s_full[-4] - Test_Wiggler.s_full[5]
num_steps = 1000  # Number of time steps
dt = length/c/num_steps  # Time step in seconds
# Arrays to store the trajectory
x_traj = np.zeros(num_steps)
y_traj = np.zeros(num_steps)
z_traj = np.zeros(num_steps)
px_traj = np.zeros(num_steps)
py_traj = np.zeros(num_steps)
pz_traj = np.zeros(num_steps)

# Set initial values
x_traj[0] = x0
y_traj[0] = y0
z_traj[0] = z0
px_traj[0] = px0
py_traj[0] = py0
pz_traj[0] = pz0

start_time = time.time()
# Time integration using Runge-Kutta method
for i in range(1, num_steps):
    x_traj[i], y_traj[i], z_traj[i], px_traj[i], py_traj[i], pz_traj[i] = runge_kutta_step(
        x_traj[i-1], y_traj[i-1], z_traj[i-1],
        px_traj[i-1], py_traj[i-1], pz_traj[i-1],
        q, m, gamma, Bxfun, Byfun, Bsfun, dt
    )

end_time = time.time()
print(f"Time to do manual tracking: {end_time - start_time} seconds")

# Plot the trajectory
plt.figure(figsize=(10, 6))
plt.plot(z_traj, x_traj * 1e3, label='x (mm)')
plt.plot(z_traj, y_traj * 1e3, label='y (mm)')
plt.xlabel('z (m)')
plt.ylabel('Transverse Position (mm)')
plt.title('Particle Trajectory through the Wiggler')
plt.legend()
plt.grid()
plt.show()