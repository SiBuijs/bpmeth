from wiggler_class import FieldFitter
from wiggler_class import SymbolicGenerator
from wiggler_class import FieldCalculator
import os.path
import inspect
#from wiggler_class import WigglerFull
import sympy as sp
import xtrack as xt
import matplotlib.pyplot as plt
import numpy as np
import bpmeth
import pandas as pd
import time

import cProfile

########################################################################################################################
# TEST THE CLASS
########################################################################################################################

dz = 0.001  # Step size in the z direction for numerical differentiation.

file_path = 'example_data/knot_map_test.txt'
#file_path = 'example_data/UE36kn3_LH.dat'
#file_path = 'example_data/UE36_LH_highres_2.dat'

test_wiggler = FieldFitter(
                                    file_path,
                                    xy_point=(0, 0),
                                    dx=0.001,
                                    dy=0.001,
                                    ds=0.001,
                                    min_region_size=10,
                                    deg=2,
                            )

fit_par_path = 'fit_parameters.csv'

test_wiggler.set()
test_wiggler.save_fit_pars(fit_par_path)

df_fit_pars = test_wiggler.df_fit_pars

symbolic_wiggler = SymbolicGenerator(test_wiggler)

symbolic_wiggler.write_to_python(field='B')
symbolic_wiggler.write_to_python(field='A')
symbolic_wiggler.write_to_c(field='B')
symbolic_wiggler.write_to_c(field='A')

start_time = time.time()
symbolic_wiggler.compile_C_code(field='B')
symbolic_wiggler.compile_C_code(field='A')
end_time = time.time()
print(f"Time taken to compile C code for B field: {end_time - start_time} seconds")

field_calculator = FieldCalculator(filepath=fit_par_path)

x = 0.001
y = 0.001

iter = 1000
steps = 10000

s_vals = np.linspace(-1, 1, steps)
x_vals = np.zeros_like(s_vals)
y_vals = np.zeros_like(s_vals)

#start_time = time.time()
with cProfile.Profile() as prof:
    Bx_p, By_p, Bz_p = field_calculator.get_Bfield(x_arr=x_vals, y_arr=y_vals, s_arr=s_vals, python=False)
print(prof.print_stats())
#end_time = time.time()
#print(f"Time taken to evaluate B field on array using C code: {end_time - start_time} seconds")
#print(f"Time per evaluation using C code on array: {(end_time - start_time)/(iter*steps)} seconds")

field_calculator.plot_B_field(x_arr=x, y_arr=y, python=False)
field_calculator.plot_A_field(x_arr=x, y_arr=y, python=False)

# Build simple tracker through the undulator
mass = xt.ELECTRON_MASS_EV
charge = 1
energy = 2.7e9 #eV
gamma = energy / mass
beta = np.sqrt(1 - 1 / gamma**2)
p0c = np.sqrt(energy**2 - mass**2)  # eV/c

# Grid parameters:
n_steps = 10000
length = field_calculator.s_end[-1] - field_calculator.s_start[0] # meters
ds = length / n_steps
x0 = 0.0
y0 = 0.0
s0 = field_calculator.s_start[0]
px0 = 0.0
py0 = 0.0
ps0 = p0c


# Simple Euler integrator using field_calculator.get_Bfield
xs = np.zeros(n_steps + 1)
ys = np.zeros(n_steps + 1)
zs = np.zeros(n_steps + 1)
pxs = np.zeros(n_steps + 1)
pys = np.zeros(n_steps + 1)
pzs = np.zeros(n_steps + 1)

xs[0] = x0
ys[0] = y0
zs[0] = s0

# SI constants and conversions
c = 299792458.0
eV_to_J = 1.602176634e-19
mass_eV = mass          # xt.ELECTRON_MASS_EV (eV)
energy_eV = energy      # (eV)
mass_kg = mass_eV * eV_to_J / c**2
energy_J = energy_eV * eV_to_J
# convert initial momenta (eV/c) to SI momentum (kg*m/s)
pxs[0] = px0 * eV_to_J / c
pys[0] = py0 * eV_to_J / c
pzs[0] = ps0 * eV_to_J / c

q_SI = charge * 1.602176634e-19  # elementary charge * sign

with cProfile.Profile() as prof:
    for i in range(n_steps):
        s_i = s0 + i * ds
        # request B field at current position (single-point arrays)
        Bx_arr, By_arr, Bz_arr = field_calculator.get_Bfield(
            x_arr=np.array([xs[i]]), y_arr=np.array([ys[i]]), s_arr=np.array([s_i]), python=False
        )
        B = np.array([Bx_arr[0], By_arr[0], Bz_arr[0]])

        p_vec = np.array([pxs[i], pys[i], pzs[i]])
        # relativistic velocity: v = p * c^2 / E
        v = p_vec * c**2 / energy_J
        vz = v[2]
        if abs(vz) < 1e-12:
            vz = 1e-12
        dt = ds / vz

        # Lorentz force (no E-field): dp/dt = q * (v x B)
        dpdt = q_SI * np.cross(v, B)
        dp = dpdt * dt

        p_new = p_vec + dp
        r_new = np.array([xs[i], ys[i], zs[i]]) + v * dt

        xs[i + 1], ys[i + 1], zs[i + 1] = r_new
        pxs[i + 1], pys[i + 1], pzs[i + 1] = p_new
print(prof.print_stats())


# Plot:
plt.figure()
plt.plot(zs, xs, label='x(z)')
plt.plot(zs, ys, label='y(z)')
plt.xlabel('s (m)')
plt.ylabel('Transverse position (m)')
plt.title('Particle trajectory through the wiggler')
plt.legend()
plt.grid()
plt.show()
"""
prrr

field_calculator.set_integrator()
wig_line = field_calculator.get_line()

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, energy0=2.7e9)

wig_line.particle_ref = p0.copy()

tw_wig = wig_line.twiss4d(include_collective=True, betx=1, bety=1)

tw_wig.plot('x y')
tw_wig.plot('betx bety', 'dx dy')
plt.show()

#env = xt.load('example_data/b075_2024.09.25.madx')
#line = env.ring
"""