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

field_calculator = FieldCalculator(filepath=fit_par_path)

x = 0.001
y = 0.001

iter = 100000

s_vals = np.linspace(-1, 1, iter)

start_time = time.time()
Bx, By, Bz = field_calculator.get_Bfield(x=x, y=y, s=s_vals, python=True)
#for s_val in s_vals:
#    Bx, By, Bz = field_calculator.get_Bfield(x=x, y=y, s=s_val, python=True)
end_time = time.time()

print(f"Time taken to evaluate B field on grid: {end_time - start_time} seconds")
print(f"Time per evaluation: {(end_time - start_time)/iter} seconds")

start_time = time.time()
symbolic_wiggler.compile_C_code(field='B')
end_time = time.time()
print(f"Time taken to compile C code for B field: {end_time - start_time} seconds")

start_time = time.time()
Bx_c, By_c, Bz_c = field_calculator.get_Bfield(x=x, y=y, s=s_vals, python=True)
#for s_val in s_vals:
#    Bx_c, By_c, Bz_c = field_calculator.get_Bfield(x=x, y=y, s=s_val, python=False)
end_time = time.time()
print(f"Time taken to evaluate B field on grid using C code: {end_time - start_time} seconds")
print(f"Time per evaluation using C code: {(end_time - start_time)/iter} seconds")

field_calculator.plot_B_field(x=x, y=y, python=False)
#field_calculator.plot_A_field(x=x, y=y, python=False)

field_calculator.set_integrator()
wig_line = field_calculator.get_line()

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, energy0=2.4e9)

wig_line.particle_ref = p0.copy()

tw_wig = wig_line.twiss4d(include_collective=True, betx=1, bety=1)

tw_wig.plot('x y')

#env = xt.load('example_data/b075_2024.09.25.madx')
#line = env.ring

