from wiggler_class import FieldFitter
from wiggler_class import SymbolicGenerator
from wiggler_class import FieldCalculator
from fieldmap_parsers import StandardFieldMapParser, UE36FieldMapParser
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

# Standard 6-column format (X Y Z Bx By Bs)
# Resolve path relative to script location to handle different working directories
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'example_data', 'knot_map_test.txt')

# Alternative: UE36 format (Z, then Bx/By/Bs for each Y position)
# file_path = 'example_data/UE36kn3_LH.dat'
# file_path = 'example_data/UE36_LH_highres_2.dat'

# Create FieldFitter with auto-detection (default behavior)
# The parser will be automatically detected based on file format
test_wiggler = FieldFitter(
    file_path,
    xy_point=(0, 0),
    dx=0.001,
    dy=0.001,
    ds=0.001,
    min_region_size=10,
    deg=2,
)

# Alternative: Explicitly specify parser for standard format
# test_wiggler = FieldFitter(
#     file_path,
#     parser=StandardFieldMapParser(),  # Explicit parser specification
#     xy_point=(0, 0),
#     dx=0.001,
#     dy=0.001,
#     ds=0.001,
#     min_region_size=10,
#     deg=2,
# )

# Alternative: For UE36 format with custom Y positions
# test_wiggler = FieldFitter(
#     'example_data/UE36kn3_LH.dat',
#     parser=UE36FieldMapParser(y_positions=None, field_order='Bx,By,Bs', dx=0.001, dy=0.001),
#     xy_point=(0, 0),
#     dx=0.001,
#     dy=0.001,
#     ds=0.001,
#     min_region_size=10,
#     deg=2,
# )

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

field_calculator = FieldCalculator(df_fit_pars=df_fit_pars)

x = 0.001
y = 0.001

iter = 1000
steps = 13

s_vals = np.random.rand(steps)
x_vals = np.random.rand(steps)
y_vals = np.random.rand(steps)

start_time = time.time()
#with cProfile.Profile() as prof:
Bx_p, By_p, Bz_p = field_calculator.get_Bfield(x_arr=x_vals, y_arr=y_vals, s_arr=s_vals, python=False)
#print(prof.print_stats())
end_time = time.time()
print(f"Time taken to evaluate B field on array using C code: {end_time - start_time} seconds")
print(f"Time per evaluation using C code on array: {(end_time - start_time)/(steps)} seconds")
#field_calculator.plot_B_field(x_arr=x, y_arr=y, python=False)
#field_calculator.plot_A_field(x_arr=x, y_arr=y, python=False)

field_calculator.set_integrator()
wig_line = field_calculator.get_line()

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, energy0=2.7e9)

#field_calculator.correctors(p0)

wig_line.particle_ref = p0.copy()

start_time = time.time()
tw_wig = wig_line.twiss4d(include_collective=True, betx=1, bety=1)
end_time = time.time()
print(f"Numer of steps in wiggler: {len(field_calculator.integrator)}")
print(f"Time taken to compute twiss: {end_time - start_time} seconds")

tw_wig.plot('x y')
tw_wig.plot('betx bety', 'dx dy')
plt.show()

# Additional examples (commented out)
# env = xt.load('example_data/b075_2024.09.25.madx')
# line = env.ring
# 
# wiggler_places = ['ars11_uind_0610_1']
# 
# tt = line.get_table()
# for wig_place in wiggler_places:
#     line.insert(wig_line, anchor='start', at=tt['s', wig_place])
# 
# env['on_wig_corr'] = 0
# mywig.scale = 0
# tw_no_wig = line.twiss4d(strengths=True)
# tw_vs_momentum_no_wig = {}
# for dd in deltas:
#     tw_vs_momentum_no_wig[dd] = line.twiss4d(delta0=dd,
#                                          compute_chromatic_properties=False)
# 
# tw = line.twiss4d(include_collective=True, particle_on_co=p_co,
#                   compute_chromatic_properties=False)