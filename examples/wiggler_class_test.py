from wiggler_class import FieldFitter
from wiggler_class import SymbolicGenerator
from wiggler_class import FieldCalculator
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

test_wiggler.set()
df_fit_pars = test_wiggler.df_fit_pars

symbolic_wiggler = SymbolicGenerator(test_wiggler)

symbolic_wiggler.write_to_python(field='B')
symbolic_wiggler.write_to_python(field='A')
symbolic_wiggler.write_to_c(field='B')
symbolic_wiggler.write_to_c(field='A')

field_calculator = FieldCalculator(df_fit_pars)

x = 0.001
y = 0.001

#field_calculator.plot_B_field(x=x, y=y)
#field_calculator.plot_A_field(x=x, y=y)


iter = 100000

s_vals = np.linspace(-1, 1, iter)

start_time = time.time()
for s_val in s_vals:
    Bx, By, Bz = field_calculator.get_Bfield(x=x, y=y, s=s_val, Python=True)
end_time = time.time()

print(f"Time taken to evaluate B field on grid: {end_time - start_time} seconds")
print(f"Time per evaluation: {(end_time - start_time)/iter} seconds")

start_time = time.time()
symbolic_wiggler.compile_C_code(field='B')
end_time = time.time()
print(f"Time taken to compile C code for B field: {end_time - start_time} seconds")

start_time = time.time()
for s_val in s_vals:
    Bx_c, By_c, Bz_c = field_calculator.get_Bfield(x=x, y=y, s=s_val, Python=False)
end_time = time.time()
print(f"Time taken to evaluate B field on grid using C code: {end_time - start_time} seconds")
print(f"Time per evaluation using C code: {(end_time - start_time)/iter} seconds")