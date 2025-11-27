from wiggler_class import FieldFitter
from wiggler_class import SymbolicGenerator
#from wiggler_class import FieldCalculator
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

symbolic_wiggler = SymbolicGenerator(test_wiggler)

symbolic_wiggler.write_to_python(field='B')

import importlib
import B_field_eval
importlib.reload(B_field_eval)

param_dict = test_wiggler.df_fit_pars[['param_name', 'param_value']].set_index('param_name').to_dict()['param_value']

print(param_dict)

iter = 100000
start_time = time.time()
for _ in range(iter):
    result = B_field_eval.evaluate_B(0.001, 0, -1.1, **param_dict)
end_time = time.time()
print(f"Time for {iter} iterations: {end_time - start_time} seconds")
print(f"Time per iteration: {(end_time - start_time)/iter} seconds")
print(result)