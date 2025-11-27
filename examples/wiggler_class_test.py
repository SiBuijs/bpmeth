from wiggler_class import FieldFitter, FieldCalculator
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
from collections import defaultdict

symbolic_wiggler = SymbolicGenerator(test_wiggler)

prrrr

#print(f"Symbolic Ax : {symbolic_wiggler.symbolic_Ax}")
#print(f"Symbolic Ay : {symbolic_wiggler.symbolic_Ay}")
#print(f"Symbolic As : {symbolic_wiggler.symbolic_As}")
#print(f"Symbolic Bx : {symbolic_wiggler.symbolic_Bx}")
#print(f"Symbolic By : {symbolic_wiggler.symbolic_By}")
#print(f"Symbolic Bs : {symbolic_wiggler.symbolic_Bs}")

field_calculator = FieldCalculator(symbolic_wiggler, test_wiggler.df_fit_pars)

field_calculator.plot_B_field(x=0.001, y=0.000)

print(inspect.getsource(field_calculator.Bx_region_funcs[10]))

iter = 1000
s_vals = np.linspace(0.1, 2.1, iter)
for i in range(iter):
    B_field = field_calculator.get_Bfield(x=0.001, y=0.001, s=s_vals[i])