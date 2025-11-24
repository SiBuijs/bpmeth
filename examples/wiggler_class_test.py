from wiggler_class import FieldFitter
from wiggler_class import SymbolicGenerator
from wiggler_class import WigglerFull
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

symbolic_test = SymbolicGenerator(test_wiggler)
symbolic_test._set_symbolic_exprs()

print(symbolic_test.generic_exprs)

#for der in range(3):
#    test_wiggler.plot_fields(der=der)
#test_wiggler.plot_integrated_fields()