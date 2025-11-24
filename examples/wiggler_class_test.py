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
from collections import defaultdict

df_here = test_wiggler.df_fit_pars

# per (field_component, derivative_x, region_name) pick the row(s) with largest s_start <= s_value
s_value = 0.5

df = df_here.reset_index()  # make s_start a column
group_cols = ['field_component', 'derivative_x', 'region_name']

num_iter = 10000

# --- simple boolean mask (returns all rows where s_start <= s_value <= s_end) ---
s_start = df['s_start'].values
s_end = df['s_end'].values
s_val = s_value

start_time = time.time()
for _ in range(num_iter):
    mask = (s_start <= s_val) & (s_end >= s_val)
df_ok = df.loc[mask]
end_time = time.time()
print(f"Time taken for {num_iter} iterations: {end_time - start_time} seconds")
print(f"Average time per iteration: {(end_time - start_time)/num_iter} seconds")

print(df_ok)

param_dict = df_ok.set_index("param_name")["param_value"].to_dict()
print(param_dict)