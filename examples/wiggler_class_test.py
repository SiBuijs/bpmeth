from wiggler_class import WigglerFieldFitter
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

test_wiggler = WigglerFieldFitter(
                                    file_path,
                                    xy_point=(0, 0),
                                    dx=0.001,
                                    dy=0.001,
                                    ds=0.001,
                                    peak_window=(100, 2100),
                                    n_modes=6,
                                    poly_pieces=[15, 15],
                                    deg=2,
                                    filter_params=None,
                            )

test_wiggler.set()
test_wiggler._set_derivative_df()
test_wiggler._set_df_fit_pars()
print(test_wiggler.df_on_axis_raw.head())
print(test_wiggler.df_fit_pars.head())
#test_wiggler.plot_fields()
prrrr
#print(test_wiggler.fit_pars[0]['Bx']['edge_L'])
#print(test_wiggler.fit_pars[0]['Bx']['sines'])

test_wigglerfull = WigglerFull(test_wiggler)


test_wigglerfull.plot_field(x0=0.001, y0=0.001, plot_data=True)

prrr
"""
#test_wigglerfull.plot_fields()
#test_wigglerfull.plot_vector_potential(x=0.001, y=0.001)

n_slices = 1000
n_steps  = 1000

start_time = time.time()
test_wigglerfull.set_integrator(n_steps=n_steps, n_slices=n_slices)
end_time = time.time()
print(f"Time to set the integrator: {end_time - start_time} seconds")


start_time = time.time()
wiggler_line = test_wigglerfull.get_line()
end_time = time.time()
print(f"Time to get the integrator: {end_time - start_time} seconds")

particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, energy0=2.7e9)
wiggler_line.particle_ref = particle_ref
test_wigglerfull.correctors(particle_ref)

#Corrector strengths [T]:
#  k0l_corr1 = -2.745802e-04, k0sl_corr1 = 7.976874e-04
#  k0l_corr2 = 7.079133e-04, k0sl_corr2 = -1.137973e-03
#  k0l_corr3 = 7.680491e-04, k0sl_corr3 = 5.204176e-04
#  k0l_corr4 = -6.717367e-04, k0sl_corr4 = -3.003631e-04
#Time to compute twiss:      0.6319551467895508 seconds

start_time = time.time()
tw = wiggler_line.twiss(include_collective=True, betx=1, bety=1)
end_time = time.time()
print(f"Time to compute twiss:      {end_time - start_time} seconds")

plt.plot(tw.x, tw.y)
tw.plot('x y')
tw.plot('betx bety', 'dx dy')
plt.show()
"""