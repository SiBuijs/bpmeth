from wiggler_class import WigglerFieldFitter
from wiggler_class import WigglerFull
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

test_wiggler = WigglerFieldFitter(file_path=file_path,
                                  xy_point=(0, 0),
                                  dx=dz,
                                  dy=dz,
                                  ds=dz,
                                  peak_window=(99, 2100),
                                  n_modes=6,
                                  poly_pieces=[[15, 15], [15, 15], [10, 10]],
                                  deg=2
                                  )
test_wiggler.set()

#test_wiggler.plot_fields()
#test_wiggler.plot_integrated_fields()

test_wigglerfull = WigglerFull(test_wiggler)

start_time = time.time()
test_wigglerfull.set_segments()
end_time = time.time()
print(f"Time to make the segments:  {end_time - start_time} seconds")

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

start_time = time.time()
tw = wiggler_line.twiss(include_collective=True, betx=1, bety=1)
end_time = time.time()
print(f"Time to compute twiss:      {end_time - start_time} seconds")

plt.plot(tw.x, tw.y)
tw.plot('x y')
tw.plot('betx bety', 'dx dy')
plt.show()