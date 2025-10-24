from wiggler_class_derivatives import WigglerFieldFitter
from wiggler_class_derivatives import WigglerFull
import time

########################################################################################################################
# TEST THE CLASS
########################################################################################################################

dz = 0.001  # Step size in the z direction for numerical differentiation.

print("FIELDS:")
test_wiggler = WigglerFieldFitter(file_path='example_data/knot_map_test.txt',
                                  xy_point=(0, 0),
                                  dx=dz,
                                  dy=dz,
                                  ds=dz,
                                  peak_window=(99, 2100),
                                  n_modes=[6, 6, 3],
                                  poly_deg=[[4, 4], [4, 4], [4, 4]],
                                  poly_pieces=[[25, 25], [25, 25], [25, 25]],
                                  deg=2
                                  )
test_wiggler.set()

test_wigglerfull = WigglerFull(test_wiggler)
start_time = time.time()
test_wigglerfull.set_segments()
end_time = time.time()
print(f"Time to make the segments: {end_time - start_time} seconds")

test_wigglerfull.plot_fields(x=0.001, y=0.001)