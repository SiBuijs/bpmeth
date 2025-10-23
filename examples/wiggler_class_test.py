from wiggler_class_derivatives import WigglerFieldFitter
from wiggler_class_derivatives import WigglerSegment
from wiggler_class_derivatives import WigglerFull
#from wiggler_class import WigglerFull
import bpmeth as bp
import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
# TEST THE CLASS
########################################################################################################################

# Class has been successfully tested for the sinusoidal region in the middle.
# The procedure is as follows: We fit the field on-axis (0,0).
# We fit the transverse second derivative at (0,0).
# This is a function of s, which is in turn fitted to a sinusoid.
# We pass the resulting function parameters to bpmeth.
# bpmeth correctly reproduces the data off-axis.

dz = 0.001  # Step size in the z direction for numerical differentiation.

# Create a Wiggler with parameters:
# file_path: Path to the field map file.
# xy_point: (x,y) coordinates of the axis where the field is evaluated.
# dz: Step size in the z direction for numerical differentiation.
#       In this case, dz=0.001 rescales the distances to m instead of mm.
# x_left_slices: Number of slices to the left in the x direction for fitting.
# x_right_slices: Number of slices to the right in the x direction for fitting.
# y_left_slices: Number of slices to the left in the y direction for fitting.
# y_right_slices: Number of slices to the right in the y direction for fitting.
# n_modes_x: Number of modes in the x direction for fitting the sinusoid.
# n_modes_y: Number of modes in the y direction for fitting the sinusoid.
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
#test_wiggler.plot_fields(der=0)
#test_wiggler.plot_fields(der=1)
#test_wiggler.plot_fields(der=2)
#segments = test_wiggler.export_piecewise_segments("Bx")
# segments is a list of tuples (start, end, coefficients)
# Taking segment[n] gives the n+1th tuple
# Taking segment[n][0] gives the start of the n+1th segment
# Taking segment[n][1] gives the end of the n+1th segment
# Taking segment[n][2] gives the sympy expression of the n+1th segment
#print(segments)
#print(segments[0])
#print(segments[0][0])
#print(segments[0][1])
#print(segments[0][2])


test_wigglerfull = WigglerFull(test_wiggler)
import time
start_time = time.time()
test_wigglerfull.set_segments()
end_time = time.time()
print(f"Time to make the segments: {end_time - start_time} seconds")

test_wigglerfull.plot_fields(x=0.001, y=0.001)

# Code below prints all the segments.
# There are 123 segments in total, not sure how that arises, should be around 51.
# Also, it plots the field in segment 67 (sinusoidal) to compare with the original fit.
# It seems to work well.
"""
for i, segment in enumerate(test_wigglerfull.segments):
    print(f"Segment {i}: s0 = {segment.s0}, length = {segment.length}")

segment = test_wigglerfull.segments[67]
s0 = segment.s0
length = segment.length
s1 = s0 + length
s_array = np.linspace(s0, s1, 1000)
Bx, By, Bs = segment.get_field(0.001, 0.001, s_array)

test_wiggler.xy_point = (1, 1)
test_wiggler.select_xy()

plt.figure(figsize=(10, 4))
plt.plot(s_array, Bx, label='Bx')
plt.plot(test_wiggler.s_full, test_wiggler.raw_data[0]['Bx'])
plt.plot(s_array, By, label='By')
plt.plot(test_wiggler.s_full, test_wiggler.raw_data[0]['By'])
plt.plot(s_array, Bs, label='Bs')
plt.plot(test_wiggler.s_full, test_wiggler.raw_data[0]['Bs'])
plt.title(f'Fields in Segment starting at s0={s0:.3f} m')
plt.xlabel('s [m]')
plt.ylabel('Field [T]')
plt.legend()
plt.grid()
plt.show()
"""

