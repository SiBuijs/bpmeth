from wiggler_class import Wiggler

########################################################################################################################
# TEST THE CLASS
########################################################################################################################

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
Test_Wiggler = Wiggler(file_path='example_data/knot_map_test.txt',
                       xy_point=(0, 0),
                       dz=0.001,
                       x_left_slices=23,
                       x_right_slices=21,
                       y_left_slices=20,
                       y_right_slices=30,
                       n_modes_x=3,
                       n_modes_y=3)

# The Wiggler.fit() method fits the field with sinusoids in the middle and polynomials on the edges.
# The Wiggler.tune_slices_for_zero_integral() method tunes the number of slices for the polynomials
# to achieve zero integral of the field.
Test_Wiggler.fit()
Test_Wiggler.tune_slices_for_zero_integral(field="both", tradeoff_mse=0.0,
                                           left_candidates=range(8, 24),
                                           right_candidates=range(8, 24),
                                           verbose=True)

# The Wiggler.plot_fields() method plots the original field and the fitted field.
# The Wiggler.plot_integral() method plots the integral of the field.
Test_Wiggler.plot_fields()
Test_Wiggler.plot_integral()

'''
# We change the axis at which we evaluate the field.
Test_Wiggler.xy_point = (-1, 1)

# It turns out that the fit is not very good with 3 modes, so we increase the number of modes to 5.
Test_Wiggler.n_modes_x = 5
Test_Wiggler.n_modes_y = 5

# We fit and plot again.
Test_Wiggler.fit()
Test_Wiggler.tune_slices_for_zero_integral(field="both", tradeoff_mse=1e-6,
                                           left_candidates=range(8, 40),
                                           right_candidates=range(8, 40),
                                           verbose=True)

Test_Wiggler.plot_fields()
Test_Wiggler.plot_integral()
'''