# Auto-generated symbolic field expressions for A
from operator import itemgetter
def evaluate_A(x, y, s, **params):
    # Parameter assignments
    bs_0, bs_1, bs_2, bs_3, bs_4, a_1_0, a_1_1, a_1_2, a_1_3, a_1_4, a_2_0, a_2_1, a_2_2, a_2_3, a_2_4, a_3_0, a_3_1, a_3_2, a_3_3, a_3_4, b_1_0, b_1_1, b_1_2, b_1_3, b_1_4, b_2_0, b_2_1, b_2_2, b_2_3, b_2_4, b_3_0, b_3_1, b_3_2, b_3_3, b_3_4 = itemgetter('bs_0', 'bs_1', 'bs_2', 'bs_3', 'bs_4', 'a_1_0', 'a_1_1', 'a_1_2', 'a_1_3', 'a_1_4', 'a_2_0', 'a_2_1', 'a_2_2', 'a_2_3', 'a_2_4', 'a_3_0', 'a_3_1', 'a_3_2', 'a_3_3', 'a_3_4', 'b_1_0', 'b_1_1', 'b_1_2', 'b_1_3', 'b_1_4', 'b_2_0', 'b_2_1', 'b_2_2', 'b_2_3', 'b_2_4', 'b_3_0', 'b_3_1', 'b_3_2', 'b_3_3', 'b_3_4')(params)

    # Common sub-expressions
    x0 = y**2
    x1 = b_2_1/2
    x2 = s*x
    x3 = s**2
    x4 = 3*x3/2
    x5 = s**3
    x6 = 2*x5
    x7 = x**2
    x8 = x7/2
    x9 = s*x8
    x10 = s**4
    x11 = x**3
    x12 = x*x3
    x13 = x*x5
    x14 = x3/2
    x15 = b_2_0/2 + b_2_2*x14 + b_2_3*x5/2 + b_2_4*x10/2 + s*x1
    x16 = x*x10

    # Reduced expressions
    Ax = -x0*(b_1_1/2 + b_1_2*s + b_1_3*x4 + b_1_4*x6 + b_2_2*x2 + b_2_3*x*x4 + b_2_4*x*x6 + b_3_1*x7/4 + b_3_2*x9 + 3*b_3_3*x3*x7/4 + b_3_4*x5*x7 + x*x1) - y*(a_1_1*x + 2*a_1_2*x2 + 3*a_1_3*x12 + 4*a_1_4*x13 + a_2_1*x8 + a_2_2*s*x7 + a_2_3*x4*x7 + a_2_4*x6*x7 + a_3_1*x11/6 + a_3_2*s*x11/3 + a_3_3*x11*x14 + 2*a_3_4*x11*x5/3 + bs_0 + bs_1*s + bs_2*x3 + bs_3*x5 + bs_4*x10)
    Ay = 0
    As = -x*(b_1_0 + b_1_1*s + b_1_2*x3 + b_1_3*x5 + b_1_4*x10) + x0*(b_3_0*x/2 + b_3_1*x2/2 + b_3_2*x12/2 + b_3_3*x13/2 + b_3_4*x16/2 + x15) - x11*(b_3_0/6 + b_3_1*s/6 + b_3_2*x3/6 + b_3_3*x5/6 + b_3_4*x10/6) - x15*x7 + y*(a_1_0 + a_1_1*s + a_1_2*x3 + a_1_3*x5 + a_1_4*x10 + a_2_0*x + a_2_1*x2 + a_2_2*x12 + a_2_3*x13 + a_2_4*x16 + a_3_0*x8 + a_3_1*x9 + a_3_2*x3*x8 + a_3_3*x5*x8 + a_3_4*x10*x8)

    return Ax, Ay, As
