import inspect

import bpmeth
import numpy as np
import sympy as sp

symbols = sp.symbols('x y s Ac As k')

s = symbols[2]
Ac = symbols[3]
As = symbols[4]
k = symbols[5]

curv = 0

b1 = sp.sympify(f"{Ac}*cos({k}*{s})")
a0 = sp.sympify(f"{As}*sin({k}*{s})")

# Changed GeneralVectorPotential to accept expressions with free parameters such as Ac, As, k.
wiggler = bpmeth.GeneralVectorPotential(hs=f"{curv}",a=(f"{a0}",),b=(f"{b1}",))

# Without lambdifying, we get sympy expressions for Bx, By, Bs
Bx, By, Bs = wiggler.get_Bfield(lambdify=False)
Ax, Ay, As = wiggler.get_A()

print("Symbolic expressions for B field components:")
print(Bx)
print(By)
print(Bs)

print("Symbolic expressions for A field components:")
print(Ax)
print(Ay)
print(As)

# Now we can substitute numerical values for the parameters
# Still symbolic expressions
params = {sp.Symbol("Ac"): 1.23, sp.Symbol("As"): -0.45, sp.Symbol("k"): -0.45}
Bx_num = Bx.subs(params)
By_num = By.subs(params)
Bs_num = Bs.subs(params)

Ax_num = Ax.subs(params)
Ay_num = Ay.subs(params)
As_num = As.subs(params)

print("Symbolic expressions for B field components with numerical parameters substituted:")
print(Bx_num)
print(By_num)
print(Bs_num)

print("Symbolic expressions for A field components with numerical parameters substituted:")
print(Ax_num)
print(Ay_num)
print(As_num)

# With the numerical values for the parameters substituted, we lambdify to get numerical functions
Bx_func = sp.lambdify((wiggler.x, wiggler.y, wiggler.s), Bx_num, "numpy")
By_func = sp.lambdify((wiggler.x, wiggler.y, wiggler.s), By_num, "numpy")
Bs_func = sp.lambdify((wiggler.x, wiggler.y, wiggler.s), Bs_num, "numpy")

Ax_func = sp.lambdify((wiggler.x, wiggler.y, wiggler.s), Ax_num, "numpy")
Ay_func = sp.lambdify((wiggler.x, wiggler.y, wiggler.s), Ay_num, "numpy")
As_func = sp.lambdify((wiggler.x, wiggler.y, wiggler.s), As_num, "numpy")

# We can print the functions themselves
print("Numerical functions for B field components:")
print(inspect.getsource(Bx_func))
print(inspect.getsource(By_func))
print(inspect.getsource(Bs_func))

print("Numerical functions for A field components:")
print(inspect.getsource(Ax_func))
print(inspect.getsource(Ay_func))
print(inspect.getsource(As_func))

# And evaluate them.
print("Evaluated B field components at (x=0, y=0, s=1.0):")
print(Bx_func(0, 0, 1.0))
print(By_func(0, 0, 1.0))
print(Bs_func(0, 0, 1.0))

print("Evaluated A field components at (x=0, y=0.001, s=1.0):")
print(Ax_func(0, 0.001, 1.0))
print(Ay_func(0, 0.001, 1.0))
print(As_func(0, 0.001, 1.0))