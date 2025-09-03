import bpmeth
import numpy as np
import matplotlib.pyplot as plt

p_sp = bpmeth.SympyParticle()

qp0 = [0,0,0,0,0,0]  # Initial conditions x, y, tau, px, py, ptau
p_np = bpmeth.NumpyParticle(qp0)

curv=0  # Curvature of the reference frame
length=10
k=2
amp=0.3
b1 = f"{amp}*cos({k}*s)"
a0 = f"{-amp}*sin({k}*s)"

wiggler = bpmeth.GeneralVectorPotential(hs=f"{curv}",a=(f"{a0}",),b=(f"{b1}",))
# wiggler = bpmeth.FringeVectorPotential(hs=f"{curv}", b1=b1) # Does the same as above
wiggler.plotfield_yz()
wiggler.plotfield_xz()

#A_wiggler = wiggler.get_Aval(p_sp)
H_wiggler = bpmeth.Hamiltonian(length, curv, wiggler)

ivp_opt={"rtol":1e-4, "atol":1e-7}
sol_wiggler = H_wiggler.solve(qp0, ivp_opt=ivp_opt)
H_wiggler.plotsol(qp0, ivp_opt=ivp_opt)

import xtrack as xt

p = xt.Particles(x = np.linspace(-1e-3, 1e-3, 10), energy0=10e9, mass0=xt.ELECTRON_MASS_EV)
sol = H_wiggler.track(p, return_sol=True)
