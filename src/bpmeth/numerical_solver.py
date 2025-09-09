import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from .generate_expansion import FieldExpansion
from .frames import Frame, BendFrame
import numba as nb


class Hamiltonian:
    isthick = True

    def __init__(self, length, curv, vectp, s_start=0):
        self.length = length
        self.s_start = s_start
        self.curv = curv
        self.vectp = vectp
        self.angle = curv * length

        self.vectorfield = self.get_vectorfield()

    def get_H(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        px, py, ptau = coords.px, coords.py, coords.ptau
        beta0 = coords.beta0
        h = self.curv
        A = self.vectp.get_A()
        A = [Ai.subs({self.vectp.x: x, self.vectp.y: y, self.vectp.s: s}) for Ai in A]

        sqrt = coords._m.sqrt
        tmp1 = sqrt(
            1 + 2 * ptau / beta0 + ptau ** 2 - (px - A[0]) ** 2 - (py - A[1]) ** 2
        )
        H = ptau / beta0 - (1 + h * x) * (tmp1 + A[2])

        return H

    def get_A(self, x, y, s):
        Ax, Ay, As = self.vectp.get_A(lambdify=True)
        return Ax(x, y, s), Ay(x, y, s), As(x, y, s)

    def get_vectorfield(self, coords=None, lambdify=True):
        if coords is None:
            beta0 = sp.symbols("beta0", real=True, positive=True)
            coords = SympyParticle(beta0=beta0)
        x, y, tau = coords.x, coords.y, coords.tau
        px, py, ptau = coords.px, coords.py, coords.ptau
        H = self.get_H(coords)
        fx = H.diff(px)
        fy = H.diff(py)
        ftau = H.diff(ptau)
        fpx = -H.diff(x)
        fpy = -H.diff(y)
        fptau = -H.diff(tau)
        qpdot = [fx, fy, ftau, fpx, fpy, fptau]

        if lambdify:
            qp = (x, y, tau, px, py, ptau)
            s = coords.s
            f = sp.lambdify((s, qp, beta0), qpdot, modules="numpy")

            f_numba = nb.njit(f)
            return f_numba
        return qpdot

    def solve(self, qp0, s_span=None, ivp_opt={}, backtrack=False, beta0=1):
        ivp_opt = ivp_opt.copy()
        if s_span is None and not backtrack:
            s_span = [self.s_start, self.length + self.s_start]
        if backtrack:
            if s_span is None:
                s_span = [self.length + self.s_start, self.s_start]
            assert s_span[0] > s_span[1], "s_span not compatible with backtracking"

        if "t_eval" not in ivp_opt:
            ivp_opt["t_eval"] = np.linspace(s_span[0], s_span[1], 500)
        if "rtol" not in ivp_opt:
            ivp_opt["rtol"] = 1e-5
        if "atol" not in ivp_opt:
            ivp_opt["atol"] = 1e-8

        f = self.vectorfield
        sol = solve_ivp(f.py_func, s_span, qp0, args=(beta0,), **ivp_opt)
        return sol

    def track(self, particle, s_span=None, return_sol=False, ivp_opt={}, backtrack=False):
        if s_span is None and not backtrack:
            s_span = [self.s_start, self.length + self.s_start]
        if backtrack:
            if s_span is None:
                s_span = [self.length + self.s_start, self.s_start]
            assert s_span[0] > s_span[1], "s_span not compatible with backtracking"

        if isinstance(particle.x, np.ndarray) or isinstance(particle.x, list):
            results = []
            out = []
            for i in range(len(particle.x)):
                # adjust vector potential for each particle
                ax, ay, _ = self.get_A(particle.x[i], particle.y[i], s_span[0])
                kin_px = particle.px[i] - particle.ax[i]
                kin_py = particle.py[i] - particle.ay[i]
                particle.px[i] = kin_px + ax
                particle.py[i] = kin_py + ay
                qp0 = [
                    particle.x[i],
                    particle.y[i],
                    particle.zeta[i] / particle.beta0[i],
                    particle.px[i],
                    particle.py[i],
                    particle.ptau[i],
                ]
                sol = self.solve(qp0, s_span=s_span, beta0=particle.beta0[i], ivp_opt=ivp_opt)
                s = sol.t
                x, y, tau, px, py, ptau = sol.y
                results.append(
                    {
                        "x": x[-1],
                        "y": y[-1],
                        "zeta": tau[-1] * particle.beta0[i],
                        "px": px[-1],
                        "py": py[-1],
                        "ptau": ptau[-1],
                    }
                )
                if return_sol:
                    out.append(sol)
            particle.x = [res["x"] for res in results]
            particle.y = [res["y"] for res in results]
            particle.zeta = [res["zeta"] for res in results]
            particle.px = [res["px"] for res in results]
            particle.py = [res["py"] for res in results]
            particle.ptau = [res["ptau"] for res in results]
            if backtrack:
                particle.s -= self.length
            else:
                particle.s += self.length
            # manage vector potential for each particle
            ax, ay, _ = self.get_A(particle.x, particle.y, s_span[0])
            # if we want to keep the vector potential in the particle
            particle.ax = ax
            particle.ay = ay
            # we could remove it, but the operation is not symplectic
            # particle.ax = np.zeros_like(ax)
            # particle.ay = np.zeros_like(ay)
            # particle.px -= ax
            # particle.py -= ay

        else:
            qp0 = [
                particle.x,
                particle.y,
                particle.zeta / particle.beta0,
                particle.px,
                particle.py,
                particle.ptau,
            ]
            sol = self.solve(qp0, s_span=s_span, beta0=particle.beta0, ivp_opt=ivp_opt)
            s = sol.t
            x, y, tau, px, py, ptau = sol.y
            particle.x = x[-1]
            particle.y = y[-1]
            particle.zeta = tau[-1] * particle.beta0
            particle.px = px[-1]
            particle.py = py[-1]
            particle.ptau = ptau[-1]
            if backtrack:
                particle.s -= self.length
            else:
                particle.s += self.length
            if return_sol:
                out = sol

        if return_sol:
            return out

    def plotsol(
            self,
            qp0,
            s_span=None,
            ivp_opt=None,
            figname_zx=None,
            figname_zxy=None,
            canvas_zx=None,
            canvas_zxy=None,
    ):
        sol = self.solve(qp0, s_span, ivp_opt)
        s = sol.t
        x, y, tau, px, py, ptau = sol.y

        fr = Frame()
        fb = BendFrame(fr, self.length, self.angle)

        fb.plot_trajectory_zx(s, x, y, figname=figname_zx, canvas=canvas_zx)
        # fb.plot_trajectory_zxy(s, x, y, figname=figname_zxy, canvas=canvas_zxy)

    def __repr__(self):
        return f"Hamiltonian({self.length}, {self.curv}, {self.vectp})"


###########################################
# A few common cases, to make life easier #
###########################################


class DriftVectorPotential(FieldExpansion):
    def __init__(self):
        super().__init__(nphi=0)

    def get_A(self):
        return [sp.S.Zero, sp.S.Zero, sp.S.Zero]


class DipoleVectorPotential(FieldExpansion):
    def __init__(self, curv, b1):
        """
        Dipoles without s-dependence, in this case the vector potential is known analytically.
        Can also be used to study dipoles without any field derivatives!

        :param curv (float): Curvature of the reference frame.
        :param b1 (float): Dipole field strength.
        """

        self.curv = curv
        self.b1 = b1
        super().__init__(b=(b1,), hs=f"{curv}", nphi=0)

    def get_A(self, lambdify=False):
        h = self.curv
        x, y, s = self.x, self.y, self.s
        print(self.b[0])
        As = -(x + h / 2 * x ** 2) / (1 + h * x) * self.b[0]
        print(As)
        if lambdify:
            return [
                np.vectorize(sp.lambdify([x, y, s], 0, "numpy")),
                np.vectorize(sp.lambdify([x, y, s], 0, "numpy")),
                np.vectorize(sp.lambdify([x, y, s], As, "numpy")),
            ]
        return sp.Integer(0), sp.Integer(0), As


class SolenoidVectorPotential(FieldExpansion):
    def __init__(self, bs):
        """
        Solenoids without s-dependence, in this case the vector potential is known analytically.
        Curvature of the reference frame is not implemented, use GeneralVectorPotential instead.

        :param bs (float): Solenoid field strength.
        """

        self.bs = bs
        super().__init__(bs=f"{bs}", nphi=0)

    def get_A(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        bs = self.bs
        return [-bs * y / 2, bs * x / 2, 0]


class FringeVectorPotential(FieldExpansion):
    def __init__(self, b1, nphi=5):
        """
        Fringe fields in a straight coordinate frame.

        :param b1 (str): Fringe field b1 as a function of s.
        :param nphi (int): Number of terms in the expansion of the scalar potential.
        """

        super().__init__(b=(b1,), nphi=nphi)


class GeneralVectorPotential(FieldExpansion):
    def __init__(self, a=("0*s",), b=("0*s",), bs="0", hs="0", nphi=5):
        """
        General field expansion.

        :param a (tuple of str): a coefficients as a function of s.
        :param b (tuple of str): b coefficients as a function of s.
        :param bs (str): bs coefficient as a function of s.
        :param hs (str): Curvature of the reference frame as a function of s. (to be tested for not constant hs)
        :param nphi (int): Number of terms in the expansion of the scalar potential.
        """

        super().__init__(a=a, b=b, bs=bs, hs=hs, nphi=nphi)


class SympyParticle:
    def __init__(self, beta0=1):
        self.x, self.y, self.tau = sp.symbols("x y tau")
        self.zeta = sp.symbols("zeta")  ## to be checked
        self.s = sp.symbols("s")
        self.px, self.py, self.ptau = sp.symbols("px py ptau")
        self.beta0 = beta0
        self._m = sp
        self.npart = 1


class NumpyParticle:
    def __init__(self, qp0, s=0, beta0=1):
        self.beta0 = beta0
        self.x, self.y, self.tau, self.px, self.py, self.ptau = qp0
        self.zeta = self.tau * self.beta0
        self.s = s
        self._m = np
        self.npart = 1


class MultiParticle:
    def __init__(self, npart, x=0, y=0, tau=0, px=0, py=0, ptau=0, s=0, beta0=1):
        if isinstance(x, int) or isinstance(x, float):
            x = np.full(npart, x)
        if isinstance(y, int) or isinstance(y, float):
            y = np.full(npart, y)
        if isinstance(tau, int) or isinstance(tau, float):
            tau = np.full(npart, tau)
        if isinstance(px, int) or isinstance(px, float):
            px = np.full(npart, px)
        if isinstance(py, int) or isinstance(py, float):
            py = np.full(npart, py)
        if isinstance(ptau, int) or isinstance(ptau, float):
            ptau = np.full(npart, ptau)
        if isinstance(s, int) or isinstance(s, float):
            s = np.full(npart, s)

        assert len(x) == npart, "Invalid x"
        assert len(y) == npart, "Invalid y"
        assert len(tau) == npart, "Invalid tau"
        assert len(px) == npart, "Invalid px"
        assert len(py) == npart, "Invalid py"
        assert len(ptau) == npart, "Invalid ptau"
        assert len(s) == npart, "Invalid s"
        assert isinstance(beta0, int) or isinstance(beta0, float), "Invalid beta0"

        self.beta0 = beta0
        self.npart = npart

        self.x = np.array(x)
        self.y = np.array(y)
        self.tau = np.array(tau)
        self.px = np.array(px)
        self.py = np.array(py)
        self.ptau = np.array(ptau)
        self.zeta = self.tau * self.beta0
        self.s = np.array(s)

        self._m = np

    def copy(self):
        return MultiParticle(self.npart, x=self.x.copy(), y=self.y.copy(), tau=self.tau.copy(),
                             px=self.px.copy(), py=self.py.copy(), ptau=self.ptau.copy(), s=self.s.copy())
