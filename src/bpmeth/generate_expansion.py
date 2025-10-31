import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import warnings


def phinplus2(phi, x, s, hs):
    """
    Recursion relation for the magnetic scalar potential
    """

    xx = ((1 + hs * x) * phi.diff(x)).diff(x)
    ss = (1 / (1 + hs * x) * phi.diff(s)).diff(s)
    return -1 / (1 + hs * x) * (xx + ss)


class FieldExpansion:
    def __init__(self, a=(), b=(), bs="0", hs="0", nphi=5):
        self.x, self.y, self.s = sp.symbols("x y s", real=True)

        self.a = tuple(sp.sympify(aa, locals={"s": self.s}) for aa in a)
        self.b = tuple(sp.sympify(bb, locals={"s": self.s}) for bb in b)
        self.bs = sp.sympify(bs, locals={"s": self.s})
        self.hs = sp.sympify(hs, locals={"s": self.s})

        self.nphi = nphi
        self._B_expr = None
        self._B_funcs = None

        self.afun = [sp.Function(f"a{i + 1}")(self.s) for i in range(len(a))]
        self.bfun = [sp.Function(f"b{i + 1}")(self.s) for i in range(len(b))]
        self.bsfun = sp.Function("bs")(self.s) if self.bs != 0 else sp.Integer(0)

    @staticmethod
    def _pw_diff(expr, s, n):
        import sympy as sp
        if n == 0:
            return expr
        if isinstance(expr, sp.Piecewise):
            parts = [(sp.diff(e, s, n), c) for (e, c) in expr.args]
            return sp.Piecewise(*parts, evaluate=False)
        return sp.diff(expr, s, n)

    def subs(self, func, coord_subs=None):
        import sympy as sp
        if coord_subs is None:
            coord_subs = {}
        s = self.s

        # 1) Build the base mapping (sym -> expr), sympifying all values
        base_map = {}
        for sym, val in zip(self.afun, self.a):
            base_map[sym.subs(coord_subs)] = sp.sympify(val).subs(coord_subs)
        for sym, val in zip(self.bfun, self.b):
            base_map[sym.subs(coord_subs)] = sp.sympify(val).subs(coord_subs)
        # Always map bs(s) as well (sympify handles int/float/expr uniformly)
        base_map[self.bsfun.subs(coord_subs)] = sp.sympify(self.bs).subs(coord_subs)

        # 2) Start with 0th-order replacements
        mapping = dict(base_map)

        # 3) Add derivative replacements that actually appear in func
        for d in func.atoms(sp.Derivative):
            vars_ = d.variables
            if not vars_:
                continue
            if all(v == s for v in vars_):
                base = d.expr.subs(coord_subs)
                if base in base_map:
                    order = len(vars_)
                    mapping[d] = self._pw_diff(base_map[base], s, order)

        # 4) Structural replacement, no .doit() (avoids Piecewise blow-up)
        return func.xreplace(mapping)

    def get_phi(self, tolerance=1e-4, apperture=0.05, subs=True):
        """
        a=(a1,a2,a3,...)
        b=(b1,b2,b3,...)
        """

        x, y, s = self.x, self.y, self.s
        a = self.afun
        b = self.bfun
        bs = self.bsfun
        hs = self.hs
        nphi = self.nphi

        phi0 = sum((an * x ** (n + 1) / sp.factorial(n + 1) for n, an in enumerate(a))) + sp.integrate(bs, s)
        phi1 = sum((bn * x**n / sp.factorial(n) for n, bn in enumerate(b)))
        phiv = [phi0, phi1]
        for i in range(nphi-2):
            phiv.append(phinplus2(phiv[i], x, s, hs))  # no simplify
        phi = sp.expand(sum(pp * y ** i / sp.factorial(i) for i, pp in enumerate(phiv)))

        if subs:
            """
            Truncating the expansion in phi will violate Maxwell equations. Verify if the 
            deviation is not too large by checking if the laplacian of the potential remains 
            within the allowed tolerances.
            """

            phi =  self.subs(phi)

            lapl = (
                1/(1 + hs*x) * (1/(1 + hs*x) * phi.diff(s)).diff(s) +
                1/(1 + hs*x) * ((1 + hs*x) * phi.diff(x)).diff(x) +
                phi.diff(y, 2)
            )
            laplval = lapl.subs({x: apperture, y: apperture, s: apperture}).evalf()
            if isinstance(laplval, float) or isinstance(laplval, sp.core.numbers.Float):
                if laplval < tolerance:
                    warnings.warn(f"More terms are needed in the expansion for aperture {apperture}m, the laplacian is {laplval}")
            else:
                warnings.warn("The laplacian could not be evaluated, are you doing symbolic calculations?")

        return phi

    def get_Bfield(self, lambdify=True, subs=True):
        x, y, s = self.x, self.y, self.s
        hs = self.hs

        # Build once per instance
        if getattr(self, "_B_expr", None) is None:
            phi = self.get_phi(subs=False)  # keep coefficients symbolic here
            Bx = phi.diff(x)
            By = phi.diff(y)
            Bs = 1 / (1 + hs * x) * phi.diff(s)

            if subs or lambdify:
                Bx = self.subs(Bx)  # now cheap; no .doit()
                By = self.subs(By)
                Bs = self.subs(Bs)

            self._B_expr = (Bx, By, Bs)

        if not lambdify:
            return self._B_expr

        if getattr(self, "_B_funcs", None) is None:
            Bx, By, Bs = self._B_expr
            Bxf = sp.lambdify((x, y, s), Bx, "numpy")
            Byf = sp.lambdify((x, y, s), By, "numpy")
            Bsf = sp.lambdify((x, y, s), Bs, "numpy")
            self._B_funcs = (Bxf, Byf, Bsf)

        return self._B_funcs

    def get_A(self, lambdify=False, subs=True):
        x, y, s = self.x, self.y, self.s
        hs = self.hs
        Bx, By, Bs = self.get_Bfield(lambdify=False, subs=False)

        Ax = - sp.integrate(Bs, (y, 0, y))
        Ay = sp.S(0)
        As = sp.integrate(Bx, (y, 0, y)) - 1/(1+hs*x) * sp.integrate((1+hs*x)*By.subs({y:0}), (x, 0, x))

        if subs:
            Ax = self.subs(Ax)
            Ay = self.subs(Ay)
            As = self.subs(As)

        if lambdify:
            return (
                np.vectorize(sp.lambdify([x, y, s], Ax, "numpy")),
                np.vectorize(sp.lambdify([x, y, s], Ay, "numpy")),
                np.vectorize(sp.lambdify([x, y, s], As, "numpy")),
            )
        
        return Ax, Ay, As


    #########################################
    # TO BE OPTIMISED FROM HERE             #
    #########################################

    
    def transform(self, theta_E=0, rho=np.inf, maxpow=None):
        """
        So far only implemented for the entrance of the element. Some signs will differ for the exit!
        
        param theta_E (float): Rotation angle in radians.
        param rho (float): Bending radius of the frame. If rho=np.inf, the frame is straight.
        param maxpow (int): Maximum power of the series expansion to keep.
        return (FieldExpansion): New field expansion in the rotated frame.
        """
        
        assert self.hs==0, "Transform only implemented starting from straight frame!"        
        
        x, y, s = self.x, self.y, self.s
        at = self.a
        bt = self.b
        bst = self.bs
        nphi = self.nphi
        if maxpow is None:
            maxpow = nphi
        
        # Determine the scalar potential 0 and 1 terms and transform to new frame
        phi0 = sum((an * x ** (n + 1) / sp.factorial(n + 1) for n, an in enumerate(at))) + sp.integrate(bst, s)
        phi1 = sum((bn * x**n / sp.factorial(n) for n, bn in enumerate(bt))) 

        ss, xx = sp.symbols("ss xx")  # Needed for simultaneous substitution
        
        if rho==np.inf:  # Straight frame
            xt = ss*sp.sin(theta_E) + xx*sp.cos(theta_E)
            st = ss*sp.cos(theta_E) - xx*sp.sin(theta_E)
            print(f"Rotating over angle {theta_E} to new straight frame...")

        else:  # Curved frame
            xt = (rho+xx)*(sp.cos(theta_E-ss/rho)) - rho*sp.cos(theta_E)
            st = rho*sp.sin(theta_E) - (rho+xx)*(sp.sin(theta_E-ss/rho))
            print(f"Rotating over angle {theta_E} to new curved frame with bending radius {rho}...")

        phi0 = phi0.subs({x:xt, s:st}).subs({xx:x, ss:s})
        phi1 = phi1.subs({x:xt, s:st}).subs({xx:x, ss:s})

        phi0 = phi0.series(x, 0, maxpow).removeO().as_poly(x)
        phi1 = phi1.series(x, 0, maxpow).removeO().as_poly(x)
        
        # Extract the multipole coefficients.
        a = []
        degree0 = 0 if phi0==0 else phi0.degree(x)
        for i in range(degree0+1):
            a.append(phi0.coeff_monomial(x**(i+1)) * sp.factorial(i+1))
            
        bs = phi0.coeff_monomial(x**0).diff(s)

        b = []
        degree1 = 0 if phi1==0 else phi1.degree(x)
        for i in range(degree1+1):
            b.append(phi1.coeff_monomial(x**i) * sp.factorial(i))
            
        return FieldExpansion(a=a, b=b, bs=bs, nphi=nphi)
            
            
    def cut_at_angle(self, theta_E, maxpow=None):
        """
        Cut a general magnet at an angle theta_E. theta_E = 0 means no edge angle.
        
        :param theta_E (float): Edge angle in radians.
        """
        
        x, y, s = self.x, self.y, self.s
        at = self.a
        bt = self.b
        bst = self.bs
        nphi = self.nphi
        if maxpow is None:
            maxpow = nphi
            
        phi0 = sum((an * x ** (n + 1) / sp.factorial(n + 1) for n, an in enumerate(at))) + sp.integrate(bst, s)
        phi1 = sum((bn * x**n / sp.factorial(n) for n, bn in enumerate(bt))) 

        st = s - x*sp.tan(theta_E)
        print(f"Cutting magnet at angle {theta_E}...")
        
        phi0 = phi0.subs([(s,st)])
        phi1 = phi1.subs([(s,st)])

        phi0 = phi0.series(x, 0, maxpow).removeO().as_poly(x)
        phi1 = phi1.series(x, 0, maxpow).removeO().as_poly(x)        
        
        # Extract the multipole coefficients.
        a = []
        degree0 = 0 if phi0==0 else phi0.degree(x)
        for i in range(degree0+1):
            a.append(phi0.coeff_monomial(x**(i+1)) * sp.factorial(i+1))
            
        bs = phi0.coeff_monomial(x**0).diff(s)

        b = []
        degree1 = 0 if phi1==0 else phi1.degree(x)
        for i in range(degree1+1):
            b.append(phi1.coeff_monomial(x**i) * sp.factorial(i))
            
        return FieldExpansion(a=a, b=b, bs=bs, nphi=nphi)
    
    def plot_components(self, ax=None, smin=-2, smax=2, ns=100, plot_b=True, plot_a=True, plot_bs=True):
        ss = np.linspace(smin, smax, ns)
        if ax is None:
            fig, ax = plt.subplots()
        
        if plot_a:
            for i, aa in enumerate(self.a):
                ax.plot(ss, [sp.sympify(aa).subs(self.s, sval).evalf() for sval in ss], label=f"a_{i+1}")
        if plot_b:
            for i, bb in enumerate(self.b):
                ax.plot(ss, [sp.sympify(bb).subs(self.s, sval).evalf() for sval in ss], label=f"b_{i+1}")
        if plot_bs:
            ax.plot(ss, [sp.sympify(self.bs).subs(self.s, sval).evalf() for sval in ss], label=f"bs")
        ax.legend()
    
    def plotfield_z(self, X=0, Y=0, ax=None, zmin=-2, zmax=2, zstep=0.01):
        if ax is None:
            fig, ax = plt.subplots()
            
        Bxfun, Byfun, Bsfun = self.get_Bfield()
        Z = np.arange(zmin, zmax, zstep)
        
        ax.plot(Bxfun(X, Y, Z), label="Bx")
        ax.plot(Byfun(X, Y, Z), label="By")
        ax.plot(Bsfun(X, Y, Z), label="Bs")
        plt.legend()
        

    def plotfield_yz(self, X=0, ax=None, bmin=None, bmax=None, includebx=False, ymin=-2, ymax=2, ystep=0.05, zmin=-3, zmax=3, zstep=0.05):
        Y = np.arange(ymin, ymax, ystep)
        Z = np.arange(zmin, zmax, zstep)
        Z, Y = np.meshgrid(Z, Y)

        Bxfun, Byfun, Bsfun = self.get_Bfield()

        By = Byfun(X, Y, Z)
        Bs = Bsfun(X, Y, Z)

        bmagn = np.sqrt(By**2 + Bs**2)
        if includebx:
            Bx = Bxfun(X, Y, Z)
            bmagn_color = np.sqrt(bmagn**2 + Bx**2)
        else:
            bmagn_color = bmagn
            
        if bmin is None:
            bmin = bmagn_color.min()
        if bmax is None:
            bmax = bmagn_color.max()

        if ax is None:
            fig, ax = plt.subplots()
        plt.imshow(bmagn_color, extent=(zmin, zmax, ymin, ymax), origin='lower', vmin=bmin, vmax=bmax, aspect='auto')
        plt.colorbar()
        skip = 4
        plt.quiver(Z[::skip, ::skip], Y[::skip, ::skip], Bs[::skip, ::skip]/bmagn[::skip, ::skip], 
                By[::skip, ::skip]/bmagn[::skip, ::skip], scale=50, pivot='mid') 
        
        plt.xlabel("s")
        plt.ylabel("y")

        plt.show()
        
    def plotfield_xz(self, Y=0, ax=None, bmin=None, bmax=None, includeby=False, xmin=-2, xmax=2, xstep=0.05, zmin=-3, zmax=3, zstep=0.05):
        X = np.arange(xmin, xmax, xstep)
        Z = np.arange(zmin, zmax, zstep)
        Z, X = np.meshgrid(Z, X)

        Bxfun, Byfun, Bsfun = self.get_Bfield()

        Bx = Bxfun(X, Y, Z)
        Bs = Bsfun(X, Y, Z)


        bmagn = np.sqrt(Bx**2 + Bs**2)
        if includeby:
            By = Byfun(X, Y, Z)
            bmagn_color = np.sqrt(bmagn**2 + By**2)
        else:
            bmagn_color = bmagn
            
        if bmin is None:
            bmin = bmagn_color.min()
        if bmax is None:
            bmax = bmagn_color.max()

        if ax is None:
            fig, ax = plt.subplots()
        plt.imshow(bmagn_color, extent=(zmin, zmax, xmin, xmax), origin='lower', vmin=bmin, vmax=bmax, aspect='auto')
        plt.colorbar()
        skip = 4
        plt.quiver(Z[::skip, ::skip], X[::skip, ::skip], Bs[::skip, ::skip]/bmagn[::skip, ::skip], 
                Bx[::skip, ::skip]/bmagn[::skip, ::skip], scale=50, pivot='mid')
        
        plt.xlabel("s")
        plt.ylabel("x")

        plt.show()
        
    def plot_By(self, Y=0, ax=None, bmin=None, bmax=None, includeby=False, xmin=-2, xmax=2, xstep=0.05, zmin=-3, zmax=3, zstep=0.05):
        X = np.arange(xmin, xmax, xstep)
        Z = np.arange(zmin, zmax, zstep)
        Z, X = np.meshgrid(Z, X)

        Bxfun, Byfun, Bsfun = self.get_Bfield()
        By = Byfun(X, Y, Z)
            
        if bmin is None:
            bmin = By.min()
        if bmax is None:
            bmax = By.max()

        if ax is None:
            fig, ax = plt.subplots()
        plt.imshow(By, extent=(zmin, zmax, xmin, xmax), origin='lower', vmin=bmin, vmax=bmax, aspect='auto')
        plt.colorbar()
        
        plt.xlabel("s")
        plt.ylabel("x")

        plt.show()
                
                
    def calc_RDTs(self, n, betx=1, bety=1, alphx=0, alphy=0):
        """
        Calculate the RDTs (the h_pqrt values) of the given element symbolically as a function of s
        
        :param n (int): Highest order of the RDTs to calculate.
        :param betx (float or sp.symbol): Beta function in x.
        :param bety (float or sp.symbol): Beta function in y.
        :param alphx (float or sp.symbol): Alpha function in x.
        :param alphy (float or sp.symbol): Alpha function in y.
        :return (List of lists of sp expressions): Symbolic expression of the h values h_pqrt 
        to calculate the RDTs as a function of s.
        """

        x, y, s = self.x, self.y, self.s
        hs = self.hs
        Ax, Ay, As = self.get_A()
        
        hxp, hxm, hyp, hym = sp.symbols("hxp hxm hyp hym")
        xsubs = sp.sqrt(betx)/2*(hxp + hxm)
        ysubs = sp.sqrt(bety)/2*(hyp + hym)
        pxsubs = - 1/(2*sp.sqrt(betx))*((alphx + 1j)*hxp + (alphx - 1j)*hxm)
        pysubs = - 1/(2*sp.sqrt(bety))*((alphy + 1j)*hyp + (alphy - 1j)*hym)
        
        # H = -(1+hx) As
        H_As = -((1+hs*x) * As).series(x,0,n).removeO().subs([(x, xsubs), (y, ysubs)])
        H_As_poly = H_As.as_poly(hxp, hxm, hyp, hym)
        
        # H = -(1+hx) px Ax
        H_pxAx = -((1+hs*x) * pxsubs * Ax).cancel().series(x,0,n).removeO().subs([(x, xsubs), (y, ysubs)]).expand()
        H_pxAx_poly = H_pxAx.as_poly(hxp, hxm, hyp, hym)
        
        # H = (1+hx) 1/2 Ax^2
        H_Ax2 = ((1+hs*x) * 1/2 * Ax**2).cancel().series(x,0,n).removeO().subs([(x, xsubs), (y, ysubs)]).expand()
        H_Ax2_poly = H_Ax2.as_poly(hxp, hxm, hyp, hym)

        # H = -(1+hx) py Ay
        H_pyAy = -((1+hs*x) * pysubs * Ay).cancel().series(x,0,n).removeO().subs([(x, xsubs), (y, ysubs)]).expand()
        H_pyAy_poly = H_pyAy.as_poly(hxp, hxm, hyp, hym)
        
        # H = (1+hx) 1/2 Ay^2
        H_Ay2 = ((1+hs*x) * 1/2 * Ay**2).cancel().series(x,0,n).removeO().subs([(x, xsubs), (y, ysubs)]).expand()
        H_Ay2_poly = H_Ay2.as_poly(hxp, hxm, hyp, hym)
        
        h = sp.MutableDenseNDimArray(np.zeros((n+1, n+1, n+1, n+1), dtype=object))

        for p in range(n+1):
            for q in range(n+1-p):
                for r in range(n+1-p-q):
                    for t in range(n+1-p-q-r):
                        h[p,q,r,t] += H_As_poly.coeff_monomial(hxp**p*hxm**q*hyp**r*hym**t)
                        h[p,q,r,t] += H_pxAx_poly.coeff_monomial(hxp**p*hxm**q*hyp**r*hym**t)
                        h[p,q,r,t] += H_Ax2_poly.coeff_monomial(hxp**p*hxm**q*hyp**r*hym**t)
                        h[p,q,r,t] += H_pyAy_poly.coeff_monomial(hxp**p*hxm**q*hyp**r*hym**t)
                        h[p,q,r,t] += H_Ay2_poly.coeff_monomial(hxp**p*hxm**q*hyp**r*hym**t)

        return h
        
        

    def writefile(self, xarr, yarr, zarr, filename):
        X, Y, Z = np.meshgrid(xarr, yarr, zarr)

        Bxfun, Byfun, Bsfun = self.get_Bfield()

        Bx = Bxfun(X, Y, Z)
        By = Byfun(X, Y, Z)
        Bs = Bsfun(X, Y, Z)

        with open(f'{filename}.csv', 'w') as file:
            file.write(f'"X","Y","Z","Bx","By","Bz"\n')
            np.savetxt(file, np.c_[X.flat, Y.flat, Z.flat, Bx.flat, By.flat, Bs.flat], delimiter=',', fmt='%.10f')


    def __call__(self, x, y, z):
        # Return the magnetic field at the point (x, y, z)
        Bx, By, Bs = self.get_Bfield(lambdify=False)
        return (
            Bx.subs({self.x: x, self.y: y, self.s: z}).evalf(),
            By.subs({self.x: x, self.y: y, self.s: z}).evalf(),
            Bs.subs({self.x: x, self.y: y, self.s: z}).evalf(),
        )



