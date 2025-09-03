import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0

# --- geometry helpers ---
def r_array(lam_w, h_gap, n, z, sign_y):
    """
    r(z,n) as complex distances:
      Re = n*lam_w/4 - z (longitudinal)
      Im = +/- h_gap/2      (vertical)
    """
    z = np.asarray(z)
    n = np.asarray(n)
    r_real = (n * lam_w / 4.0)[None, :] - z[:, None]
    r_imag = sign_y * (h_gap / 2.0)
    return r_real + 1j * r_imag

def m_sequence(n, phi0, direction):
    """
    Dipole moments for a Halbach-like sequence.
    direction = +1  -> rotate +90° per pole:  m_n = e^{i phi0} * ( +i )**n
    direction = -1  -> rotate -90° per pole:  m_n = e^{i phi0} * ( -i )**n
    """
    step = 1j if direction > 0 else -1j
    return np.exp(1j * phi0) * (step ** n)

def B_from_dipoles(r, m_n):
    """
    Complex 2D dipole field:
      B = 3*Re(m*r)*r/|r|^5 - m/|r|^3
    r : (Nz, Nn), m_n : (Nn,)
    Returns B_total(z) with shape (Nz,)
    """
    m = m_n[None, :]
    abs_r = np.abs(r)
    # Taking the real part of a multiplication of two complex numbers
    # is equivalent to computing the dot product in the vector representation.
    term1 = 3.0 * np.real(m * r) * r / (abs_r**5)
    term2 = m / (abs_r**3)
    return mu_0 / (4 * np.pi) * (term1 - term2).sum(axis=1)

# --- array builders matching your requested sequences ---
def B_single_array(lam_w, h_gap, n_periods, z, phi0, sign_y, direction, odd=True):
    """
    direction = +1 (up-left-down-right), -1 (up-right-down-left)
    sign_y = +1 top array (at +h/2), -1 bottom array (at -h/2)
    """
    if odd:
        n = np.arange(-2*n_periods, 2*n_periods - 1)
    else:
        n = np.arange(-2*n_periods, 2*n_periods + 1)
    r = r_array(lam_w, h_gap, n, z, sign_y=sign_y)
    m_n = m_sequence(n, phi0=phi0, direction=direction)
    return B_from_dipoles(r, m_n)

def B_two_arrays(lam_w, h_gap, n_periods, z, phi_0=np.pi/2, odd=True):
    """
    Top: up-right-down-left  (direction = -1)
    Bottom: up-left-down-right (direction = +1)
    Both start at 'up' => phi0 = pi/2
    """
    B_top = B_single_array(lam_w, h_gap, n_periods, z, phi0=np.pi/2, sign_y=+1, direction=-1, odd=odd)
    B_bot = B_single_array(lam_w, h_gap, n_periods, z, phi0=np.pi, sign_y=-1, direction=+1, odd=odd)
    return B_top + B_bot

# --- example usage ---
lam_w = 0.01
h_gap = 0.005
n_periods = 15
z = np.linspace(-0.12, 0.12, 1000)
phi_0 = np.pi/2

B = B_two_arrays(lam_w, h_gap, n_periods, z, phi_0=phi_0, odd=True)

print(f"Integrated B_y: {np.trapezoid(B.imag, z):.6f} (arb. units)")
print(f"Integrated B_z: {np.trapezoid(B.real, z):.6f} (arb. units)")

plt.figure()
plt.plot(z, B.imag, label='B_y (vertical)', alpha=0.7)
plt.plot(z, B.real, label='B_z (longitudinal)', alpha=0.7)
plt.xlabel('z (m)')
plt.ylabel('B (arb. units)')
plt.title('Wiggler Field')
plt.grid(True)
plt.legend()
plt.show()