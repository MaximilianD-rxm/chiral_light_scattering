
import numpy as np
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import CenteredNorm
from scipy.special import spherical_jn, spherical_yn, lpmv, factorial

from light_scattering_chiral_tetrahedral_dipol import inv_T_phi, inv_T_phi_multipol

#Streusystem, theta [0, pi], chi [-pi, pi]
c = 1.25*10**-6 
h = 4

x1, y1, z1 = -c,  c, -c
x2, y2, z2 =  c, -c, -c
x3, y3, z3 =  c,  c,  c
x4, y4, z4 = -c, -c,  c

def a(h, index):
    return (30 + (2*h)*(index - 2.5))*10**-9

#Einfallende Welle
lmbda=1064*10**(-9)
k = (2*np.pi)/(lmbda)
s_x = 0
s_y = 0
s_z = 1

pol_mode = "LCP" 
q_x = 1j/np.sqrt(2)  
q_y = -1/np.sqrt(2)
q_z = 0

p = np.cross([s_x, s_y, s_z], [q_x, q_y, q_z])
p_x, p_y, p_z = p

def test_Mie_coeffs_dipol_multipol():
    #beliebige Testwerte
    a_q = 30e-9
    
    value_old = inv_T_phi(a_q)
    value_new = inv_T_phi_multipol(a_q)
    
    assert value_new == pytest.approx(value_old)