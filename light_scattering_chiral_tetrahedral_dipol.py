
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from scipy.special import spherical_jn, spherical_yn, lpmv, factorial

PLOT_XZ = True
PLOT_YZ = False
PLOT_DIAGONAL = False

PLOT_PHI_PSI   = True
PLOT_E_H       = True
PLOT_Im_EH_tot = True
PLOT_Im_EH_all = False

#3D Gitter
res = 91
limit = 4*10**-6

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
print('Polarisation:',p_x, p_y, p_z)



#Konstanten
epsilon_0 = 8.854*10**-12
mu_0 = 4*np.pi*10**-7
c_0 = 299792458
H_konst=(np.sqrt(epsilon_0*mu_0))/(1j*k*mu_0)


#Kugelkoordinaten berechnen
def r_i(x, y, z):
	return np.sqrt(x**2 + y**2 + z**2)

def theta_i(z, r):
	return np.arccos(z/r)

def chi_i(x, y):
	return np.atan2(y, x)


r1, r2, r3, r4 = r_i(x1, y1, z1), r_i(x2, y2, z2), r_i(x3, y3, z3), r_i(x4, y4, z4)
theta1, theta2, theta3, theta4 = theta_i(z1, r1), theta_i(z2, r2), theta_i(z3, r3), theta_i(z4, r4)
chi1, chi2, chi3, chi4 = chi_i(x1, y1), chi_i(x2, y2), chi_i(x3, y3), chi_i(x4, y4)


#Kugelkoordinaten der Verbindungsvektoren
def r_pq(x_p, y_p, z_p, x_q, y_q, z_q):
    return np.sqrt((x_q - x_p)**2 + (y_q - y_p)**2 + (z_q - z_p)**2)

def theta_pq(z_p, z_q, r_pq):
    val = (z_q - z_p) / r_pq
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val)

def chi_pq(x_p, y_p, x_q, y_q):
    return np.atan2((y_q - y_p), (x_q - x_p))

r12 = r_pq(x1, y1, z1, x2, y2, z2)
r13 = r_pq(x1, y1, z1, x3, y3, z3)
r14 = r_pq(x1, y1, z1, x4, y4, z4)
r23 = r_pq(x2, y2, z2, x3, y3, z3)
r24 = r_pq(x2, y2, z2, x4, y4, z4)
r34 = r_pq(x3, y3, z3, x4, y4, z4)


theta12, chi12 = theta_pq(z1, z2, r12), chi_pq(x1, y1, x2, y2)
theta13, chi13 = theta_pq(z1, z3, r13), chi_pq(x1, y1, x3, y3)
theta14, chi14 = theta_pq(z1, z4, r14), chi_pq(x1, y1, x4, y4)
theta23, chi23 = theta_pq(z2, z3, r23), chi_pq(x2, y2, x3, y3)
theta24, chi24 = theta_pq(z2, z4, r24), chi_pq(x2, y2, x4, y4)
theta34, chi34 = theta_pq(z3, z4, r34), chi_pq(x3, y3, x4, y4)


def inv_components(r_pq, theta_pq, chi_pq):
    r_qp = r_pq
    theta_qp = np.pi - theta_pq
    chi_qp = (chi_pq + 2 * np.pi) % (2 * np.pi) - np.pi
    return r_qp, theta_qp, chi_qp


r21, theta21, chi21 = inv_components(r12, theta12, chi12)
r31, theta31, chi31 = inv_components(r13, theta13, chi13)
r41, theta41, chi41 = inv_components(r14, theta14, chi14)
r32, theta32, chi32 = inv_components(r23, theta23, chi23)
r42, theta42, chi42 = inv_components(r24, theta24, chi24)
r43, theta43, chi43 = inv_components(r34, theta34, chi34)




#Plot des Tetraeders
def plot_spherical_vectors(vectors):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'b', 'g', 'y']

    for i, (r, theta, chi) in enumerate(vectors):
        x_std = r * np.sin(theta) * np.cos(chi)
        y_std = r * np.sin(theta) * np.sin(chi)
        z_std = r * np.cos(theta)
        
        x_plot = y_std
        y_plot = -x_std
        z_plot = z_std
        
        ax.quiver(0, 0, 0, x_plot, y_plot, z_plot, color=colors[i % len(colors)], 
                  label=f'r{i+1}', arrow_length_ratio=0.1)
        
    max_val = max([v[0] for v in vectors]) if vectors else 1
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    ax.set_title('3D Vektoren aus Kugelkoordinaten')
    ax.legend()
    ax.view_init(elev=9, azim=-75) 
    plt.show()

v1 = (r1, theta1, chi1)
v2 = (r2, theta2, chi2)
v3 = (r3, theta3, chi3)
v4 = (r4, theta4, chi4)

plot_spherical_vectors([v1, v2, v3, v4])










#sphärische Hankelfunktion 1.Art und Kugelflächenfunktionen + Ableitungen
def spherical_hn1(n, z):
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)

def d_spherical_hn1(n, z):
    return ( n*spherical_hn1(n-1, z) - (n+1) * spherical_hn1(n+1, z) ) / (2*n+1)

def Y_nm(n, m, theta, chi):
    abs_m = np.abs(m)
    norm = np.sqrt(((2*n + 1)/(4*np.pi)) * (factorial(n - abs_m)/factorial(n + abs_m)))
        
    return (-1.0) ** m * norm * lpmv(abs_m, n, np.cos(theta)) * np.exp(1j * m * chi)

def dY_nm_dchi(n, m, theta, chi):
    return 1j * m * Y_nm(n, m, theta, chi)

def dY_nm_dtheta(n, m, theta, chi):
    abs_m = np.abs(m)
    norm = np.sqrt(((2*n + 1)/(4*np.pi)) * (factorial(n - abs_m)/factorial(n + abs_m)))
    A = (n+abs_m)*(n-abs_m+1)
    dtheta_P_nm = 1/2* lpmv(abs_m+1, n, np.cos(theta))-1/2*A* lpmv(abs_m-1, n, np.cos(theta))
    
    return (-1.0) ** abs_m * norm * dtheta_P_nm * np.exp(1j * m * chi)












#Mie Koeffizienten und T-Matrix, n = 1
def inv_T_phi(a_q):
    Z = spherical_hn1(1, k*a_q)
    N = spherical_jn(1, k*a_q)
    return - Z / N


def inv_T_psi(a_q):
    Z = spherical_hn1(1, k*a_q)+k*a_q*1/(2*1+1)*(1*spherical_hn1(0, k*a_q)-(1+1)*spherical_hn1(2, k*a_q))
    N = spherical_jn(1, k*a_q)+k*a_q*1/(2*1+1)*(1*spherical_jn(0, k*a_q)-(1+1)*spherical_jn(2, k*a_q))
    return - Z / N


def inv_T_q(a_q):
    return np.array([
        [inv_T_phi(a_q), 0, 0, 0, 0, 0], 
        [0, inv_T_phi(a_q), 0, 0, 0, 0], 
        [0, 0, inv_T_phi(a_q), 0, 0, 0], 
        [0, 0, 0, inv_T_psi(a_q), 0, 0],
        [0, 0, 0, 0, inv_T_psi(a_q), 0], 
        [0, 0, 0, 0, 0, inv_T_psi(a_q)]])






#Rotationsmatrix
def epsilon(m):
  if m >= 0:
    return (-1) ** m
  else:
    return 1

def Hvm_n(theta, v, m, n):
    eps_m = epsilon(m)
    eps_v = epsilon(v)
    const4 = (factorial(n+v) * factorial(n-v) * factorial(n+m) * factorial(n-m))**(1/2)
    sigma_min = max(0, -(m+v))
    sigma_max = min(n-m, n-v)
    
    sum_val = 0
    for sigma in range(sigma_min, sigma_max + 1):
        Z = (((-1)**(n-sigma)) * 
        (np.cos(0.5*theta)**(2*sigma+v+m)) * 
        (np.sin(0.5*theta)**(2*n-2*sigma-v-m)))
        
        N = (factorial(sigma) * factorial(n-m-sigma) * factorial(n-v-sigma) * factorial(m+v+sigma))
        
        sum_val = sum_val + Z/N
        
    return eps_m * eps_v * const4 * sum_val


def Tvm_n(theta, phi, chi, v, m, n):
    return np.exp(1j*m*chi) * np.exp(-1j*v*phi) * Hvm_n(theta, v, m, n)


def RotQ(theta_pq, phi_pq, chi_pq):
    arg = (theta_pq, phi_pq, chi_pq)
    return np.array([
        [Tvm_n(*arg, -1, -1, 1), Tvm_n(*arg, -1, 0, 1), Tvm_n(*arg, -1, 1, 1)],
        [Tvm_n(*arg, 0, -1, 1), Tvm_n(*arg, 0, 0, 1), Tvm_n(*arg, 0, 1, 1)],
        [Tvm_n(*arg, 1, -1, 1), Tvm_n(*arg, 1, 0, 1), Tvm_n(*arg, 1, 1, 1)]])

def inv_RotQ(theta_pq, phi_pq, chi_pq):
    return RotQ(theta_pq, phi_pq, chi_pq).conj().T




#coaxiale Conversion Operatoren, Näherung da ohne n-1, n+1 Terme
def C11_coax(r_pq):
    return np.array([[1, 0, 0], 
                     [0, 1, 0], 
                     [0, 0, 1]])

def C21_pq_coax(r_pq):
    return np.array([[(-1j*r_pq)/2, 0,         0], 
                     [0,    	    0,         0], 
                     [0,            0, (1j*r_pq)/2]])

#allgemeine Conversion Operatoren
def C11_pq(r_pq, theta_pq, phi_pq, chi_pq):
    arg = (theta_pq, phi_pq, chi_pq)
    return inv_RotQ(*arg) @ C11_coax(r_pq) @ RotQ(*arg)

def C21_pq(r_pq, theta_pq, phi_pq, chi_pq):
    arg = (theta_pq, phi_pq, chi_pq)
    return inv_RotQ(*arg) @ C21_pq_coax(r_pq) @ RotQ(*arg)




#coaxiale Translationsoperatoren
def SR00_11(r_pq):
    return spherical_hn1(0, k*r_pq) - 2*spherical_hn1(2, k*r_pq)
def SR11_11(r_pq): 
    return spherical_hn1(0, k*r_pq) + spherical_hn1(2, k*r_pq)

def SR_pq_coax(r_pq):
    return np.array([[SR11_11(r_pq),    0,         0], 
                     [0,         SR00_11(r_pq),    0], 
                     [0,                0,   SR11_11(r_pq)]])


#allgemeine Translationsoperatoren
def SR_pq(r_pq, theta_pq, phi_pq, chi_pq):
    arg = (theta_pq, phi_pq, chi_pq)
    return inv_RotQ(*arg) @ SR_pq_coax(r_pq) @ RotQ(*arg)


def U_pq(r_pq, theta_pq, phi_pq, chi_pq):
    arg = (r_pq, theta_pq, phi_pq, chi_pq)
    A = C11_pq(*arg) @ SR_pq(*arg)
    B = C21_pq(*arg) @ SR_pq(*arg)
    return np.block([[A, k**2 * B],
                     [B,        A]])


def inv_T1(h):
    return inv_T_q(a(h, 1))
def inv_T2(h):
    return inv_T_q(a(h, 2))
def inv_T3(h):
    return inv_T_q(a(h, 3))
def inv_T4(h):
    return inv_T_q(a(h, 4))

U12 = U_pq(r12, theta12, 0, chi12)
U21 = U_pq(r21, theta21, 0, chi21)

U13 = U_pq(r13, theta13, 0, chi13)
U31 = U_pq(r31, theta31, 0, chi31)

U14 = U_pq(r14, theta14, 0, chi14)
U41 = U_pq(r41, theta41, 0, chi41)

U23 = U_pq(r23, theta23, 0, chi23)
U32 = U_pq(r32, theta32, 0, chi32)

U24 = U_pq(r24, theta24, 0, chi24)
U42 = U_pq(r42, theta42, 0, chi42)

U34 = U_pq(r34, theta34, 0, chi34)
U43 = U_pq(r43, theta43, 0, chi43)


#Matrix L (24x24)
def L(h):
    return np.block([[inv_T1(h), -U21,   -U31,   -U41],
                     [-U12,   inv_T2(h), -U32,   -U42],
                     [-U13,   -U23,   inv_T3(h), -U43],
                     [-U14,   -U24,   -U34,   inv_T4(h)]])


 



def phase_factor(r, theta, chi):
    x = r * np.sin(theta) * np.cos(chi)
    y = r * np.sin(theta) * np.sin(chi)
    z = r * np.cos(theta)
    return np.exp(1j * k * (s_x * x + s_y * y + s_z * z))

#Einfallende Welle Y 1x24
def inc_phi_m1_1(r_q, theta_q, chi_q):
    return phase_factor(r_q, theta_q, chi_q) * 2*np.pi*(-(np.sqrt(2))/2 *((p_x+1j*p_y)*Y_nm(1, 0, 0, 0))+p_z*Y_nm(1, 1, 0, 0))

def inc_phi_0_1(r_q, theta_q, chi_q):
    return phase_factor(r_q, theta_q, chi_q) * np.pi*np.sqrt(2) * ((p_x+1j*p_y)* Y_nm(1, -1, 0, 0) - (p_x-1j*p_y)* Y_nm(1, 1, 0, 0))

def inc_phi_1_1(r_q, theta_q, chi_q):
    return phase_factor(r_q, theta_q, chi_q) * 2*np.pi*((np.sqrt(2))/2 *((p_x-1j*p_y)*Y_nm(1, 0, 0, 0))-p_z*Y_nm(1, -1, 0, 0))

def inc_psi_m1_1(r_q, theta_q, chi_q):
    return phase_factor(r_q, theta_q, chi_q) * (-2*np.pi*1j)/k*(-(np.sqrt(2))/2 *((q_x+1j*q_y)*Y_nm(1, 0, 0, 0))+q_z*Y_nm(1, 1, 0, 0))

def inc_psi_0_1(r_q, theta_q, chi_q):
    return phase_factor(r_q, theta_q, chi_q) * (-1j*np.pi*np.sqrt(2))/k * ((q_x+1j*q_y)* Y_nm(1, -1, 0, 0) - (q_x-1j*q_y)* Y_nm(1, 1, 0, 0))

def inc_psi_1_1(r_q, theta_q, chi_q):
    return phase_factor(r_q, theta_q, chi_q) * (-2*np.pi*1j)/k*((np.sqrt(2))/2 *((q_x-1j*q_y)*Y_nm(1, 0, 0, 0))-q_z*Y_nm(1, -1, 0, 0))

def y_q(r_q, theta_q, chi_q):
    arg = (r_q, theta_q, chi_q)
    return np.array([[inc_phi_m1_1(*arg)], [inc_phi_0_1(*arg)], [inc_phi_1_1(*arg)],
                   [inc_psi_m1_1(*arg)], [inc_psi_0_1(*arg)], [inc_psi_1_1(*arg)]], dtype=complex)


#Phasenfaktor 1 im Ursprung
Y = np.block([[y_q(r1, theta1, chi1)],
              [y_q(r2, theta2, chi2)],
              [y_q(r3, theta3, chi3)],
              [y_q(r4, theta4, chi4)]])




def X(h):
    return np.linalg.solve(L(h), Y)

def abs_X(h):
    return np.abs(X(h))

print(f"Radien der Kugeln: a1={a(h, 1):.1e} a2={a(h, 2):.1e} a3={a(h, 3):.1e} a4={a(h, 4):.1e} für h={h}")
print('Produkt für Dipolnäherung: ka4=',k*a(h, 4))
print(rf"X_{pol_mode}(h={h})=")
print(X(h))

X_val=X(h)
X_flat = X_val.flatten()







#Debye-Potentiale
def S_m1_1(r_prime_p, theta_prime_p, chi_prime_p):
    return spherical_hn1(1, k*r_prime_p)* Y_nm(1, -1, theta_prime_p, chi_prime_p)
def S_0_1(r_prime_p, theta_prime_p, chi_prime_p):
    return spherical_hn1(1, k*r_prime_p)* Y_nm(1, 0, theta_prime_p, chi_prime_p)
def S_1_1(r_prime_p, theta_prime_p, chi_prime_p):
    return spherical_hn1(1, k*r_prime_p)* Y_nm(1, 1, theta_prime_p, chi_prime_p)


def phi_scat_sphere(r_prime, theta_prime, chi_prime, coeffs):
    arg = (r_prime, theta_prime, chi_prime)
    return coeffs[0]*S_m1_1(*arg) + coeffs[1]*S_0_1(*arg) + coeffs[2]*S_1_1(*arg)

def psi_scat_sphere(r_prime, theta_prime, chi_prime, coeffs):
    arg = (r_prime, theta_prime, chi_prime)
    return coeffs[0]*S_m1_1(*arg) + coeffs[1]*S_0_1(*arg) + coeffs[2]*S_1_1(*arg)







#Elektrische und magnetische Felder
# ∇S_nm
def grad_S_nm(n, m, r, theta, chi):
    
    grad_r = Y_nm(n, m, theta, chi) * k * d_spherical_hn1(n, k*r)
    
    grad_t = (1/r) * spherical_hn1(n, k*r) * dY_nm_dtheta(n, m, theta, chi)
    
    grad_c = (1/(r*np.sin(theta))) * spherical_hn1(n, k*r) * dY_nm_dchi(n, m, theta, chi)
    
    return grad_r, grad_t, grad_c


# M_nm = ∇S_nm × r
def M_nm(n, m, r, theta, chi):
    grad_r, grad_t, grad_c = grad_S_nm(n, m, r, theta, chi)
    
    M_nm_r = np.zeros_like(r)
    M_nm_t = grad_c * r
    M_nm_c = - grad_t * r
    
    return np.array([M_nm_r, M_nm_t, M_nm_c])


# N_nm = ∇ × M_nm
def N_nm(n, m, r, theta, chi):
    
    N_nm_r = (n * (n + 1) / r) * spherical_hn1(n, k*r) * Y_nm(n, m, theta, chi)
    N_nm_t = (1/r)*dY_nm_dtheta(n, m, theta, chi)*(spherical_hn1(n, k*r) + k*r*d_spherical_hn1(n, k*r))
    N_nm_c = 1/(r*np.sin(theta)) * dY_nm_dchi(n, m, theta, chi)*(spherical_hn1(n, k*r) + k*r*d_spherical_hn1(n, k*r))
    
    return np.array([N_nm_r, N_nm_t, N_nm_c])



# gestreutes E-Feld
def E_scat_sphere(r_prime, theta_prime, chi_prime, coeffs):
    M_nm_1_m1 = M_nm(1, -1, r_prime, theta_prime, chi_prime)
    M_nm_1_0  = M_nm(1,  0, r_prime, theta_prime, chi_prime)
    M_nm_1_1  = M_nm(1,  1, r_prime, theta_prime, chi_prime)
    
    N_nm_1_m1 = N_nm(1, -1, r_prime, theta_prime, chi_prime)
    N_nm_1_0  = N_nm(1,  0, r_prime, theta_prime, chi_prime)
    N_nm_1_1  = N_nm(1,  1, r_prime, theta_prime, chi_prime)
    
    sum_phi_nm = coeffs[0]*M_nm_1_m1 + coeffs[1]*M_nm_1_0 + coeffs[2]*M_nm_1_1
    sum_psi_nm = coeffs[3]*N_nm_1_m1 + coeffs[4]*N_nm_1_0 + coeffs[5]*N_nm_1_1
    
    E_r, E_theta, E_chi = sum_phi_nm + sum_psi_nm
    
    E_x = E_r * np.sin(theta_prime) * np.cos(chi_prime) + E_theta * np.cos(theta_prime) * np.cos(chi_prime) - E_chi * np.sin(chi_prime)
    E_y = E_r * np.sin(theta_prime) * np.sin(chi_prime) + E_theta * np.cos(theta_prime) * np.sin(chi_prime) + E_chi * np.cos(chi_prime)
    E_z = E_r * np.cos(theta_prime) - E_theta * np.sin(theta_prime)
    
    return np.array([E_x, E_y, E_z])




# gestreutes H-Feld
def H_scat_sphere(r_prime, theta_prime, chi_prime, coeffs):
    M_nm_1_m1 = M_nm(1, -1, r_prime, theta_prime, chi_prime)
    M_nm_1_0  = M_nm(1,  0, r_prime, theta_prime, chi_prime)
    M_nm_1_1  = M_nm(1,  1, r_prime, theta_prime, chi_prime)
    
    N_nm_1_m1 = N_nm(1, -1, r_prime, theta_prime, chi_prime)
    N_nm_1_0  = N_nm(1,  0, r_prime, theta_prime, chi_prime)
    N_nm_1_1  = N_nm(1,  1, r_prime, theta_prime, chi_prime)
    
    sum_phi_nm = coeffs[0]*N_nm_1_m1 + coeffs[1]*N_nm_1_0 + coeffs[2]*N_nm_1_1
    sum_psi_nm = coeffs[3]*M_nm_1_m1 + coeffs[4]*M_nm_1_0 + coeffs[5]*M_nm_1_1
    
    H_r, H_theta, H_chi = H_konst * (k**2 * sum_psi_nm + sum_phi_nm)
    
    H_x = H_r * np.sin(theta_prime) * np.cos(chi_prime) + H_theta * np.cos(theta_prime) * np.cos(chi_prime) - H_chi * np.sin(chi_prime)
    H_y = H_r * np.sin(theta_prime) * np.sin(chi_prime) + H_theta * np.cos(theta_prime) * np.sin(chi_prime) + H_chi * np.cos(chi_prime)
    H_z = H_r * np.cos(theta_prime) - H_theta * np.sin(theta_prime)
    
    return np.array([H_x, H_y, H_z])



#Einfallende Felder für s = e_z
def E_inc_func(q_x, q_y, k, z):
    E_inc_x = -q_y * np.exp(1j*k*z)
    E_inc_y = q_x * np.exp(1j*k*z)
    E_inc_z = np.zeros_like(E_inc_x)
    return np.array([E_inc_x, E_inc_y, E_inc_z])

def H_inc_func(q_x, q_y, k, z):
    H_inc_x = -q_x * np.exp(1j*k*z) * 1/(c_0 * mu_0)
    H_inc_y = -q_y * np.exp(1j*k*z) * 1/(c_0 * mu_0)
    H_inc_z = np.zeros_like(H_inc_x)
    return np.array([H_inc_x, H_inc_y, H_inc_z])









# 3D Gitter und Berechnungen
x_range = np.linspace(-limit, limit, res)
y_range = np.linspace(-limit, limit, res)
z_range = np.linspace(-limit, limit, res)

# 'ij' indexing sorgt dafür, dass X_grid[i,j,k] zu x_range[i] gehört
X_grid, Y_grid, Z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')



r = np.sqrt(X_grid**2 + Y_grid**2 + Z_grid**2)
# Fangen von Division durch Null bei r=0, falls nötig
with np.errstate(divide='ignore', invalid='ignore'):
    theta = np.arccos(np.clip(Z_grid / r, -1.0, 1.0))
    chi = np.arctan2(Y_grid, X_grid)

x_prime_1 = X_grid - x1
y_prime_1 = Y_grid - y1
z_prime_1 = Z_grid - z1
r_prime_1_REAL = np.sqrt(x_prime_1**2 + y_prime_1**2 + z_prime_1**2)
r_prime_1 = np.clip(r_prime_1_REAL, 3*a(h, 1), None)
theta_prime_1 = np.arccos(np.clip(z_prime_1 / r_prime_1_REAL, -1.0, 1.0))
chi_prime_1 = np.arctan2(y_prime_1, x_prime_1)

x_prime_2 = X_grid - x2
y_prime_2 = Y_grid - y2
z_prime_2 = Z_grid - z2
r_prime_2_REAL = np.sqrt(x_prime_2**2 + y_prime_2**2 + z_prime_2**2)
r_prime_2 = np.clip(r_prime_2_REAL, 3*a(h, 2), None)
theta_prime_2 = np.arccos(np.clip(z_prime_2 / r_prime_2_REAL, -1.0, 1.0))
chi_prime_2 = np.arctan2(y_prime_2, x_prime_2)

x_prime_3 = X_grid - x3
y_prime_3 = Y_grid - y3
z_prime_3 = Z_grid - z3
r_prime_3_REAL = np.sqrt(x_prime_3**2 + y_prime_3**2 + z_prime_3**2)
r_prime_3 = np.clip(r_prime_3_REAL, 3*a(h, 3), None)
theta_prime_3 = np.arccos(np.clip(z_prime_3 / r_prime_3_REAL, -1.0, 1.0))
chi_prime_3 = np.arctan2(y_prime_3, x_prime_3)

x_prime_4 = X_grid - x4
y_prime_4 = Y_grid - y4
z_prime_4 = Z_grid - z4
r_prime_4_REAL = np.sqrt(x_prime_4**2 + y_prime_4**2 + z_prime_4**2)
r_prime_4 = np.clip(r_prime_4_REAL, 3*a(h, 4), None)
theta_prime_4 = np.arccos(np.clip(z_prime_4 / r_prime_4_REAL, -1.0, 1.0))
chi_prime_4 = np.arctan2(y_prime_4, x_prime_4)




#Streupotentiale
phi_scat_3D = sum([
    phi_scat_sphere(r_prime_1, theta_prime_1, chi_prime_1, X_flat[0:3]),
    phi_scat_sphere(r_prime_2, theta_prime_2, chi_prime_2, X_flat[6:9]),
    phi_scat_sphere(r_prime_3, theta_prime_3, chi_prime_3, X_flat[12:15]),
    phi_scat_sphere(r_prime_4, theta_prime_4, chi_prime_4, X_flat[18:21])
])

psi_scat_3D = sum([
    psi_scat_sphere(r_prime_1, theta_prime_1, chi_prime_1, X_flat[3:6]),
    psi_scat_sphere(r_prime_2, theta_prime_2, chi_prime_2, X_flat[9:12]),
    psi_scat_sphere(r_prime_3, theta_prime_3, chi_prime_3, X_flat[15:18]),
    psi_scat_sphere(r_prime_4, theta_prime_4, chi_prime_4, X_flat[21:24])
])




#Felder
E_scat = sum([
    E_scat_sphere(r_prime_1, theta_prime_1, chi_prime_1, X_flat[0:6]),
    E_scat_sphere(r_prime_2, theta_prime_2, chi_prime_2, X_flat[6:12]),
    E_scat_sphere(r_prime_3, theta_prime_3, chi_prime_3, X_flat[12:18]),
    E_scat_sphere(r_prime_4, theta_prime_4, chi_prime_4, X_flat[18:24])])

E_mag_3D = np.sqrt(np.abs(E_scat[0])**2 + np.abs(E_scat[1])**2 + np.abs(E_scat[2])**2)


H_scat = sum([
    H_scat_sphere(r_prime_1, theta_prime_1, chi_prime_1, X_flat[0:6]),
    H_scat_sphere(r_prime_2, theta_prime_2, chi_prime_2, X_flat[6:12]),
    H_scat_sphere(r_prime_3, theta_prime_3, chi_prime_3, X_flat[12:18]),
    H_scat_sphere(r_prime_4, theta_prime_4, chi_prime_4, X_flat[18:24])])

H_mag_3D = np.sqrt(np.abs(H_scat[0])**2 + np.abs(H_scat[1])**2 + np.abs(H_scat[2])**2)


E_inc = E_inc_func(q_x, q_y, k, Z_grid)
H_inc = H_inc_func(q_x, q_y, k, Z_grid)


E_tot = E_scat + E_inc
H_tot = H_scat + H_inc



#Chirale Dichte Im(E·H*)
Im_E_dot_H_star_inc = np.imag(np.sum(E_inc * np.conj(H_inc), axis=0))

Im_E_dot_H_star_scat = np.imag(np.sum(E_scat * np.conj(H_scat), axis=0))

Im_E_dot_H_star_mixed1 = np.imag(np.sum(E_scat * np.conj(H_inc), axis=0))
Im_E_dot_H_star_mixed2 = np.imag(np.sum(E_inc * np.conj(H_scat), axis=0))


Im_E_dot_H_star = np.imag(np.sum(E_tot * np.conj(H_tot), axis=0))





#Maximum der chiralen Dichte berechnen und finden
abs_Im_E_dot_H = np.abs(Im_E_dot_H_star)
Im_E_dot_H_star_max = np.max(abs_Im_E_dot_H)

flat_max_idx = np.argmax(abs_Im_E_dot_H)
idx_x9, idx_y9, idx_z9 = np.unravel_index(flat_max_idx, abs_Im_E_dot_H.shape)

max_x = X_grid[idx_x9, idx_y9, idx_z9]
max_y = Y_grid[idx_x9, idx_y9, idx_z9]
max_z = Z_grid[idx_x9, idx_y9, idx_z9]

print(f'max(abs(Im_E_dot_H_star_3D)) = {Im_E_dot_H_star_max} für h = {h}')
print(f'Ort des Maximums: x = {max_x:.4e}, y = {max_y:.4e}, z = {max_z:.4e}')
print(f'Zugehörige Gitter-Indizes: [{idx_x9}, {idx_y9}, {idx_z9}]')




#Streuamplitude und Wirkungsquerschnitt
def f_q(z, k, coeffs):
    phase = np.exp(-1j*k*z)
    
    f_x = 1j*1/k*coeffs[0] - 1j*1/k*coeffs[2] - 1j*coeffs[3] - 1j*coeffs[5]
    f_y = 1/k*coeffs[0] + 1/k*coeffs[2] - coeffs[3] + coeffs[5]
    f_z = 0
    
    return phase * np.sqrt(3/(8*np.pi)) * np.array([f_x, f_y, f_z])



f = f_q(z1, k, X_flat[0:6]) + f_q(z2, k, X_flat[6:12]) + f_q(z3, k, X_flat[12:18]) + f_q(z4, k, X_flat[18:24])
f_x, f_y, f_z = f


sigma = (4*np.pi)/k * np.imag( np.conj(p_x)*f_x + np.conj(p_y)*f_y + np.conj(p_z)*f_z )


print(f'Wirkungsquerschnitt: sigma = {sigma}')






        
        

#Plotten
def plot_2d(X, Y, data, title, xlabel, ylabel, cbar_label, cmap='cividis', norm=None, figsize=(6, 4.5)):
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=12)
    cf = plt.contourf(X, Y, data, levels=400, cmap=cmap, norm=norm)
    plt.colorbar(cf, label=cbar_label, fraction=0.225, pad=0.03, format='%.2e')   
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.axis('equal')
    plt.show()


#Plotten in der x-z-Ebene mit y = 0
if PLOT_XZ:
    y_idx = res // 2
    y_xz = y_range[y_idx]
    X_xz, Z_xz = X_grid[:, y_idx, :].T, Z_grid[:, y_idx, :].T
    xl, yl = '$x$ (m)', '$z$ (m)'
    
    if PLOT_PHI_PSI:
        plot_2d(X_xz, Z_xz, np.abs(phi_scat_3D[:, y_idx, :]).T, 
                fr'{pol_mode}: |$\phi_{{\mathrm{{scat}}}}$| in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, '|$\phi_{\mathrm{scat}}$| (V/m)', 'cividis')
        
        plot_2d(X_xz, Z_xz, np.abs(psi_scat_3D[:, y_idx, :]).T, 
                fr'{pol_mode}: |$\psi_{{\mathrm{{scat}}}}$| in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, '|$\psi_{\mathrm{scat}}$| (V)', 'cividis')

    if PLOT_E_H:
        plot_2d(X_xz, Z_xz, E_mag_3D[:, y_idx, :].T, 
                fr'{pol_mode}: |$\mathbf{{E}}_{{\mathrm{{scat}}}}$| in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, '|$\mathbf{E}_{\mathrm{scat}}$| (V/m)', 'viridis')
        
        plot_2d(X_xz, Z_xz, H_mag_3D[:, y_idx, :].T, 
                fr'{pol_mode}: |$\mathbf{{H}}_{{\mathrm{{scat}}}}$| in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, '|$\mathbf{H}_{\mathrm{scat}}$| (A/m)', 'viridis')

    if PLOT_Im_EH_tot:
        plot_2d(X_xz, Z_xz, Im_E_dot_H_star[:, y_idx, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{tot}}}} \cdot \mathbf{{H}}_{{\mathrm{{tot}}}}^*$) in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma')
        
        

    if PLOT_Im_EH_all:
        plot_2d(X_xz, Z_xz, Im_E_dot_H_star_inc[:, y_idx, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma')
        
        plot_2d(X_xz, Z_xz, Im_E_dot_H_star_scat[:, y_idx, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())
        
        plot_2d(X_xz, Z_xz, Im_E_dot_H_star_mixed1[:, y_idx, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())
        
        plot_2d(X_xz, Z_xz, Im_E_dot_H_star_mixed2[:, y_idx, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())





#Plotten in der y-z-Ebene mit x = 0
if PLOT_YZ:
    x_idx = res // 2
    x_yz = x_range[x_idx]
    Y_yz, Z_yz = Y_grid[x_idx, :, :].T, Z_grid[x_idx, :, :].T
    xl, yl = '$y$ (m)', '$z$ (m)'

    if PLOT_PHI_PSI:
        plot_2d(Y_yz, Z_yz, np.abs(phi_scat_3D[x_idx, :, :]).T, 
                fr'{pol_mode}: |$\phi_{{\mathrm{{scat}}}}$| in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, '|$\phi_{\mathrm{scat}}$| (V/m)', 'cividis')
        
        plot_2d(Y_yz, Z_yz, np.abs(psi_scat_3D[x_idx, :, :]).T, 
                fr'{pol_mode}: |$\psi_{{\mathrm{{scat}}}}$| in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, '|$\psi_{\mathrm{scat}}$| (V)', 'cividis')

    if PLOT_E_H:
        plot_2d(Y_yz, Z_yz, E_mag_3D[x_idx, :, :].T, 
                fr'{pol_mode}: |$\mathbf{{E}}_{{\mathrm{{scat}}}}$| in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, '|$\mathbf{E}_{\mathrm{scat}}$| (V/m)', 'viridis')
        
        plot_2d(Y_yz, Z_yz, H_mag_3D[x_idx, :, :].T, 
                fr'{pol_mode}: |$\mathbf{{H}}_{{\mathrm{{scat}}}}$| in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, '|$\mathbf{H}_{\mathrm{scat}}$| (A/m)', 'viridis')

    if PLOT_Im_EH_tot:
        plot_2d(Y_yz, Z_yz, Im_E_dot_H_star[x_idx, :, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{tot}}}} \cdot \mathbf{{H}}_{{\mathrm{{tot}}}}^*$) in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma')
        
    if PLOT_Im_EH_all:
        plot_2d(Y_yz, Z_yz, Im_E_dot_H_star_inc[x_idx, :, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma')
        
        plot_2d(Y_yz, Z_yz, Im_E_dot_H_star_scat[x_idx, :, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())
        
        plot_2d(Y_yz, Z_yz, Im_E_dot_H_star_mixed1[x_idx, :, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())
        
        plot_2d(Y_yz, Z_yz, Im_E_dot_H_star_mixed2[x_idx, :, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())





#Plotten auf der Diagonalebene entlang z mit x=y
if PLOT_DIAGONAL:
    indices = np.arange(res)
    X_diag = X_grid[indices, indices, :].T
    Y_diag = Y_grid[indices, indices, :].T
    Z_diag = Z_grid[indices, indices, :].T
    D_diag = X_diag * np.sqrt(2)
    xl, yl = '$d$ (m)', '$z$ (m)'
    fig_sz = (8, 4.5)

    if PLOT_PHI_PSI:
        plot_2d(D_diag, Z_diag, np.abs(phi_scat_3D[indices, indices, :]).T, 
                fr'{pol_mode}: |$\phi_{{\mathrm{{scat}}}}$| (x=y), h={h}', 
                xl, yl, '|$\phi_{\mathrm{scat}}$| (V/m)', 'cividis', figsize=fig_sz)
        
        plot_2d(D_diag, Z_diag, np.abs(psi_scat_3D[indices, indices, :]).T, 
                fr'{pol_mode}: |$\psi_{{\mathrm{{scat}}}}$| (x=y), h={h}', 
                xl, yl, '|$\psi_{\mathrm{scat}}$| (V)', 'cividis', figsize=fig_sz)

    if PLOT_E_H:
        plot_2d(D_diag, Z_diag, E_mag_3D[indices, indices, :].T, 
                fr'{pol_mode}: |$\mathbf{{E}}_{{\mathrm{{scat}}}}$| (x=y), h={h}', 
                xl, yl, '|$\mathbf{E}_{\mathrm{scat}}$| (V/m)', 'viridis', figsize=fig_sz)
        plot_2d(D_diag, Z_diag, H_mag_3D[indices, indices, :].T, 
                fr'{pol_mode}: |$\mathbf{{H}}_{{\mathrm{{scat}}}}$| (x=y), h={h}', 
                xl, yl, '|$\mathbf{H}_{\mathrm{scat}}$| (A/m)', 'viridis', figsize=fig_sz)

    if PLOT_Im_EH_tot:
        plot_2d(D_diag, Z_diag, Im_E_dot_H_star[indices, indices, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{tot}}}} \cdot \mathbf{{H}}_{{\mathrm{{tot}}}}^*$) (x=y), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', figsize=fig_sz)
        
    if PLOT_Im_EH_all:
        plot_2d(D_diag, Z_diag, Im_E_dot_H_star_inc[indices, indices, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) (x=y), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', figsize=fig_sz)
        
        plot_2d(D_diag, Z_diag, Im_E_dot_H_star_scat[indices, indices, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) (x=y), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm(), figsize=fig_sz)
        
        plot_2d(D_diag, Z_diag, Im_E_dot_H_star_mixed1[indices, indices, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) (x=y), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm(), figsize=fig_sz)
        
        plot_2d(D_diag, Z_diag, Im_E_dot_H_star_mixed2[indices, indices, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) (x=y), h={h}', 
                xl, yl, 'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm(), figsize=fig_sz)

