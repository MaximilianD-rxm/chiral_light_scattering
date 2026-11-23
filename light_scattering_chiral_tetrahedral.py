
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from scipy.special import spherical_jn, spherical_yn, lpmv, factorial
from scipy.linalg import block_diag

#Plots
GEOMETRY, XZ, YZ, DIAGONAL = False, True, False, False
PHI_PSI, E_H, Im_EH_tot, Im_EH_all = False, True, False, False

#3D Gitter
res = 31
limit = 4e-6

#Multipolordnung
n_max = 20
coeff_number = [0] + [2*i+1 for i in range(1, n_max+2)]
coeff_sum = [sum(coeff_number[0:i+1]) for i in range(0, n_max+2)]

#Streusystem, theta [0, pi], chi [-pi, pi]
c = 1.25e-6
h = 4

x1, y1, z1 = -c,  c, -c
x2, y2, z2 =  c, -c, -c
x3, y3, z3 =  c,  c,  c
x4, y4, z4 = -c, -c,  c

def a(h, index):
    return (30 + (2*h)*(index - 2.5)) * 1e-9

#Einfallende Welle
pol_mode = "LCP" 
lmbda = 1064e-9
k = (2*np.pi)/(lmbda)
s_x, s_y, s_z = 0, 0, 1

R_factor = -1 if pol_mode == "RCP" else 1
q_x, q_y, q_z = R_factor * 1j/np.sqrt(2), -1/np.sqrt(2), 0

p = np.cross([s_x, s_y, s_z], [q_x, q_y, q_z])
p_x, p_y, p_z = p
print('Polarisation:',p_x, p_y, p_z)



#Konstanten
epsilon_0 = 8.8541878188e-12
mu_0 = 1.25663706127e-6
c_0 = 299792458


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
if GEOMETRY:
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
    return (n*spherical_hn1(n-1, z) - (n+1) * spherical_hn1(n+1, z)) / (2*n+1)

def Y_nm(n, m, theta, chi):
    abs_m = np.abs(m)
    norm = np.sqrt(((2*n + 1)/(4*np.pi)) * (factorial(n - abs_m)/factorial(n + abs_m)))
        
    return (-1.0)**m * norm * lpmv(abs_m, n, np.cos(theta)) * np.exp(1j * m * chi)

def dY_nm_dtheta(n, m, theta, chi):
    abs_m = np.abs(m)
    norm = np.sqrt(((2*n + 1)/(4*np.pi)) * (factorial(n - abs_m)/factorial(n + abs_m)))
    A = (n+abs_m)*(n-abs_m+1)
    dtheta_P_nm = 1/2* lpmv(abs_m+1, n, np.cos(theta))-1/2*A* lpmv(abs_m-1, n, np.cos(theta))
    
    return (-1.0)**m * norm * dtheta_P_nm * np.exp(1j * m * chi)












#Mie Koeffizienten und T-Matrix
def T(a_q, n):
    x = k * a_q
    size = coeff_sum[n]
    T = np.zeros((2*size, 2*size), dtype=complex)
    
    for o in range(1, n + 1):
        g, f = coeff_sum[o-1], coeff_sum[o]
        idx = np.arange(g, f)
        
        j_n,  dj_n  = spherical_jn(o, x), spherical_jn(o, x, derivative=True)
        h_n, dh_n = spherical_hn1(o, x), d_spherical_hn1(o, x)
        
        T_phi = - j_n / h_n
        
        T_psi = - (j_n + x * dj_n) / (h_n + x * dh_n)
        
        T[idx, idx] = T_phi
        T[idx + size, idx + size] = T_psi
        
    return T

T1, T2, T3, T4 = T(a(h, 1), n_max), T(a(h, 2), n_max), T(a(h, 3), n_max), T(a(h, 4), n_max)



#Rotationsmatrix
def Hvm_n(theta, v, m, n):
    eps_m = np.where(m >= 0, (-1.0)**m, 1)
    eps_v = np.where(v >= 0, (-1.0)**v, 1)
    const4 = (factorial(n+v) * factorial(n-v) * factorial(n+m) * factorial(n-m))**(1/2)
    sum_val = np.zeros_like(v, dtype=float)
    
    sigma_min = np.maximum(0, -(m+v))
    sigma_max = np.minimum(n-m, n-v)
    
    for sigma in range(0, 2*n+1):

        valid = (sigma >= sigma_min) & (sigma <= sigma_max)
        
        if not np.any(valid):
            continue

        Z = (((-1)**(n-sigma)) * 
        (np.cos(0.5*theta)**(2*sigma+v+m)) * 
        (np.sin(0.5*theta)**(2*n-2*sigma-v-m)))
        
        N = (factorial(sigma) * factorial(n-m-sigma) * factorial(n-v-sigma) * factorial(m+v+sigma))
        N = np.where(valid, N, 1.0)
        sum_val = sum_val + np.where(valid, Z/N, 0.0)
        
    return eps_m * eps_v * const4 * sum_val


def Tvm_n(theta, phi, chi, v, m, n):
    return np.exp(1j*m*chi) * np.exp(-1j*v*phi) * Hvm_n(theta, v, m, n)


def RotQ(theta_pq, phi_pq, chi_pq, n_max):
    size = coeff_sum[n_max]
    Rot_Q = np.zeros((size, size), dtype=complex)
    
    for n in range(1, n_max + 1):
        g, f = coeff_sum[n-1], coeff_sum[n]
        indices = np.arange(-n, n + 1)
        
        v, m = np.meshgrid(indices, indices, indexing='ij')
        
        Rot_Q[g:f, g:f] = Tvm_n(theta_pq, phi_pq, chi_pq, v, m, n)

    return Rot_Q








#koaxiale Translationsoperatoren
def a_nm(n, m):
    cond = n >= np.abs(m)
    m = np.where(cond, m, 0)
    val = np.sqrt(((n + 1 + np.abs(m))*(n + 1 - np.abs(m)))/((2*n + 1)*(2*n + 3)))
    
    return np.where(cond, val, 0)

def b_nm(n, m):
    cond1 = n >= np.abs(m)
    m = np.where(cond1, m, 0)
    cond2 = m >= 0
    sqrt = np.sqrt(((n - m - 1)*(n - m))/((2*n - 1)*(2*n + 1)))
    val = np.where(cond2, sqrt, -sqrt)
    
    return np.where(cond1, val, 0)

# 3D-Array C[m, l, n] der koaxialen Translationskoeffizienten m, l, n ∈ 0, 1, ... n_max+1
def SR_coax_coeff(r_pq, n_max):
    lim = n_max + 1
    C = np.zeros((lim + 1, 2*lim + 1, lim + 1), dtype=complex)
 
    l = np.arange(2*lim + 1)
    C[0, :, 0] = (-1.0) ** l * np.sqrt(2 * l + 1) * spherical_hn1(l, k * r_pq)
 
    for m in range(0, lim):
            l_m = np.arange(m+1, 2*lim - m)
            C[m+1, l_m, m+1] = (b_nm(l_m, -m-1) * C[m, l_m-1, m] - b_nm(l_m+1, m) * C[m, l_m+1, m]) / b_nm(m+1, -m-1)
 
        
            for n in range(m, lim):
                l_n = np.arange(m, 2*lim - n)
                C[m, l_n, n + 1] = (a_nm(l_n, m)*C[m, l_n+1, n] - a_nm(l_n-1, m)*C[m, l_n-1, n] - a_nm(n-1, m)*C[m, l_n, n-1]) / -a_nm(n, m)
 
    return C[:, : lim + 1, :]


# 2D-Matrix der koaxialen Translationskoeffizienten (Dimension: coeff_sum[n_max+1]+1 x coeff_sum[n_max])
def SR_pq_coax(r_pq, n_max):
    C = SR_coax_coeff(r_pq, n_max)
    SR = np.zeros((coeff_sum[n_max+1], coeff_sum[n_max+1]), dtype=complex)
    
    for n in range(1, n_max+2):
        g, f = coeff_sum[n-1], coeff_sum[n]
        abs_m = np.concatenate([np.arange(n, -1, -1), np.arange(1, n + 1)])
        SR[g:f, g:f] = np.diag(C[abs_m, n, n])
      
        for j in range(1, n):
            g2, f2 = 0, coeff_number[j]
            abs_m2 = np.concatenate([np.arange(j, -1, -1), np.arange(1, j+1)])
            SR[g2+coeff_sum[j-1]:f2+coeff_sum[j-1], g2+coeff_sum[n-1]+n-j:f2+coeff_sum[n-1]+n-j] = np.diag(C[abs_m2, j, n])
            SR[g2+coeff_sum[n-1]+n-j:f2+coeff_sum[n-1]+n-j, g2+coeff_sum[j-1]:f2+coeff_sum[j-1]] = np.diag(C[abs_m2, n, j])
      
    row0 = np.zeros(coeff_sum[n_max], dtype=complex)
    n_0 = np.arange(1, n_max + 1)
    row0[np.asarray(coeff_sum[:n_max]) + n_0] = C[0, 0, 1:n_max + 1]
    
    return np.vstack([row0, SR[:, :coeff_sum[n_max]]])








# Koaxiale Konversionsmatrix (Dimension: 2*coeff_sum[n_max] x 2*coeff_sum[n_max+1]+2)
def C_21_coax(r_pq, n_max):
    
    size_quad = coeff_sum[n_max]
    C_21 = np.zeros((size_quad, size_quad), dtype=complex)
    
    for n in range(1, n_max + 1):
        g, f = coeff_sum[n-1], coeff_sum[n]
        m = np.arange(-n, n + 1)
        
        A = (1j * m * r_pq) / (n * (n + 1))
        C_21[g:f, g:f] = np.diag(A)
    
    column_0 = np.zeros((coeff_sum[n_max], 1), dtype=complex)
    column_max = np.zeros((coeff_sum[n_max], coeff_number[n_max + 1]), dtype=complex)
    
    return np.hstack([column_0, C_21, column_max])



def C_11_coax(r_pq, n_max):
    
    C_11 = np.zeros((coeff_sum[n_max+1], coeff_sum[n_max+1]), dtype=complex)
    
    for n in range(1, n_max+1):
        g, f = coeff_sum[n-1], coeff_sum[n]
        m = np.arange(-n, n + 1)
        C_11[g:f, g:f] = np.eye(coeff_number[n])
        
        BC = (a_nm(n, m) * k * r_pq)/(n+1)
        
        C_11[g:f, g+coeff_number[n]+1:f+coeff_number[n]+1] = np.diag(BC)
        
        C_11[g+coeff_number[n]+1:f+coeff_number[n]+1, g:f] = np.diag(BC)
    
    column_0 = np.zeros((coeff_sum[n_max], 1), dtype=complex)
    column_0[1, 0] = (a_nm(0, 0) * k * r_pq)/(1)
    
    return np.hstack([column_0, C_11[:coeff_sum[n_max],:]])







def U_pq(r_pq, theta_pq, phi_pq, chi_pq, n_max):
    C11 =  C_11_coax(r_pq, n_max)
    C21 = C_21_coax(r_pq, n_max)
    SR = SR_pq_coax(r_pq, n_max)
    R = RotQ(theta_pq, phi_pq, chi_pq, n_max)
    
    RotQ1 = block_diag(R, R)
    SR_coax = block_diag(SR, SR)
    C_coax = np.block([[C11, k**2 * C21],
                       [C21,        C11]])
    inv_RotQ1 = RotQ1.conj().T
    
    return inv_RotQ1 @ C_coax @ SR_coax @ RotQ1



U12 = U_pq(r12, theta12, 0, chi12, n_max)
U21 = U_pq(r21, theta21, 0, chi21, n_max)

U13 = U_pq(r13, theta13, 0, chi13, n_max)
U31 = U_pq(r31, theta31, 0, chi31, n_max)

U14 = U_pq(r14, theta14, 0, chi14, n_max)
U41 = U_pq(r41, theta41, 0, chi41, n_max)

U23 = U_pq(r23, theta23, 0, chi23, n_max)
U32 = U_pq(r32, theta32, 0, chi32, n_max)

U24 = U_pq(r24, theta24, 0, chi24, n_max)
U42 = U_pq(r42, theta42, 0, chi42, n_max)

U34 = U_pq(r34, theta34, 0, chi34, n_max)
U43 = U_pq(r43, theta43, 0, chi43, n_max)



#Matrix L = (I - T @ U)
I = np.identity(T1.shape[0], dtype=complex)

L = np.block([[         I, - T1 @ U21, - T1 @ U31, - T1 @ U41],
              [- T2 @ U12,          I, - T2 @ U32, - T2 @ U42],
              [- T3 @ U13, - T3 @ U23,          I, - T3 @ U43],
              [- T4 @ U14, - T4 @ U24, - T4 @ U34,          I]])


 





#Einfallende Welle Y = T @ y für s = e_z
def c_nm(n, m):
    cond = (m >= -n) & (m <= n)
    rad = np.where(cond, (n - m)*(n + m + 1), 0).astype(float)
    val = np.sqrt(rad)
    return np.where(m >= 0, val, -val)

        
def y_q_inc(n_max):
    n = np.concatenate([np.full(2*o+1, o) for o in range(1, n_max+1)]).reshape(-1, 1)
    m = np.concatenate([np.arange(-o, o+1) for o in range(1, n_max+1)]).reshape(-1, 1)

    c_1 = c_nm(n, m)
    c_2 = c_nm(n, m-1)
    Y_1 = Y_nm(n, -m-1, 0, 0)
    Y_2 = Y_nm(n, -m+1, 0, 0)
    Y_3 = Y_nm(n, -m, 0, 0)

    A_phi = - (4*np.pi*1j**(n+1))/(n*(n+1))
    y_phi = A_phi * (1/2 *((p_x + 1j*p_y)*c_1*Y_1 + (p_x - 1j*p_y)*c_2*Y_2) - (m * p_z * Y_3))

    A_psi = - (4*np.pi*1j**n)/(n*(n+1)*k)
    y_psi = A_psi * (1/2 *((q_x + 1j*q_y)*c_1*Y_1 + (q_x - 1j*q_y)*c_2*Y_2) - (m * q_z * Y_3))

    return np.concatenate([y_phi, y_psi])

y_q = y_q_inc(n_max)




#Phasenfaktor 1 im globalen Ursprung
def phase_factor(x, y, z):
    return np.exp(1j * k * (s_x * x + s_y * y + s_z * z))

Y = np.block([[phase_factor(x1, y1, z1) * T1 @ y_q],
              [phase_factor(x2, y2, z2) * T2 @ y_q],
              [phase_factor(x3, y3, z3) * T3 @ y_q],
              [phase_factor(x4, y4, z4) * T4 @ y_q]])





X_val = np.linalg.solve(L, Y)

X_flat = X_val.flatten()
X1, X2, X3, X4 = np.split(X_flat, 4)

print(f"Radien der Kugeln: a1={a(h, 1):.1e} a2={a(h, 2):.1e} a3={a(h, 3):.1e} a4={a(h, 4):.1e} für h={h}")
print(rf"X_{pol_mode}(h={h})=")
print(X_val)








#Debye-Potentiale und Elektrisches & magnetisches Feld
# S_nm = h_n * Y_nm,   M_nm = ∇S_nm × r,    N_nm = ∇ × M_nm
# E = 𝚺 φ_nm * M_nm + ψ_nm * N_nm,    H = 1/iωμ * 𝚺 k**2 * ψ_nm * M_nm + φ_nm * N_nm

def potentials_fields_scat_q(r, theta, chi, X_q):
    phi, psi, E_r, E_t, E_c, H_r, H_t, H_c, idx = 0, 0, 0, 0, 0, 0, 0, 0, 0
    offset = coeff_sum[n_max]
    x = k*r
    sin_t, sin_c = np.sin(theta), np.sin(chi)
    cos_t, cos_c = np.cos(theta), np.cos(chi)
    inv_r, inv_sin_t = 1/r, 1/sin_t
    H_const = (np.sqrt(epsilon_0 * mu_0)) / (1j * k * mu_0)
    
    for n in range(1, n_max+1):
        h_n = spherical_hn1(n, x)
        dh_n = (n*spherical_hn1(n-1, x) - (n+1) * spherical_hn1(n+1, x)) / (2*n+1)
        P  = {j: lpmv(j, n, cos_t) for j in range(-1, n + 2)}
        dP = {j: 1/2 * P[j+1] - 1/2 * (n + j) * (n - j + 1) * P[j-1] for j in range(0, n + 1)}
        a = (n * (n + 1) / r)
        
        for m in range(-n, n+1):
            abs_m = np.abs(m)
            norm = np.sqrt(((2*n + 1)/(4*np.pi)) * (factorial(n - abs_m)/factorial(n + abs_m)))
            exp_chi = np.exp(1j * m * chi)
            Y_n_m = (-1.0)**m * norm * P[abs_m] * exp_chi
            dY_dchi = 1j * m * Y_n_m
            dY_dtheta = (-1.0)**m * norm * dP[abs_m] * exp_chi
            
            X_q_phi = X_q[idx]
            X_q_psi = X_q[offset+idx]
            
            S_nm = h_n * Y_n_m
            
            phi += X_q_phi * S_nm
            psi += X_q_psi * S_nm
            
            M_nm_t = inv_sin_t * h_n * dY_dchi
            M_nm_c = -h_n * dY_dtheta
            
            N_nm_r = a * h_n * Y_n_m
            N_nm_t = inv_r * dY_dtheta * (h_n + x*dh_n)
            N_nm_c = inv_r * inv_sin_t * dY_dchi * (h_n + x*dh_n)
            
            E_r += X_q_psi * N_nm_r
            E_t += X_q_phi * M_nm_t + X_q_psi * N_nm_t
            E_c += X_q_phi * M_nm_c + X_q_psi * N_nm_c
            
            H_r += H_const * (X_q_phi * N_nm_r)
            H_t += H_const * (k**2 * X_q_psi * M_nm_t + X_q_phi * N_nm_t)
            H_c += H_const * (k**2 * X_q_psi * M_nm_c + X_q_phi * N_nm_c)
            
            idx = idx + 1
            
    E_x = E_r * sin_t * cos_c + E_t * cos_t * cos_c - E_c * sin_c
    E_y = E_r * sin_t * sin_c + E_t * cos_t * sin_c + E_c * cos_c
    E_z = E_r * cos_t - E_t * sin_t
    
    H_x = H_r * sin_t * cos_c + H_t * cos_t * cos_c - H_c * sin_c
    H_y = H_r * sin_t * sin_c + H_t * cos_t * sin_c + H_c * cos_c
    H_z = H_r * cos_t - H_t * sin_t
    
    return phi, psi, np.array([E_x, E_y, E_z]), np.array([H_x, H_y, H_z])


 


#Einfallende Felder für s = e_z
def fields_inc(q_x, q_y, k, z):
    phase = np.exp(1j*k*z)
    const = 1/(c_0 * mu_0)
    
    E_inc_x = -q_y * phase
    E_inc_y = q_x * phase
    E_inc_z = np.zeros_like(E_inc_x)
    
    H_inc_x = -q_x * phase * const
    H_inc_y = -q_y * phase * const
    H_inc_z = np.zeros_like(H_inc_x)
    
    return np.array([E_inc_x, E_inc_y, E_inc_z]), np.array([H_inc_x, H_inc_y, H_inc_z])








# 3D Gitter und Berechnungen
x_range = np.linspace(-limit, limit, res)
y_range = np.linspace(-limit, limit, res)
z_range = np.linspace(-limit, limit, res)

# 'ij' indexing sorgt dafür, dass X_grid[i,j,k] zu x_range[i] gehört
X_grid, Y_grid, Z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')


x_prime_1 = X_grid - x1
y_prime_1 = Y_grid - y1
z_prime_1 = Z_grid - z1
r_prime_1_REAL = np.sqrt(x_prime_1**2 + y_prime_1**2 + z_prime_1**2)
r_prime_1 = np.where(r_prime_1_REAL < a(h, 1), np.nan, r_prime_1_REAL)
theta_prime_1 = np.arccos(np.clip(z_prime_1 / r_prime_1_REAL, -1.0, 1.0))
chi_prime_1 = np.arctan2(y_prime_1, x_prime_1)

x_prime_2 = X_grid - x2
y_prime_2 = Y_grid - y2
z_prime_2 = Z_grid - z2
r_prime_2_REAL = np.sqrt(x_prime_2**2 + y_prime_2**2 + z_prime_2**2)
r_prime_2 = np.where(r_prime_2_REAL < a(h, 2), np.nan, r_prime_2_REAL)
theta_prime_2 = np.arccos(np.clip(z_prime_2 / r_prime_2_REAL, -1.0, 1.0))
chi_prime_2 = np.arctan2(y_prime_2, x_prime_2)

x_prime_3 = X_grid - x3
y_prime_3 = Y_grid - y3
z_prime_3 = Z_grid - z3
r_prime_3_REAL = np.sqrt(x_prime_3**2 + y_prime_3**2 + z_prime_3**2)
r_prime_3 = np.where(r_prime_3_REAL < a(h, 3), np.nan, r_prime_3_REAL)
theta_prime_3 = np.arccos(np.clip(z_prime_3 / r_prime_3_REAL, -1.0, 1.0))
chi_prime_3 = np.arctan2(y_prime_3, x_prime_3)

x_prime_4 = X_grid - x4
y_prime_4 = Y_grid - y4
z_prime_4 = Z_grid - z4
r_prime_4_REAL = np.sqrt(x_prime_4**2 + y_prime_4**2 + z_prime_4**2)
r_prime_4 = np.where(r_prime_4_REAL < a(h, 4), np.nan, r_prime_4_REAL)
theta_prime_4 = np.arccos(np.clip(z_prime_4 / r_prime_4_REAL, -1.0, 1.0))
chi_prime_4 = np.arctan2(y_prime_4, x_prime_4)



#Streupotentiale und Felder
phi_scat1, psi_scat1, E_scat1, H_scat1 = potentials_fields_scat_q(r_prime_1, theta_prime_1, chi_prime_1, X1)
phi_scat2, psi_scat2, E_scat2, H_scat2 = potentials_fields_scat_q(r_prime_2, theta_prime_2, chi_prime_2, X2)
phi_scat3, psi_scat3, E_scat3, H_scat3 = potentials_fields_scat_q(r_prime_3, theta_prime_3, chi_prime_3, X3)
phi_scat4, psi_scat4, E_scat4, H_scat4 = potentials_fields_scat_q(r_prime_4, theta_prime_4, chi_prime_4, X4)


phi_scat = phi_scat1 + phi_scat2 + phi_scat3 + phi_scat4
psi_scat = psi_scat1 + psi_scat2 + psi_scat3 + psi_scat4


E_scat = E_scat1 + E_scat2 + E_scat3 + E_scat4
H_scat = H_scat1 + H_scat2 + H_scat3 + H_scat4

E_mag_3D = np.sqrt(np.abs(E_scat[0])**2 + np.abs(E_scat[1])**2 + np.abs(E_scat[2])**2)
H_mag_3D = np.sqrt(np.abs(H_scat[0])**2 + np.abs(H_scat[1])**2 + np.abs(H_scat[2])**2)


E_inc, H_inc = fields_inc(q_x, q_y, k, Z_grid)

E_tot, H_tot = E_scat + E_inc, H_scat + H_inc



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
def f_q_n(z, k, X_q):
    theta, chi = 1e-5, 0
    f_r, f_theta, f_chi, idx = 0, 0, 0, 0
    
    for n in range(1, n_max+1):
        for m in range(-n, n+1):
            dY_dchi = 1j * m * Y_nm(n, m, theta, chi)
            dY_dtheta = dY_nm_dtheta(n, m, theta, chi)
            
            t_phi = X_q[idx] * (1j**(-n-1))/k * 1/(np.sin(theta)) * dY_dchi
            t_psi = X_q[coeff_sum[n_max] + idx] * 1j**(-n) * dY_dtheta
            
            f_theta = f_theta + t_phi + t_psi
            
            c_phi = X_q[idx] * (1j**(-n-1))/k * (-dY_dtheta)
            c_psi = X_q[coeff_sum[n_max] + idx] * 1j**(-n) * 1/(np.sin(theta)) * dY_dchi
            
            f_chi = f_chi + c_phi + c_psi
            
            idx = idx + 1
            
    f_x = f_theta * np.cos(chi) - f_chi * np.sin(chi)
    f_y = f_theta * np.sin(chi) + f_chi * np.cos(chi)
    f_z = f_r
            
    return np.exp(-1j*k*z) * np.array([f_x, f_y, f_z])

f = f_q_n(z1, k, X1) + f_q_n(z2, k, X2) + f_q_n(z3, k, X3) + f_q_n(z4, k, X4)
f_x, f_y, f_z = f     

sigma = (4*np.pi)/k * np.imag( np.conj(p_x)*f_x + np.conj(p_y)*f_y + np.conj(p_z)*f_z )
print(f'Wirkungsquerschnitt: sigma = {sigma}')







        
        

#Plotten
def plot_2d(X, Y, data, title, xlabel, ylabel, cbar_label, cmap='cividis', norm=None, figsize=(6, 4.5)):
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=12)
    cf = plt.contourf(X, Y, data, levels=400, cmap=cmap, norm=norm)
    plt.colorbar(cf, label=cbar_label, fraction=0.225, pad=0.03, format='%.4e')   
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.axis('equal')
    plt.show()


#Plotten in der x-z-Ebene mit y = 0
if XZ:
    y_idx = res // 2
    y_xz = y_range[y_idx]
    X_xz, Z_xz = X_grid[:, y_idx, :].T, Z_grid[:, y_idx, :].T
    xl, yl = '$x$ (m)', '$z$ (m)'
    
    if PHI_PSI:
        plot_2d(X_xz, Z_xz, np.abs(phi_scat[:, y_idx, :]).T, 
                fr'{pol_mode}: |$\phi_{{\mathrm{{scat}}}}$| in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, r'|$\phi_{\mathrm{scat}}$| (V/m)', 'cividis')
        
        plot_2d(X_xz, Z_xz, np.abs(psi_scat[:, y_idx, :]).T, 
                fr'{pol_mode}: |$\psi_{{\mathrm{{scat}}}}$| in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, r'|$\psi_{\mathrm{scat}}$| (V)', 'cividis')

    if E_H:
        plot_2d(X_xz, Z_xz, E_mag_3D[:, y_idx, :].T, 
                fr'{pol_mode}: |$\mathbf{{E}}_{{\mathrm{{scat}}}}$| in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, r'|$\mathbf{E}_{\mathrm{scat}}$| (V/m)', 'viridis')
        
        
        plot_2d(X_xz, Z_xz, H_mag_3D[:, y_idx, :].T, 
                fr'{pol_mode}: |$\mathbf{{H}}_{{\mathrm{{scat}}}}$| in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, r'|$\mathbf{H}_{\mathrm{scat}}$| (A/m)', 'viridis')

    if Im_EH_tot:
        plot_2d(X_xz, Z_xz, Im_E_dot_H_star[:, y_idx, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{tot}}}} \cdot \mathbf{{H}}_{{\mathrm{{tot}}}}^*$) in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma')
        
        

    if Im_EH_all:
        plot_2d(X_xz, Z_xz, Im_E_dot_H_star_inc[:, y_idx, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma')
        
        plot_2d(X_xz, Z_xz, Im_E_dot_H_star_scat[:, y_idx, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())
        
        plot_2d(X_xz, Z_xz, Im_E_dot_H_star_mixed1[:, y_idx, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())
        
        plot_2d(X_xz, Z_xz, Im_E_dot_H_star_mixed2[:, y_idx, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) in x-z (y={y_xz:.1e} m), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())





#Plotten in der y-z-Ebene mit x = 0
if YZ:
    x_idx = res // 2
    x_yz = x_range[x_idx]
    Y_yz, Z_yz = Y_grid[x_idx, :, :].T, Z_grid[x_idx, :, :].T
    xl, yl = '$y$ (m)', '$z$ (m)'

    if PHI_PSI:
        plot_2d(Y_yz, Z_yz, np.abs(phi_scat[x_idx, :, :]).T, 
                fr'{pol_mode}: |$\phi_{{\mathrm{{scat}}}}$| in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, r'|$\phi_{\mathrm{scat}}$| (V/m)', 'cividis')
        
        plot_2d(Y_yz, Z_yz, np.abs(psi_scat[x_idx, :, :]).T, 
                fr'{pol_mode}: |$\psi_{{\mathrm{{scat}}}}$| in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, r'|$\psi_{\mathrm{scat}}$| (V)', 'cividis')

    if E_H:
        plot_2d(Y_yz, Z_yz, E_mag_3D[x_idx, :, :].T, 
                fr'{pol_mode}: |$\mathbf{{E}}_{{\mathrm{{scat}}}}$| in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, r'|$\mathbf{E}_{\mathrm{scat}}$| (V/m)', 'viridis')
        
        plot_2d(Y_yz, Z_yz, H_mag_3D[x_idx, :, :].T, 
                fr'{pol_mode}: |$\mathbf{{H}}_{{\mathrm{{scat}}}}$| in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, r'|$\mathbf{H}_{\mathrm{scat}}$| (A/m)', 'viridis')

    if Im_EH_tot:
        plot_2d(Y_yz, Z_yz, Im_E_dot_H_star[x_idx, :, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{tot}}}} \cdot \mathbf{{H}}_{{\mathrm{{tot}}}}^*$) in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma')
        
    if Im_EH_all:
        plot_2d(Y_yz, Z_yz, Im_E_dot_H_star_inc[x_idx, :, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma')
        
        plot_2d(Y_yz, Z_yz, Im_E_dot_H_star_scat[x_idx, :, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())
        
        plot_2d(Y_yz, Z_yz, Im_E_dot_H_star_mixed1[x_idx, :, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())
        
        plot_2d(Y_yz, Z_yz, Im_E_dot_H_star_mixed2[x_idx, :, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) in y-z (x={x_yz:.1e} m), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm())





#Plotten auf der Diagonalebene entlang z mit x=y
if DIAGONAL:
    indices = np.arange(res)
    X_diag = X_grid[indices, indices, :].T
    Y_diag = Y_grid[indices, indices, :].T
    Z_diag = Z_grid[indices, indices, :].T
    D_diag = X_diag * np.sqrt(2)
    xl, yl = '$d$ (m)', '$z$ (m)'
    fig_sz = (8, 4.5)

    if PHI_PSI:
        plot_2d(D_diag, Z_diag, np.abs(phi_scat[indices, indices, :]).T, 
                fr'{pol_mode}: |$\phi_{{\mathrm{{scat}}}}$| (x=y), h={h}', 
                xl, yl, r'|$\phi_{\mathrm{scat}}$| (V/m)', 'cividis', figsize=fig_sz)
        
        plot_2d(D_diag, Z_diag, np.abs(psi_scat[indices, indices, :]).T, 
                fr'{pol_mode}: |$\psi_{{\mathrm{{scat}}}}$| (x=y), h={h}', 
                xl, yl, r'|$\psi_{\mathrm{scat}}$| (V)', 'cividis', figsize=fig_sz)

    if E_H:
        plot_2d(D_diag, Z_diag, E_mag_3D[indices, indices, :].T, 
                fr'{pol_mode}: |$\mathbf{{E}}_{{\mathrm{{scat}}}}$| (x=y), h={h}', 
                xl, yl, r'|$\mathbf{E}_{\mathrm{scat}}$| (V/m)', 'viridis', figsize=fig_sz)
        plot_2d(D_diag, Z_diag, H_mag_3D[indices, indices, :].T, 
                fr'{pol_mode}: |$\mathbf{{H}}_{{\mathrm{{scat}}}}$| (x=y), h={h}', 
                xl, yl, r'|$\mathbf{H}_{\mathrm{scat}}$| (A/m)', 'viridis', figsize=fig_sz)

    if Im_EH_tot:
        plot_2d(D_diag, Z_diag, Im_E_dot_H_star[indices, indices, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{tot}}}} \cdot \mathbf{{H}}_{{\mathrm{{tot}}}}^*$) (x=y), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', figsize=fig_sz)
        
    if Im_EH_all:
        plot_2d(D_diag, Z_diag, Im_E_dot_H_star_inc[indices, indices, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) (x=y), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', figsize=fig_sz)
        
        plot_2d(D_diag, Z_diag, Im_E_dot_H_star_scat[indices, indices, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) (x=y), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm(), figsize=fig_sz)
        
        plot_2d(D_diag, Z_diag, Im_E_dot_H_star_mixed1[indices, indices, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{scat}}}} \cdot \mathbf{{H}}_{{\mathrm{{inc}}}}^*$) (x=y), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm(), figsize=fig_sz)
        
        plot_2d(D_diag, Z_diag, Im_E_dot_H_star_mixed2[indices, indices, :].T, 
                fr'{pol_mode}: Im($\mathbf{{E}}_{{\mathrm{{inc}}}} \cdot \mathbf{{H}}_{{\mathrm{{scat}}}}^*$) (x=y), h={h}', 
                xl, yl, r'Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 'plasma', norm=CenteredNorm(), figsize=fig_sz)

