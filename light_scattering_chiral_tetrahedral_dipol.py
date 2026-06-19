
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.special import spherical_jn, spherical_yn, lpmv, factorial

def spherical_hn1(n, z):
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)

def d_spherical_hn1(n, z):
    return ( n*spherical_hn1(n-1, z) - (n+1) * spherical_hn1(n+1, z) ) / (2*n+1)

def Y_nm(n, m, theta, chi):
    abs_m = np.abs(m)
    norm = np.sqrt(((2*n + 1)/(4*np.pi)) * (factorial(n - abs_m)/factorial(n + abs_m)))
        
    return (-1.0) ** m * norm * lpmv(abs_m, n, np.cos(theta)) * np.exp(1j * m * chi)


#3D Gitter
res = 111 
limit = 4*10**-6

#Streuergeometrie, theta [0, pi], chi [-pi, pi]
c = 1.25*10**-6 
h = 4

x1, y1, z1 = -c,  c, -c
x2, y2, z2 =  c, -c, -c
x3, y3, z3 =  c,  c,  c
x4, y4, z4 = -c, -c,  c

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
#pol_mode = "RCP" 
#q_x = -1j/np.sqrt(2)  
#q_y = -1/np.sqrt(2)
#q_z = 0

p_x = s_y*q_z-s_z*q_y
p_y = s_z*q_x-s_x*q_z
p_z = s_x*q_y-s_y*q_x
print('Polarisation:',p_x, p_y, p_z)

#Streusystem
def w(h, i):
    return (30+(2*h)*(i-1.5))*10**-9

def a_1(h):
    return w(h, 0)
def a_2(h): 
    return w(h, 1)
def a_3(h):
    return w(h, 2)
def a_4(h):
    return w(h, 3)


#Konstanten
epsilon_0 = 8.854*10**-12
mu_0 = 4*np.pi*10**-7
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











#Mie Koeffizienten und T-Matrix
def inv_T_phi(a_q):
    return 1/(-1j * (2/3) * (k*a_q)**3)

def inv_T_psi(a_q):
    return 1 / (+1j * (1/3) * (k*a_q)**3)

def inv_T_q(a_q):
    return np.array([
        [inv_T_phi(a_q), 0, 0, 0, 0, 0], 
        [0, inv_T_phi(a_q), 0, 0, 0, 0], 
        [0, 0, inv_T_phi(a_q), 0, 0, 0], 
        [0, 0, 0, inv_T_psi(a_q), 0, 0],
        [0, 0, 0, 0, inv_T_psi(a_q), 0], 
        [0, 0, 0, 0, 0, inv_T_psi(a_q)]])








#Rotationsmatrix (keine Rotation für theta=0, chi=pi/2, phi=0)
def RotQ(theta_pq, chi_pq, phi_pq):
    T_m1_m1_1 = np.exp(-1j * chi_pq) * np.exp(1j * (phi_pq+np.pi/2)) * (-np.cos(theta_pq/2)**2)
    T_0_m1_1 = np.exp(-1j * chi_pq) * (-np.sin(theta_pq) / np.sqrt(2))
    T_1_m1_1 = np.exp(-1j * chi_pq) * np.exp(-1j * (phi_pq+np.pi/2)) * (-np.sin(theta_pq/2)**2)
    
    T_m1_0_1 = np.exp(1j * (phi_pq+np.pi/2)) * (-np.sin(theta_pq) / np.sqrt(2))
    T_0_0_1 = np.cos(theta_pq)
    T_1_0_1 = np.exp(-1j * (phi_pq+np.pi/2)) * (np.sin(theta_pq) / np.sqrt(2))
    
    T_m1_1_1 = np.exp(1j * chi_pq) * np.exp(1j * (phi_pq+np.pi/2)) * (-np.sin(theta_pq/2)**2)
    T_0_1_1 = np.exp(1j * chi_pq) * (np.sin(theta_pq) / np.sqrt(2))
    T_1_1_1 = np.exp(1j * chi_pq) * np.exp(-1j * (phi_pq+np.pi/2)) * (-np.cos(theta_pq/2)**2)  
    return np.array([
        [T_m1_m1_1,  T_0_m1_1,  T_1_m1_1],
        [T_m1_0_1,   T_0_0_1,   T_1_0_1],
        [T_m1_1_1,   T_0_1_1,   T_1_1_1]])

#Kontrolle RotQ
RotQ_test = RotQ(4, 12, 0)
det = np.linalg.det(RotQ_test)
print(f"Determinante von RotQ : {det}")


def inv_RotQ(theta_pq, chi_pq, phi_pq):
    return RotQ(theta_pq, chi_pq, phi_pq).conj().T



#coaxiale Conversion Operatoren 
def C11_coax(r_pq):
    return np.array([[1, 0, 0], 
                     [0, 1, 0], 
                     [0, 0, 1]])

def C21_pq_coax(r_pq):
    return np.array([[(-1j*r_pq)/2, 0,         0], 
                     [0,    	    0,         0], 
                     [0,            0, (1j*r_pq)/2]])

#allgemeine Conversion Operatoren
def C11_pq(r_pq, theta_pq, chi_pq, phi_pq):
    arg = (theta_pq, chi_pq, phi_pq)
    return inv_RotQ(*arg) @ C11_coax(r_pq) @ RotQ(*arg)

def C21_pq(r_pq, theta_pq, chi_pq, phi_pq):
    arg = (theta_pq, chi_pq, phi_pq)
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
def SR_pq(r_pq, theta_pq, chi_pq, phi_pq):
    arg = (theta_pq, chi_pq, phi_pq)
    return inv_RotQ(*arg) @ SR_pq_coax(r_pq) @ RotQ(*arg)


def U_pq(r_pq, theta_pq, chi_pq, phi_pq):
    arg = (r_pq, theta_pq, chi_pq, phi_pq)
    A = C11_pq(*arg) @ SR_pq(*arg)
    B = C21_pq(*arg) @ SR_pq(*arg)
    return np.block([[A, k**2 * B],
                     [B,        A]])


def inv_T1(h):
    return inv_T_q(a_1(h))
def inv_T2(h):
    return inv_T_q(a_2(h))
def inv_T3(h):
    return inv_T_q(a_3(h))
def inv_T4(h):
    return inv_T_q(a_4(h))

U12 = U_pq(r12, theta12, chi12, 0)
U21 = U_pq(r21, theta21, chi21, 0)

U13 = U_pq(r13, theta13, chi13, 0)
U31 = U_pq(r31, theta31, chi31, 0)

U14 = U_pq(r14, theta14, chi14, 0)
U41 = U_pq(r41, theta41, chi41, 0)

U23 = U_pq(r23, theta23, chi23, 0)
U32 = U_pq(r32, theta32, chi32, 0)

U24 = U_pq(r24, theta24, chi24, 0)
U42 = U_pq(r42, theta42, chi42, 0)

U34 = U_pq(r34, theta34, chi34, 0)
U43 = U_pq(r43, theta43, chi43, 0)


#Matrix L (24x24)
def L(h):
    return np.block([[inv_T1(h), -U12,   -U13,   -U14],
                     [-U21,   inv_T2(h), -U23,   -U24],
                     [-U31,   -U32,   inv_T3(h), -U34],
                     [-U41,   -U42,   -U43,   inv_T4(h)]])






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



print(f"Radien der Kugeln: a1={a_1(h):.1e} a2={a_2(h):.1e} a3={a_3(h):.1e} a4={a_4(h):.1e} für h={h}")
print('Produkt für Dipolnäherung: ka4=',k*a_4(h))
print(rf"X_{pol_mode}(h={h})=")
print(X(h))



#Eine Komponente von X in Abhängigkeit von h plotten
xAchse = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
yAchse = np.array([abs_X(0)[0], 
              abs_X(1)[0], 
              abs_X(2)[0], 
              abs_X(3)[0], 
              abs_X(4)[0], 
              abs_X(5)[0], 
              abs_X(6)[0],
              abs_X(7)[0],
              abs_X(8)[0]])


plt.scatter(xAchse, yAchse, label="Messdaten", color="blue")

yAchse = np.array(yAchse).flatten()
grad_fit = 3
poly = np.poly1d(np.polyfit(xAchse, yAchse, grad_fit))

x_poly = np.linspace(min(xAchse), max(xAchse), 200)
y_poly = poly(x_poly)

plt.plot(x_poly, y_poly, label="Polynom 3. Ordnung", linewidth=2)
plt.xlabel("h")
plt.ylabel("X[0]")
plt.title("Eine Komponente von X in Abhängigkeit von h")
plt.legend()
plt.grid(True)
plt.show()








#Berechnung des Potentials phi(vec(r))
def S_m1_1(r_prime_p, theta_prime_p, chi_prime_p):
    return spherical_hn1(1, k*r_prime_p)* Y_nm(1, -1, theta_prime_p, chi_prime_p)
def S_0_1(r_prime_p, theta_prime_p, chi_prime_p):
    return spherical_hn1(1, k*r_prime_p)* Y_nm(1, 0, theta_prime_p, chi_prime_p)
def S_1_1(r_prime_p, theta_prime_p, chi_prime_p):
    return spherical_hn1(1, k*r_prime_p)* Y_nm(1, 1, theta_prime_p, chi_prime_p)

X_val=X(h)


def phi_scat_1(r_prime_1, theta_prime_1, chi_prime_1):
    arg = (r_prime_1, theta_prime_1, chi_prime_1)
    return X_val.flatten()[0]*S_m1_1(*arg) + X_val.flatten()[1]*S_0_1(*arg) + X_val.flatten()[2]*S_1_1(*arg)

def psi_scat_1(r_prime_1, theta_prime_1, chi_prime_1):
    arg = (r_prime_1, theta_prime_1, chi_prime_1)
    return X_val.flatten()[3]*S_m1_1(*arg) + X_val.flatten()[4]*S_0_1(*arg) + X_val.flatten()[5]*S_1_1(*arg)


def phi_scat_2(r_prime_2, theta_prime_2, chi_prime_2):
    arg = (r_prime_2, theta_prime_2, chi_prime_2)
    return X_val.flatten()[6]*S_m1_1(*arg) + X_val.flatten()[7]*S_0_1(*arg) + X_val.flatten()[8]*S_1_1(*arg)

def psi_scat_2(r_prime_2, theta_prime_2, chi_prime_2):
    arg = (r_prime_2, theta_prime_2, chi_prime_2)
    return X_val.flatten()[9]*S_m1_1(*arg) + X_val.flatten()[10]*S_0_1(*arg) + X_val.flatten()[11]*S_1_1(*arg)


def phi_scat_3(r_prime_3, theta_prime_3, chi_prime_3):
    arg = (r_prime_3, theta_prime_3, chi_prime_3)
    return X_val.flatten()[12]*S_m1_1(*arg) + X_val.flatten()[13]*S_0_1(*arg) + X_val.flatten()[14]*S_1_1(*arg)

def psi_scat_3(r_prime_3, theta_prime_3, chi_prime_3):
    arg = (r_prime_3, theta_prime_3, chi_prime_3)
    return X_val.flatten()[15]*S_m1_1(*arg) + X_val.flatten()[16]*S_0_1(*arg) + X_val.flatten()[17]*S_1_1(*arg)


def phi_scat_4(r_prime_4, theta_prime_4, chi_prime_4):
    arg = (r_prime_4, theta_prime_4, chi_prime_4)
    return X_val.flatten()[18]*S_m1_1(*arg) + X_val.flatten()[19]*S_0_1(*arg) + X_val.flatten()[20]*S_1_1(*arg)

def psi_scat_4(r_prime_4, theta_prime_4, chi_prime_4):
    arg = (r_prime_4, theta_prime_4, chi_prime_4)
    return X_val.flatten()[21]*S_m1_1(*arg) + X_val.flatten()[22]*S_0_1(*arg) + X_val.flatten()[23]*S_1_1(*arg)




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
r_prime_1 = np.clip(r_prime_1_REAL, 3*a_1(h), None)
theta_prime_1 = np.arccos(np.clip(z_prime_1 / r_prime_1_REAL, -1.0, 1.0))
chi_prime_1 = np.arctan2(y_prime_1, x_prime_1)

x_prime_2 = X_grid - x2
y_prime_2 = Y_grid - y2
z_prime_2 = Z_grid - z2
r_prime_2_REAL = np.sqrt(x_prime_2**2 + y_prime_2**2 + z_prime_2**2)
r_prime_2 = np.clip(r_prime_2_REAL, 3*a_2(h), None)
theta_prime_2 = np.arccos(np.clip(z_prime_2 / r_prime_2_REAL, -1.0, 1.0))
chi_prime_2 = np.arctan2(y_prime_2, x_prime_2)

x_prime_3 = X_grid - x3
y_prime_3 = Y_grid - y3
z_prime_3 = Z_grid - z3
r_prime_3_REAL = np.sqrt(x_prime_3**2 + y_prime_3**2 + z_prime_3**2)
r_prime_3 = np.clip(r_prime_3_REAL, 3*a_3(h), None)
theta_prime_3 = np.arccos(np.clip(z_prime_3 / r_prime_3_REAL, -1.0, 1.0))
chi_prime_3 = np.arctan2(y_prime_3, x_prime_3)

x_prime_4 = X_grid - x4
y_prime_4 = Y_grid - y4
z_prime_4 = Z_grid - z4
r_prime_4_REAL = np.sqrt(x_prime_4**2 + y_prime_4**2 + z_prime_4**2)
r_prime_4 = np.clip(r_prime_4_REAL, 3*a_4(h), None)
theta_prime_4 = np.arccos(np.clip(z_prime_4 / r_prime_4_REAL, -1.0, 1.0))
chi_prime_4 = np.arctan2(y_prime_4, x_prime_4)


#Streupotentiale
phi_scat_3D = sum([
    phi_scat_1(r_prime_1, theta_prime_1, chi_prime_1),
    phi_scat_2(r_prime_2, theta_prime_2, chi_prime_2),
    phi_scat_3(r_prime_3, theta_prime_3, chi_prime_3),
    phi_scat_4(r_prime_4, theta_prime_4, chi_prime_4)
])

psi_scat_3D = sum([
    psi_scat_1(r_prime_1, theta_prime_1, chi_prime_1),
    psi_scat_2(r_prime_2, theta_prime_2, chi_prime_2),
    psi_scat_3(r_prime_3, theta_prime_3, chi_prime_3),
    psi_scat_4(r_prime_4, theta_prime_4, chi_prime_4)
])



#E-Feld
dx = x_range[1] - x_range[0]
dy = y_range[1] - y_range[0]
dz = z_range[1] - z_range[0]

grad_phi_x, grad_phi_y, grad_phi_z = np.gradient(phi_scat_3D, dx, dy, dz, edge_order=2)
grad_psi_x, grad_psi_y, grad_psi_z = np.gradient(psi_scat_3D, dx, dy, dz, edge_order=2)


# A = ∇φ × r
A_x = grad_phi_y * Z_grid - grad_phi_z * Y_grid
A_y = grad_phi_z * X_grid - grad_phi_x * Z_grid
A_z = grad_phi_x * Y_grid - grad_phi_y * X_grid

# B = ∇ψ × r
B_x = grad_psi_y * Z_grid - grad_psi_z * Y_grid
B_y = grad_psi_z * X_grid - grad_psi_x * Z_grid
B_z = grad_psi_x * Y_grid - grad_psi_y * X_grid

# ∇ × B
dBx_dx, dBx_dy, dBx_dz = np.gradient(B_x, dx, dy, dz, edge_order=2)
dBy_dx, dBy_dy, dBy_dz = np.gradient(B_y, dx, dy, dz, edge_order=2)
dBz_dx, dBz_dy, dBz_dz = np.gradient(B_z, dx, dy, dz, edge_order=2)
curlB_x = dBz_dy - dBy_dz
curlB_y = dBx_dz - dBz_dx
curlB_z = dBy_dx - dBx_dy

Ex_3D = A_x + curlB_x
Ey_3D = A_y + curlB_y
Ez_3D = A_z + curlB_z

E_mag_3D = np.sqrt(np.abs(Ex_3D)**2 + np.abs(Ey_3D)**2 + np.abs(Ez_3D)**2)



# H-Feld
dAx_dx, dAx_dy, dAx_dz = np.gradient(A_x, dx, dy, dz, edge_order=2)
dAy_dx, dAy_dy, dAy_dz = np.gradient(A_y, dx, dy, dz, edge_order=2)
dAz_dx, dAz_dy, dAz_dz = np.gradient(A_z, dx, dy, dz, edge_order=2)

curlAx = dAz_dy - dAy_dz
curlAy = dAx_dz - dAz_dx
curlAz = dAy_dx - dAx_dy

Hx_3D = H_konst * (k**2 * B_x + curlAx)
Hy_3D = H_konst * (k**2 * B_y + curlAy)
Hz_3D = H_konst * (k**2 * B_z + curlAz)

H_mag_3D = np.sqrt(np.abs(Hx_3D)**2 + np.abs(Hy_3D)**2 + np.abs(Hz_3D)**2)




# Im(E H*)
E_dot_H_star = (Ex_3D * np.conj(Hx_3D) +
                Ey_3D * np.conj(Hy_3D) +
                Ez_3D * np.conj(Hz_3D))

Im_E_dot_H_star = np.imag(E_dot_H_star)


abs_Im_E_dot_H = np.abs(Im_E_dot_H_star)
Im_E_dot_H_star_max = np.max(abs_Im_E_dot_H)


flat_max_idx = np.argmax(abs_Im_E_dot_H)
idx_x9, idx_y9, idx_z9 = np.unravel_index(flat_max_idx, abs_Im_E_dot_H.shape)


max_x = X_grid[idx_x9, idx_y9, idx_z9]
max_y = Y_grid[idx_x9, idx_y9, idx_z9]
max_z = Z_grid[idx_x9, idx_y9, idx_z9]

# Ausgabe
print(f'max(abs(Im_E_dot_H_star_3D)) = {Im_E_dot_H_star_max} für h = {h}')
print(f'Ort des Maximums: x = {max_x:.4e}, y = {max_y:.4e}, z = {max_z:.4e}')
print(f'Zugehörige Gitter-Indizes: [{idx_x9}, {idx_y9}, {idx_z9}]')



#Plotten in der x-z-Ebene mit y = 0
y_idx = res // 2
y_xz = y_range[y_idx]
X_xz = X_grid[:, y_idx, :]
Z_xz = Z_grid[:, y_idx, :]

phi_scat_xz = np.abs(phi_scat_3D[:, y_idx, :])
psi_scat_xz = np.abs(psi_scat_3D[:, y_idx, :])

E_mag_xz = E_mag_3D[:, y_idx, :]
H_mag_xz = H_mag_3D[:, y_idx, :]

Im_E_dot_H_star_xz = Im_E_dot_H_star[:, y_idx, :]


plt.figure(figsize=(6, 4.5))
plt.title(fr'{pol_mode}: |$\phi_{{\mathrm{{scat}}}}$| in der x-z-Ebene (y={y_xz:.1e} m) für h={h}', fontsize=12)
cf = plt.contourf(X_xz.T, Z_xz.T, phi_scat_xz.T, levels=400, cmap='cividis')
plt.xlabel('$x$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\phi_{\mathrm{scat}}$|  (V/m)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 4.5))
plt.title(fr'{pol_mode}: |$\psi_{{\mathrm{{scat}}}}$| in der x-z-Ebene (y={y_xz:.1e} m) für h={h}', fontsize=12)
cf = plt.contourf(X_xz.T, Z_xz.T, psi_scat_xz.T, levels=400, cmap='cividis')
plt.xlabel('$x$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\psi_{\mathrm{scat}}$|  (V)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 4.5))
plt.title(fr'{pol_mode}: |$\mathbf{{E}}_{{\mathrm{{scat}}}}$| in der x-z-Ebene (y={y_xz:.1e}) für h={h}', fontsize=12)
cf = plt.contourf(X_xz.T, Z_xz.T, E_mag_xz.T, levels=400, cmap='viridis')
plt.xlabel('$x$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\mathbf{E}_{\mathrm{scat}}$| (V/m)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 4.5))
plt.title(fr'{pol_mode}: |$\mathbf{{H}}_{{\mathrm{{scat}}}}$| in der x-z-Ebene (y={y_xz:.1e}) für h={h}', fontsize=12)
cf = plt.contourf(X_xz.T, Z_xz.T, H_mag_xz.T, levels=400, cmap='viridis')
plt.xlabel('$x$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\mathbf{H}_{\mathrm{scat}}$| (A/m)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 4.5))
cf = plt.contourf(X_xz.T, Z_xz.T, Im_E_dot_H_star_xz.T, levels=400, cmap='plasma')
plt.title(fr'{pol_mode}: Im($\mathbf{{E}} \cdot \mathbf{{H}}^*$) in der x-z-Ebene (y={y_xz:.1e}) für h={h}', fontsize=14)
plt.xlabel('$x$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()








#Plotten in der y-z-Ebene mit x = 0
x_idx = res // 2
x_yz = x_range[x_idx]
Y_yz = Y_grid[x_idx, :, :]
Z_yz = Z_grid[x_idx, :, :]

phi_scat_yz = np.abs(phi_scat_3D[x_idx, :, :])
psi_scat_yz = np.abs(psi_scat_3D[x_idx, :, :])

E_mag_yz = E_mag_3D[x_idx, :, :]
H_mag_yz = H_mag_3D[x_idx, :, :]

Im_E_dot_H_star_yz = Im_E_dot_H_star[x_idx, :, :]

E_mag_LCP_yz=E_mag_yz

plt.figure(figsize=(6, 4.5))
plt.title(fr'{pol_mode}: |$\phi_{{\mathrm{{scat}}}}$| in der y-z-Ebene (x={x_yz:.1e} m) für h={h}', fontsize=12)
cf = plt.contourf(Y_yz.T, Z_yz.T, phi_scat_yz.T, levels=400, cmap='cividis')
plt.xlabel('$y$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\phi_{\mathrm{scat}}$|  (V/m)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 4.5))
plt.title(fr'{pol_mode}: |$\psi_{{\mathrm{{scat}}}}$| in der y-z-Ebene (x={x_yz:.1e} m) für h={h}', fontsize=12)
cf = plt.contourf(Y_yz.T, Z_yz.T, psi_scat_yz.T, levels=400, cmap='cividis')
plt.xlabel('$y$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\psi_{\mathrm{scat}}$|  (V)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 4.5))
plt.title(fr'{pol_mode}: |$\mathbf{{E}}_{{\mathrm{{scat}}}}$| in der y-z-Ebene (x={x_yz:.1e}) für h={h}', fontsize=12)
cf = plt.contourf(Y_yz.T, Z_yz.T, E_mag_yz.T, levels=400, cmap='viridis')
plt.xlabel('$y$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\mathbf{E}_{\mathrm{scat}}$| (V/m)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 4.5))
plt.title(fr'{pol_mode}: |$\mathbf{{H}}_{{\mathrm{{scat}}}}$| in der y-z-Ebene (x={x_yz:.1e}) für h={h}', fontsize=12)
cf = plt.contourf(Y_yz.T, Z_yz.T, H_mag_yz.T, levels=400, cmap='viridis')
plt.xlabel('$y$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\mathbf{H}_{\mathrm{scat}}$| (A/m)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 4.5))
cf = plt.contourf(Y_yz.T, Z_yz.T, Im_E_dot_H_star_yz.T, levels=400, cmap='plasma')
plt.title(fr'{pol_mode}: Im($\mathbf{{E}} \cdot \mathbf{{H}}^*$) in der y-z-Ebene (x={x_yz:.1e}) für h={h}', fontsize=14)
plt.xlabel('$y$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()



















#Plotten auf der Diagonalebene entlang z mit x=y
indices = np.arange(res)
X_diag = X_grid[indices, indices, :]
Y_diag = Y_grid[indices, indices, :]
Z_diag = Z_grid[indices, indices, :]
D_diag = X_diag * np.sqrt(2)

phi_scat_45 = np.abs(phi_scat_3D[indices, indices, :])
psi_scat_45 = np.abs(psi_scat_3D[indices, indices, :])

E_mag_45 = E_mag_3D[indices, indices, :]
H_mag_45 = H_mag_3D[indices, indices, :]

Im_E_dot_H_star_45 = Im_E_dot_H_star[indices, indices, :]

plt.figure(figsize=(8, 4.5))
plt.title(fr'{pol_mode}: |$\phi_{{\mathrm{{scat}}}}$| in der Ebene (x = y) für h={h}', fontsize=12)
cf = plt.contourf(D_diag.T, Z_diag.T, phi_scat_45.T, levels=200, cmap='cividis')
plt.xlabel('$d$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\phi_{\mathrm{scat}}$|  (V/m)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(8, 4.5))
plt.title(fr'{pol_mode}: |$\psi_{{\mathrm{{scat}}}}$| in der Ebene (x = y) für h={h}', fontsize=12)
cf = plt.contourf(D_diag.T, Z_diag.T, psi_scat_45.T, levels=200, cmap='cividis')
plt.xlabel('$d$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\psi_{\mathrm{scat}}$|  (V)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(8, 4.5))
plt.title(fr'{pol_mode}: |$\mathbf{{E}}_{{\mathrm{{scat}}}}$| in der Ebene (x = y) für h={h}', fontsize=12)
cf = plt.contourf(D_diag.T, Z_diag.T, E_mag_45.T, levels=200, cmap='viridis')
plt.xlabel('$d$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\mathbf{E}_{\mathrm{scat}}$| (V/m)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()

plt.figure(figsize=(8, 4.5))
plt.title(fr'{pol_mode}: |$\mathbf{{H}}_{{\mathrm{{scat}}}}$| in der Ebene (x = y) für h={h}', fontsize=12)
cf = plt.contourf(D_diag.T, Z_diag.T, H_mag_45.T, levels=200, cmap='viridis')
plt.xlabel('$d$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(cf, label='|$\mathbf{H}_{\mathrm{scat}}$| (A/m)', fraction=0.225, pad=0.03, format='%.2e')
plt.axis('equal')
plt.show()


plt.figure(figsize=(8, 4.5))
pcm = plt.pcolormesh(D_diag.T, Z_diag.T, Im_E_dot_H_star_45.T, cmap='plasma', 
                     vmin=-6e-7, vmax=6e-7)

plt.title(fr'{pol_mode}: Im($\mathbf{{E}} \cdot \mathbf{{H}}^*$) in der Ebene (x = y) für h={h}', fontsize=14)
plt.xlabel('$d$ (m)', fontsize=12)
plt.ylabel('$z$ (m)', fontsize=12)
cb = plt.colorbar(pcm, label='Im($\mathbf{E} \cdot \mathbf{H}^*$) (W/m²)', 
                  fraction=0.225, pad=0.03, format='%.2e', extend='both')
plt.axis('equal')
plt.show()













#Plotten einer Isofläche des Debye-Potentials in 3D

# 1. 3D-Gitter erstellen
limit = 30 * 10**-6
N = 20  # Auflösung (50x50x50 = 125.000 Punkte)
x_lin = np.linspace(-limit, limit, N)
y_lin = np.linspace(-limit, limit, N)
z_lin = np.linspace(-limit, limit, N)
X3D, Y3D, Z3D = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

# 2. Koordinaten transformation für das gesamte 3D-Volumen
#Nutzt Numpy Broadcasting
r_3d = np.sqrt(X3D**2 + Y3D**2 + Z3D**2)
theta_3d = np.arccos(np.clip(Z3D / r_3d, -1.0, 1.0))
chi_3d = np.arctan2(Y3D, X3D)

# Hilfsfunktion für Abstandsberechnung im 3D-Raum
def get_loc_coords(x_off, y_off, z_off, a_radius):
    x_p = X3D - x_off
    y_p = Y3D - y_off
    z_p = Z3D - z_off
    r_real = np.sqrt(x_p**2 + y_p**2 + z_p**2)
    # Clipping um Singularitäten im Ursprung der Streuer zu vermeiden
    r_p = np.clip(r_real, 3*a_radius, None) 
    th_p = np.arccos(np.clip(z_p / r_p, -1.0, 1.0))
    chi_p = np.arctan2(y_p, x_p)
    return r_p, th_p, chi_p

# Berechnung der lokalen Koordinaten für alle 4 Streuer
r_p1, th_p1, ch_p1 = get_loc_coords(x1, y1, z1, a_1(0))
r_p2, th_p2, ch_p2 = get_loc_coords(x2, y2, z2, a_2(0))
r_p3, th_p3, ch_p3 = get_loc_coords(x3, y3, z3, a_3(0))
r_p4, th_p4, ch_p4 = get_loc_coords(x4, y4, z4, a_4(0))

# 3. Berechnung des Skalarfeldes (Betrag) im 3D Raum
# Hinweis: Wir nutzen den bereits berechneten Lösungsvektor X_sol 
X_sol = X_val.flatten() # Einmalig berechnen für Effizienz

# Wir bauen die Summe manuell zusammen, um mehrfache Funktionsaufrufe zu sparen
# S_m1_1, S_0_1, S_1_1 müssen vektorsiert funktionieren (tun sie dank numpy)
def compute_part(idx_start, r_p, th_p, ch_p):
    return (X_sol[idx_start]   * S_m1_1(r_p, th_p, ch_p) + 
            X_sol[idx_start+1] * S_0_1(r_p, th_p, ch_p) + 
            X_sol[idx_start+2] * S_1_1(r_p, th_p, ch_p))

phi_3d_vol = (compute_part(0, r_p1, th_p1, ch_p1) +
              compute_part(6, r_p2, th_p2, ch_p2) +
              compute_part(12, r_p3, th_p3, ch_p3) +
              compute_part(18, r_p4, th_p4, ch_p4))

scalar_field = np.abs(phi_3d_vol)

# 4. Isosurface Generierung (Marching Cubes)
# Level bestimmt, bei welcher Feldstärke die Fläche gezogen wird
level_val = np.mean(scalar_field) + 1.0 * np.std(scalar_field) #Mittelwert + 1 StdAbw, Wolken

try:
    verts, faces, normals, values = measure.marching_cubes(scalar_field, level=level_val, spacing=(x_lin[1]-x_lin[0], y_lin[1]-y_lin[0], z_lin[1]-z_lin[0]))
    
    # Korrektur der Position (marching_cubes startet bei 0,0,0)
    verts[:, 0] += x_lin[0]
    verts[:, 1] += y_lin[0]
    verts[:, 2] += z_lin[0]

    # 5. Plotten
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Erstellen der Polygon-Collection
    mesh = Poly3DCollection(verts[faces])
    mesh.set_alpha(0.3) # Transparenz
    mesh.set_facecolor('dodgerblue')
    mesh.set_edgecolor('none')

    ax.add_collection3d(mesh)

    # Plot-Grenzen setzen
    ax.set_xlim(x_lin.min(), x_lin.max())
    ax.set_ylim(y_lin.min(), y_lin.max())
    ax.set_zlim(z_lin.min(), z_lin.max())
    
    ax.view_init(elev=45, azim=60)
    
    ax.set_xlabel('$y$ (m)')
    ax.set_ylabel('$x$ (m)')
    ax.set_zlabel('$z$ (m)')
    ax.set_title(f'{pol_mode}: Isosurface von |$\phi_{{scat}}$| = {level_val:.2e} V/m')

    plt.tight_layout()
    plt.show()

except RuntimeError:
    print("Konnte keine Isosurface finden.")

