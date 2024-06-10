import numpy as np
import matplotlib.pyplot as plt

# parameters
L = 3       # length of Laval nozzle
Npt = 61    # number of discritisation points
dx = L/(Npt-1)
x = np.linspace(0,3,Npt)  # dimensionless x/L
gamma = 1.4    
R = 287.1    # unit: J/(K*kg)
C = 0.5        # Courant number

# initial condition functions
def rho_init_fun(x):
    return 1 - 0.314*x
def T_init_fun(x):
    return 1 - 0.2314*x
def V_init_fun(x,T):
    return (0.1 + 1.09*x)*(np.sqrt(T))
def A_fun(x):
    return 1 + 2.2*(x-1.5)**2

# initial conditions
N = 1000                 # number of time steps
time = 0
rho = np.zeros((N,Npt))
T = np.zeros((N,Npt))
V = np.zeros((N,Npt))
Ma = np.zeros((N,Npt))
p = rho*T

rho[0,:] = rho_init_fun(x) # dimensionless rho/rho0
T[0,:] = T_init_fun(x)     # dimensionless T/T0
V[0,:] = V_init_fun(x,T[0,:])   # dimensionelss V/V0
a = np.sqrt(T)   # dimensionless a/a0
A = A_fun(x)          # dimensionless A/A*
drhodt = np.zeros(Npt)   
dVdt = np.zeros(Npt)
dTdt = np.zeros(Npt)
drhobdt = np.zeros(Npt)   
dVbdt = np.zeros(Npt)
dTbdt = np.zeros(Npt)
drhodtav = np.zeros(Npt)  
dVdtav = np.zeros(Npt)
dTdtav = np.zeros(Npt)

# time step
# initial conditions
N = 1400                 # number of time steps
time = 0
rho = np.zeros((N,Npt))
T = np.zeros((N,Npt))
V = np.zeros((N,Npt))
p = np.zeros((N,Npt))

rho[0,:] = rho_init_fun(x) # dimensionless rho/rho0
T[0,:] = T_init_fun(x)     # dimensionless T/T0
V[0,:] = V_init_fun(x,T[0,:])   # dimensionelss V/V0
a = np.sqrt(T)   # dimensionless a/a0
Ma = V/a
A = A_fun(x)          # dimensionless A/A*
drhodt = np.zeros(Npt)   
dVdt = np.zeros(Npt)
dTdt = np.zeros(Npt)
drhobdt = np.zeros(Npt)   
dVbdt = np.zeros(Npt)
dTbdt = np.zeros(Npt)
drhodtav = np.zeros(Npt)  
dVdtav = np.zeros(Npt)
dTdtav = np.zeros(Npt)

# time step
for n in range(N-1):
    for i in range(1,Npt-1):
        drhodt[i] = -V[n,i]*(rho[n,i+1]-rho[n,i])/dx - rho[n,i]*(V[n,i+1]-V[n,i])/dx - rho[n,i]*V[n,i]*(np.log(A[i+1])-np.log(A[i]))/dx
        dVdt[i] = -V[n,i]*(V[n,i+1]-V[n,i])/dx - 1/gamma*((T[n,i+1]-T[n,i])/dx+T[n,i]/rho[n,i]*(rho[n,i+1]-rho[n,i])/dx)
        dTdt[i] = -V[n,i]*(T[n,i+1]-T[n,i])/dx - (gamma-1)*T[n,i]*((V[n,i+1]-V[n,i])/dx+V[n,i]*(np.log(A[i+1])-np.log(A[i]))/dx)
    dtime = C*dx/(V[n,:]+a[n,:])
    dt = min(dtime)
    rhob = rho[n,:] + drhodt*dt
    Vb = V[n,:] + dVdt*dt
    Tb = T[n,:] + dTdt*dt
    for i in range(1,Npt-1):
        drhobdt[i] = -Vb[i]*(rhob[i]-rhob[i-1])/dx - rhob[i]*(Vb[i]-Vb[i-1])/dx - rhob[i]*Vb[i]*(np.log(A[i])-np.log(A[i-1]))/dx
        dVbdt[i] = -Vb[i]*(Vb[i]-Vb[i-1])/dx - 1/gamma*((Tb[i]-Tb[i-1])/dx + Tb[i]/rhob[i]*(rhob[i]-rhob[i-1])/dx)
        dTbdt[i] = -Vb[i]*(Tb[i]-Tb[i-1])/dx - (gamma-1)*Tb[i]*((Vb[i]-Vb[i-1])/dx + Vb[i]*(np.log(A[i])-np.log(A[i-1]))/dx)
    drhodtav = 1/2*(drhodt + drhobdt)
    dVdtav = 1/2*(dVdt + dVbdt)
    dTdtav = 1/2*(dTdt + dTbdt)
    rho[n+1,:] = rho[n,:] + drhodtav*dt
    V[n+1,:] = V[n,:] + dVdtav*dt
    T[n+1,:] = T[n,:] + dTdtav*dt
    p[n+1,:] = rho[n+1,:]*T[n+1,:]
    a[n+1,:] = np.sqrt(T[n+1,:])
    Ma[n+1,:] = V[n+1,:]/a[n+1,:]
    time += dt

    # update boundaries
    V[n+1,0] = 2*V[n+1,1] - V[n+1,2]
    V[n+1,Npt-1] = 2*V[n+1,Npt-2] - V[n+1,Npt-3]
    rho[n+1,Npt-1] = 2*rho[n+1,Npt-2] - rho[n+1,Npt-3]
    T[n+1,Npt-1] = 2*T[n+1,Npt-2] - T[n+1,Npt-3]

# Figures
X = np.linspace(0,N+1,N)
plt.rc('text', usetex=True)   # latex mode on
plt.figure(figsize=(10,6))
plt.plot(X, Ma[:,int(Npt/2)], label=r'$Ma$')
plt.plot(X, rho[:,int(Npt/2)], label=r'$\rho$')
plt.plot(X, T[:,int(Npt/2)], label=r'$T$')
#plt.plot(X, V[:,int(Npt/2)], label='V')
plt.plot(X, p[:,int(Npt/2)], label=r'$p$')
plt.xlabel('Time step', fontsize=14)
plt.title('Steady solution for 1D Laval nozzle in %.3f sec' %time,fontsize=20)
plt.legend()
plt.grid(True)
plt.show()