import numpy as np
from barotropic_model import Barotropic
import time
import xarray as xr
import matplotlib.pyplot as plt

H = 5000
h = 500
dx = 2000
dy = 2000
Nx = 500
Ny = 500
Lx = dx*Nx 
Ly = dy*Ny

# bathy_mount = [[(H-h*np.sin((np.pi*(i*dx))/Lx)*np.sin((np.pi*(j*dy))/Ly)) for i in range(Nx+1)] for j in range(Ny+1)]
bathy_flat = np.ones((Ny+1,Nx+1))*H

test_diags = Barotropic(d=2000,Nx=Nx,Ny=Ny,bathy=bathy_flat,f0=0.7E-4,beta=2.E-11)

test_diags.gen_init_psi(k_peak=2,const=1E12)
test_diags.xi_from_psi()

diagnostics = ['xi_u','xi_v','u_u','v_v','xi_xi']
start_time = time.time_ns()
data = test_diags.model(dt=900,Nt=96000,gamma_q=0,r_BD=0,r_diff=2,tau_0=0,rho_0=1000,dumpFreq=86400,meanDumpFreq=864000,diags=diagnostics)
data.to_netcdf('./model_data/FDT_MOUNT_2km_1000days')
end_time = time.time_ns()
print((end_time - start_time)/1E9)

