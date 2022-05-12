#%%

import numpy as np
import pyfftw.interfaces.numpy_fft as fftw
import matplotlib.pyplot as plt
import xarray as xr
import scipy as sp
from functools import wraps

#%%

class Barotropic:

    def __init__(self,d,Nx,Ny,bathy,f0,beta):

        try:
            np.shape(bathy) == (Ny+1,Nx+1)
        except ValueError:
            print('bathy ust have shape (Ny+1,Nx+1) to be defined on the grid corners.')
        else:

            self.d = d
            self.dx = d
            self.dy = d
            self.Nx = Nx
            self.Ny = Ny 
            self.Lx = int(d*Nx)
            self.Ly = int(d*Ny)
            self.f0 = f0 
            self.beta = beta
            self.f = [(f0+(j*beta*self.dy))*np.ones(Nx+1) for j in range(self.Ny+1)]      
            self.XG = [i*self.dx for i in range(self.Nx+1)]
            self.YG = [j*self.dy for j in range(self.Ny+1)]
            self.XC = [(i+0.5)*self.dx for i in range(self.Nx)]
            self.YC = [(j+0.5)*self.dy for j in range(self.Ny)]
            self.NxG = self.Nx + 1
            self.NyG = self.Ny + 1
            self.bathy = xr.DataArray(
                bathy,
                dims = ['YG', 'XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )
        

    def init_psi(self,k_peak,const):

        rng = np.random.default_rng(seed=1234)
         
        # for generating a function at 2N+1 points, first need to generate 
        # Fourier coefficients at K=-N,...,N
        # Note: psi values are calculated on grid corners so that velocities are calculated 
        # at correct points of Arakawa C-grid.

        kx = np.concatenate((np.arange(0,(self.Nx+1)/2), np.arange(-(self.Nx+1)/2,0)))
        ky = np.concatenate((np.arange(0,(self.Ny+1)/2), np.arange(-(self.Ny+1)/2,0)))

        def wavenumber_std(k):
            return (k ** (-1) * (1 + (k / k_peak) ** 4) ** (-1)) ** (0.5) # where does this formula come from? 

        # array to contain Fourier coefficients
        Ak = np.zeros((self.Ny+1, self.Nx+1), dtype='complex128')

        # run through x and y wavenumbers
        for i in range(self.Nx+1):
            for j in range(self.Ny+1):
                # calculate total wavenumber
                k = (kx[i] ** 2 + ky[j] ** 2) ** 0.5
                if i + j > 0:
                    # generate random complex number as Fourier coefficient
                    Ak[j,i] = ((rng.normal() + 1j * rng.normal()) * wavenumber_std(k)) # why multiply by wavenumber_std(K)?

        # inverse FFT on Fourier coefficients to get value of psi at (2N+1)x(2N+1) grid points
        psi = np.real(fftw.ifft2(Ak))
        stream = np.zeros_like(psi)

        # make zero at boundaries
        for i in range(self.Nx+1):
            for j in range(self.Ny+1):
                stream[j][i] = const*(1-np.cos(np.pi*self.XG[i]/self.Lx)**50)*(1-np.cos(np.pi*self.YG[j]/self.Ly)**50)*psi[j][i]
        
        self.psi_0 = xr.DataArray(
            stream,
            dims = ['YG','XG'],
            coords = {
                'YG': self.YG,
                'XG': self.XG
            }
        )

    def model(self,dt,Nt,gamma_q,r):

        try:
            print('try')
            self.psi_0

        except AttributeError:
            print('Run init_psi() before running model to generate initial conditions')

        else:
        
            # generate matrices to solve for psibar
            self.elliptic()

            # store model parameters in object
            self.dt = dt
            self.Nt = Nt
            self.gamma_q = gamma_q
            self.r = r

            #calculate initial xibar from initial psibar
            xibar_0 = self.psi_0.differentiate(coord='XG').differentiate(coord='XG') + self.psi_0.differentiate(coord='YG').differentiate(coord='YG')
            self.xibar_0 = xr.DataArray(
                xibar_0,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )
            xibar_n = xibar_0
            # create xibar_np1 array
            xibar_np1 = np.zeros_like(xibar_n)

            # initialise parameters to store F_n, F_nm1 and F_nm2 to calculate xibar_np1 with AB3
            F_n = np.zeros_like(xibar_n)
            F_nm1 = np.zeros_like(xibar_n)
            F_nm2 = np.zeros_like(xibar_n)

            # initialise parameters pisbar, zetabar, and grad xibar components
            psibar_n = self.psi_0
            psibar_np1 = np.zeros_like(psibar_n)
            zetabar_n = xibar_0+self.f
            xibar_dx_n = np.zeros_like(xibar_n)
            xibar_dy_n = np.zeros_like(xibar_n)

            # time stepping function

            #def time_step(scheme,F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1):
            def time_step(scheme,**kw):
                def wrapper(*args,**kwargs):
                    # calculate values needed to step forward xibar
                    # advection term
                    print('adv')
                    zetabar_n = kw.get('xibar_n') + self.f
                    adv_n = self.advection(zetabar_n,kw.get('psibar_n'))

                    # flux term

                    # dissipation 
                    print('D')
                    D_n = self.r*kw.get('xibar_n')

                    # F_n
                    print('F')
                    F_n = -adv_n - D_n # - div_flux_n

                    print('xibar')
                    xibar_np1 = scheme(kw.get('xibar_n'),self.dt,F_n,kw.get('F_nm1'),kw.get('F_nm2'))

                    # SOLVE FOR PSIBAR
                    psibar_np1 = np.linalg.solve(self.matrix_elliptic,np.array(xibar_np1).flatten())
                    psibar_np1 = psibar_np1.reshape((self.NyG,self.NxG))
                    print(np.shape(psibar_np1))

                    # DUMP XIBAR AND PSIBAR AT CERTAIN INTERVALS

                    # reset values
                    print('reset')
                    F_nm2 = kw.get('F_nm1').copy()
                    F_nm1 = F_n.copy()
                    F_n = np.zeros_like(F_n)
                    xibar_n = xibar_np1.copy()
                    xibar_np1 = np.zeros_like(xibar_np1)
                    psibar_n = psibar_np1.copy()
                    psibar_np1 = np.zeros_like(psibar_np1)

                    print('return')

                    return F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1
                return wrapper

            def forward_Euler(xibar_n,dt,F_n,F_nm1,F_nm2):
                xibar_np1 = xibar_n + dt*F_n
                return xibar_np1

            def AB3(xibar_n,dt,F_n,F_nm1,F_nm2):
                xibar_np1 = xibar_n + (dt/12)*(23*F_n - 16*F_nm1 + 5*F_nm2)
                return xibar_np1

            # first two time steps with forward Euler
            print('ts=1')
            func_to_run = time_step(scheme=forward_Euler,\
                F_nm2=F_nm2, F_nm1=F_nm1, F_n=F_n, xibar_n=xibar_n, xibar_np1=xibar_np1, \
                    psibar_n=psibar_n, psibar_np1=psibar_np1)
            F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1 = func_to_run()
            print('ts=2')
            func_to_run = time_step(scheme=forward_Euler,\
                F_nm2=F_nm2, F_nm1=F_nm1, F_n=F_n, xibar_n=xibar_n, xibar_np1=xibar_np1, \
                    psibar_n=psibar_n, psibar_np1=psibar_np1)
            F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1 = func_to_run()

            # all other time steps with AB3:
            for t in range(3,Nt):
                print('ts='+str(t))
                func_to_run = time_step(scheme=AB3,\
                F_nm2=F_nm2, F_nm1=F_nm1, F_n=F_n, xibar_n=xibar_n, xibar_np1=xibar_np1, \
                    psibar_n=psibar_n, psibar_np1=psibar_np1)
                F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1 = func_to_run()

            return psibar_n



    def advection(self,zetabar,psibar):

        # psibar and zetabar have coordinates [y,x]

        print('J start')
        print(np.shape(zetabar))
        print(np.shape(psibar))

        J = np.zeros((self.Ny+1,self.Nx+1))

        for i in range(1,self.Nx):
            for j in range(1,self.Ny):

                J[j,i] = (1/(12*self.d**2))*(psibar[j+1,i]*(3*zetabar[j,i+1] - 3*zetabar[j,i-1] + zetabar[j+1,i+1] - zetabar[j-1,i-1]) + \

                            psibar[j-1,i]*(-3*zetabar[j,i+1] + 3*zetabar[j,i-1]) + \

                            psibar[j,i+1]*(-3*zetabar[j+1,i] + 3*zetabar[j-1,i] - zetabar[j+1,i+1] + zetabar[j-1,i-1]) + \

                            psibar[j,i-1]*(3*zetabar[j+1,i] - 3*zetabar[j-1,i]) + \

                            psibar[j+1,i+1]*(zetabar[j,i+1] - zetabar[j+1,i]) + \

                            psibar[j-1,i-1]*(zetabar[j,i+1] - zetabar[j+1,i])
                
                )
        
        print('J done')
        return J

    def elliptic(self):

        # create coefficient matrices

        H_2 = np.array([np.concatenate(([(1/(self.bathy[j,0]*self.d**2) + (self.bathy[j,1] - self.bathy[j,0])/(self.d*self.bathy[j,0]**2))],\
                        [(1/(self.bathy[j,i]*self.d**2) + (self.bathy[j,i+1] - self.bathy[j,i-1])/(2*self.d*self.bathy[j,i]**2)) for i in range(1,self.NxG-1)],\
                            [(1/(self.bathy[j,self.NxG-1]*self.d**2) + (self.bathy[j,self.NxG-1] - self.bathy[j,self.NxG-2])/(self.d*self.bathy[j,self.NxG-1]**2))])) for j in range(self.NyG)])

        H_3 = np.array([np.concatenate(([(1/(self.bathy[j,0]*self.d**2) - (self.bathy[j,1] - self.bathy[j,0])/(self.d*self.bathy[j,0]**2))],\
                        [(1/(self.bathy[j,i]*self.d**2) - (self.bathy[j,i+1] - self.bathy[j,i-1])/(2*self.d*self.bathy[j,i]**2)) for i in range(1,self.NxG-1)],\
                            [(1/(self.bathy[j,self.NxG-1]*self.d**2) - (self.bathy[j,self.NxG-1] - self.bathy[j,self.NxG-2])/(self.d*self.bathy[j,self.NxG-1]**2))])) for j in range(self.NyG)])

        H_4 = np.array([np.concatenate(([(1/(self.bathy[0,i]*self.d**2) + (self.bathy[1,i] - self.bathy[0,i])/(self.d*self.bathy[0,i]**2))],\
                        [(1/(self.bathy[j,i]*self.d**2) + (self.bathy[j+1,i] - self.bathy[j-1,i])/(2*self.d*self.bathy[j,i]**2)) for j in range(1,self.NyG-1)],\
                            [(1/(self.bathy[self.NyG-1,i]*self.d**2) + (self.bathy[self.NyG-1,i] - self.bathy[self.NyG-2,i])/(self.d*self.bathy[self.NyG-1,i]**2))])) for i in range(self.NxG)])

        H_5 = np.array([np.concatenate(([(1/(self.bathy[0,i]*self.d**2) - (self.bathy[1,i] - self.bathy[0,i])/(self.d*self.bathy[0,i]**2))],\
                        [(1/(self.bathy[j,i]*self.d**2) - (self.bathy[j+1,i] - self.bathy[j-1,i])/(2*self.d*self.bathy[j,i]**2)) for j in range(1,self.NyG-1)],\
                            [(1/(self.bathy[self.NyG-1,i]*self.d**2) - (self.bathy[self.NyG-1,i] - self.bathy[self.NyG-2,i])/(self.d*self.bathy[self.NyG-1,i]**2))])) for i in range(self.NxG)])

        H_6 = np.array([[-4/(self.bathy[j,i]*self.d**2) for i in range(self.NxG)]for j in range(self.NyG)])

        # create matrix to solve for psibar

        centre_ones = np.concatenate(([1],[0 for i in range(1,self.NxG-1)],[1]))
        diags_ones = np.concatenate((np.ones(self.NxG),np.tile(centre_ones,self.NyG-2),np.ones(self.NxG)))
        diagonal_ones = np.diagflat(diags_ones)

        centre = np.concatenate(([0],[1 for i in range(1,self.NxG-1)],[0]))
        diags = np.concatenate((np.zeros(self.NxG),np.tile(centre,self.NyG-2),np.zeros(self.NxG)))
        diagonal = np.diagflat(diags)*H_6.flatten()

        centre_pi = np.concatenate(([0],[1 for i in range(1,self.NxG-1)],[0]))
        diags_pi = np.concatenate((np.zeros(self.NxG),np.tile(centre_pi,self.NyG-2),np.zeros(self.NxG)))
        diag_pi = np.diagflat(diags_pi[:len(diags_pi)-1],k=1)*H_2.flatten()

        centre_mi = np.concatenate(([0],[1 for i in range(1,self.NxG-1)],[0]))
        diags_mi = np.concatenate((np.zeros(self.NxG),np.tile(centre_mi,self.NyG-2),np.zeros(self.NxG)))
        diag_mi = np.diagflat(diags_mi[1:],k=-1)*H_3.flatten()

        centre_pj = np.concatenate(([0],[1 for i in range(1,self.NxG-1)],[0]))
        diags_pj = np.concatenate((np.zeros(self.NxG),np.tile(centre_pj,self.NyG-2),np.zeros(self.NxG)))
        diag_pj = np.diagflat(diags_pj[:len(diags_pi)-self.NxG],k=self.NxG)*H_4.flatten()

        centre_mj = np.concatenate(([0],[1 for i in range(1,self.NxG-1)],[0]))
        diags_mj = np.concatenate((np.zeros(self.NxG),np.tile(centre_mj,self.NyG-2),np.zeros(self.NxG)))
        diag_mj = np.diagflat(diags_mj[self.NxG:],k=-self.NxG)*H_5.flatten()

        matrix_elliptic = diagonal + diag_pi + diag_mi + diag_pj + diag_mj + diagonal_ones

        self.matrix_elliptic = matrix_elliptic

    



#%%
Nx = 3
Ny= 3
test = Barotropic(d=5000,Nx=Nx,Ny=Ny,bathy=100*np.ones((Ny+1,Nx+1)),f0=0.7E-4,beta=2E-11)

#%%
test.init_psi(k_peak=2,const=1)

X,Y = np.meshgrid(test.XG,test.YG)
fig,axs = plt.subplots(1,1)
axs.contour(X,Y,test.psi_0)
axs.set_aspect(1)
plt.show()

# %%
out = test.model(dt=10,Nt=10,gamma_q=0.1,r=0.01)

# %%
X,Y = np.meshgrid(test.XG,test.YG)
fig,axs = plt.subplots(1,1)
axs.contour(X,Y,out)
axs.set_aspect(1)
plt.show()

# %%
