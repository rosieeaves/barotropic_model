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

        self.d = d
        self.dx = d
        self.dy = d
        self.Nx = Nx
        self.Ny = Ny 
        self.Lx = int(d*Nx)
        self.Ly = int(d*Ny)
        self.bathy = bathy
        self.f0 = f0 
        self.beta = beta
        self.f = [(f0+(j*beta*self.dy))*np.ones(Nx+1) for j in range(self.Ny+1)]      
        self.XG = [i*self.dx for i in range(self.Nx+1)]
        self.YG = [j*self.dy for j in range(self.Ny+1)]
        self.XC = [(i+0.5)*self.dx for i in range(self.Nx)]
        self.YC = [(j+0.5)*self.dy for j in range(self.Ny)]

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
                    xibar_np1 = scheme(kw.get('xibar_n'),self.dt,F_n)

                    # SOLVE FOR PSIBAR
                    # 1. solve xibar = div A using equation ellip1 A = xibar
                    print('A')
                    A = np.linalg.solve(self.ellip1,np.array(xibar_np1).flatten())
                    # 2. B=A*H using H_ellip (H_ellip has repeated values for x and y components of A)
                    print('B')
                    B = A*self.H_ellip
                    # 3. solve B = grad psibar using equation ellip2 psibar = B
                    print('psibar_np1')
                    psibar_np1 = np.linalg.solve(self.ellip2,B)

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

            def forward_Euler(xibar_n,dt,F_n):
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
            for t in range(2,Nt):
                print('ts='+str(t))
                func_to_run = time_step(scheme=forward_Euler,\
                F_nm2=F_nm2, F_nm1=F_nm1, F_n=F_n, xibar_n=xibar_n, xibar_np1=xibar_np1, \
                    psibar_n=psibar_n, psibar_np1=psibar_np1)
            F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1 = func_to_run()

            return psibar_n



    def advection(self,zetabar,psibar):

        # psibar and zetabar have coordinates [y,x]

        print('J start')

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

        # create matrices to be used to solve elliptic function for psibar
        # 1. solve xibar = div A where A = (grad psibar)/H
        # 2. B = A*H
        # 3. solve grad psibar = B

        # 1. matrix to solve xibar = div A where A = (grad psibar)/H
        ellip1 = np.zeros((self.Nx*self.Ny,self.Nx*self.Ny*2),dtype=object)
        #ellip1 = np.zeros((self.Nx*self.Ny,self.Nx*self.Ny*2),dtype=object)

        for i in range(self.Nx):
            for j in range(self.Ny):
                A = np.zeros((self.Ny,self.Nx,2)) # A[j,i,0] = A[j,i]^x, A[j,i,1] = A[j,i]^y
                if i == 0 and j == 0:
                    A[0,1,0] = 1/self.dx
                    A[0,0,0] = -1/self.dx
                    A[1,0,1] = 1/self.dy
                    A[0,0,1] = -1/self.dy

                elif i ==0 and j == self.Ny-1:
                    A[self.Ny-1,1,0] = 1/self.dx
                    A[self.Ny-1,0,0] = -1/self.dx
                    A[self.Ny-1,0,1] = 1/self.dy
                    A[self.Ny-2,0,1] = -1/self.dy

                elif i == self.Nx-1 and j ==0:
                    A[0,self.Nx-1,0] = 1/self.dx
                    A[0,self.Nx-2,0] = -1/self.dx
                    A[1,self.Nx-1,1] = 1/self.dy
                    A[0,self.Nx-1,1] = -1/self.dy 

                elif i == self.Nx-1 and j == self.Ny-1:
                    A[self.Ny-1,self.Nx-1,0] = 1/self.dx
                    A[self.Ny-1,self.Nx-2,0] = -1/self.dx
                    A[self.Ny-1,self.Nx-1,1] = 1/self.dy 
                    A[self.Ny-2,self.Nx-1,1] = -1/self.dy 

                elif i == 0:
                    A[j,1,0] = 1/self.dx 
                    A[j,0,0] = -1/self.dx
                    A[j+1,0,1] = 1/(2*self.dy)
                    A[j-1,0,1] = -1/(2*self.dy) 

                elif i == self.Nx-1:
                    A[j,self.Nx-1,0] = 1/self.dx
                    A[j,self.Nx-2,0] = -1/self.dx 
                    A[j+1,self.Nx-1,1] = 1/(2*self.dy)
                    A[j-1,self.Nx-1,1] = -1/(2*self.dy)

                elif j ==0:
                    A[0,i+1,0] = 1/(2*self.dx) 
                    A[0,i-1,0] = -1/(2*self.dx)
                    A[1,i,1] = 1/self.dy 
                    A[0,i,1] = -1/self.dy 

                elif j == self.Ny-1:
                    A[self.Ny-1,i+1,0] = 1/(2*self.dx) 
                    A[self.Ny-1,i-1,0] = -1/(2*self.dx) 
                    A[self.Ny-1,i,1] = 1/self.dy 
                    A[self.Ny-2,i,1] = -1/self.dy

                else:
                    A[j,i+1,0] = 1/(2*self.dx) 
                    A[j,i-1,0] = -1/(2*self.dx) 
                    A[j+1,i,1] = 1/(2*self.dy) 
                    A[j-1,i,1] = -1/(2*self.dy) 

                ellip1[j*self.Nx+i] = A.flatten()
        
        self.ellip1 = ellip1

        # 2. B=A*H
        # need H flattened and with repeated values for x and y components of A
        self.H_ellip = np.repeat(self.bathy.flatten(),repeats=2,axis=0)

        # 3. matrix to solve grad psibar = B
        ellip2 = np.zeros((self.Ny*self.Nx*2,self.Ny*self.Nx))

        for j in range(self.Ny):
            for i in range(self.Nx):
                for X in range(2): # X = 0 => x, X = 1 => y
                    psi_ellip = np.zeros((self.Ny,self.Nx))

                    if i == 0 and X == 0:
                        psi_ellip[j,1] = 1/self.dx
                        psi_ellip[j,0] = -1/self.dx

                    elif i == self.Nx-1 and X == 0:
                        psi_ellip[j,self.Nx-1] = 1/self.dx
                        psi_ellip[j,self.Nx-2] = -1/self.dx

                    elif j == 0 and X == 1:
                        psi_ellip[1,i] = 1/self.dy 
                        psi_ellip[0,i] = -1/self.dy 

                    elif j == self.Ny-1 and X == 1:
                        psi_ellip[self.Ny-1,i] = 1/self.dy 
                        psi_ellip[self.Ny-2,i] = -1/self.dy 

                    elif X == 0:
                        psi_ellip[j,i+1] = 1/(2*self.dx) 
                        psi_ellip[j,i-1] = -1/(2*self.dx)

                    else: # should be when X == 1
                        psi_ellip[j+1,i] = 1/(2*self.dy) 
                        psi_ellip[j-1,i] = -1/(2*self.dy) 
                    

                    ellip2[j*self.Nx*2 + i*2 +X] = psi_ellip.flatten()

        self.ellip2 = ellip2



#%%
Nx = 100
Ny=100
test = Barotropic(d=5000,Nx=Nx,Ny=Ny,bathy=100*np.ones((Ny+1,Nx+1)),f0=0.7E-4,beta=2E-11)

#%%
test.init_psi(k_peak=2,const=1)

X,Y = np.meshgrid(test.XG,test.YG)
fig,axs = plt.subplots(1,1)
axs.contour(X,Y,test.psi_0)
axs.set_aspect(1)
plt.show()

# %%
test.model(dt=10,Nt=10,gamma_q=0.1,r=0.01)

# %%
X,Y = np.meshgrid(test.XG,test.YG)
fig,axs = plt.subplots(1,1)
axs.contour(X,Y,test.psibar_np1)
axs.set_aspect(1)
plt.show()

# %%
print(test.psibar_np1)
# %%
