#%%

import numpy as np
import pyfftw.interfaces.numpy_fft as fftw
import matplotlib.pyplot as plt
import xarray as xr
import scipy as sp
import scipy.sparse.linalg as linalg
import matplotlib.colors as colors
import scipy.interpolate as interp

#%%

class Barotropic:

    def __init__(self,d,Nx,Ny,bathy,f0,beta):

        try:
            np.shape(bathy) == (Ny+1,Nx+1)
        except ValueError:
            print('bathy must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
        else:

            self.d = d
            self.dx = d
            self.dy = d
            self.Nx = Nx
            self.Ny = Ny 
            self.Lx = int(d*Nx)
            self.Ly = int(d*Ny)
            self.XG = np.array([i*self.dx for i in range(self.Nx+1)])
            self.YG = np.array([j*self.dy for j in range(self.Ny+1)])
            self.XC = np.array([(i+0.5)*self.dx for i in range(self.Nx)])
            self.YC = np.array([(j+0.5)*self.dy for j in range(self.Ny)])
            self.NxG = self.Nx + 1
            self.NyG = self.Ny + 1

            self.f0 = f0 
            self.beta = beta
            f = np.array([(f0+(j*beta*self.dy))*np.ones(self.NxG) for j in range(self.NyG)])

            self.f = xr.DataArray(
                f,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }      
            )

            self.bathy = xr.DataArray(
                bathy,
                dims = ['YG', 'XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )
        

    def gen_init_psi(self,k_peak,const):

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
        
        self.psibar_0 = xr.DataArray(
            stream,
            dims = ['YG','XG'],
            coords = {
                'YG': self.YG,
                'XG': self.XG
            }
        )

    def init_psi(self,psi):
        try:
            np.shape(psi) == ((self.NyG,self.NxG))
        except ValueError:
            print('Initial psi must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
        else:
            # set zero on boundaries
            psi_remove = psi[1:-1,1:-1]
            psi_pad = np.pad(psi_remove,((1,1)),constant_values=0)
            self.psibar_0 = xr.DataArray(
                psi_pad,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

    def init_xi(self,xi):
        try:
            np.shape(xi) == ((self.NyG,self.NxG))
        except ValueError:
            print('Initial xi must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
        else:
            # set zero on boundaries
            xi_remove = xi[1:-1,1:-1]
            xi_pad = np.pad(xi_remove,((1,1)),constant_values=0)
            self.xibar_0 = xr.DataArray(
                xi_pad,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

    def psi_from_xi(self):
        try:
            self.xibar_0
        except NameError:
            print('Specifiy initial xi to calculate xi from psi.')
        else:
            # get elliptic solver
            self.elliptic()

            psibar = self.LU.solve((-np.array(self.xibar_0)).flatten())
            psibar = psibar.reshape((self.NyG,self.NxG))

            self.psibar_0 = xr.DataArray(
                psibar,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            ubar_0 = -self.psibar_0.differentiate(coord='YG')

            self.ubar_0 = xr.DataArray(
                ubar_0,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            vbar_0 = self.psibar_0.differentiate(coord='XG')

            self.vbar_0 = xr.DataArray(
                ubar_0,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )


    def xi_from_psi(self):
        try:
            self.psibar_0
        except NameError:
            print('Specify initial psi or run init_psi() to calculate inttial xi from initial psi.')
        else:
            #calculate initial xibar from initial psibar
            xibar_0 = np.array((self.psibar_0.differentiate(coord='XG').differentiate(coord='XG') + \
                self.psibar_0.differentiate(coord='YG').differentiate(coord='YG'))/self.bathy - \
                    (self.bathy.differentiate(coord='XG')*self.psibar_0.differentiate(coord='XG') + 
                    self.bathy.differentiate(coord='YG')*self.psibar_0.differentiate(coord='YG'))/(self.bathy**2))

            xibar_0 = np.pad(xibar_0[1:-1,1:-1],((1,1)),constant_values=0)

            self.xibar_0 = xr.DataArray(
                xibar_0,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            ubar_0 = -self.psibar_0.differentiate(coord='YG')

            self.ubar_0 = xr.DataArray(
                ubar_0,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            vbar_0 = self.psibar_0.differentiate(coord='XG')

            self.vbar_0 = xr.DataArray(
                vbar_0,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )


    def model(self,dt,Nt,gamma_q,r_BD,r_diff,tau_0,rho_0):

        try:
            self.psibar_0

        except AttributeError:
            print('Run init_psi() before running model to generate initial conditions')

        else:
        
            # generate matrices to solve for psibar
            try:
                self.LU
            except:
                self.elliptic()

            # store model parameters in object
            self.dt = dt
            self.Nt = Nt
            self.gamma_q = gamma_q
            self.r_BD = r_BD
            self.r_diff = r_diff
            self.tau_0 = tau_0 
            self.rho_0 = rho_0

            # initialise parameters pisbar, zetabar, and grad xibar components
            psibar_n = self.psibar_0
            psibar_np1 = np.zeros_like(psibar_n)
            xibar_n = self.xibar_0
            xibar_np1 = np.zeros_like(xibar_n)
            # comment out ubar and vbar for solid body rotation
            ubar_n = self.ubar_0
            ubar_np1 = np.zeros_like(ubar_n)
            vbar_n = self.vbar_0
            vbar_np1 = np.zeros_like(vbar_n)

            self.xibar = [self.xibar_0]
            self.psibar = [self.psibar_0]
            self.adv = [np.zeros((self.NyG,self.NxG))]
            # comment out ubar and vbar for solid body rotation
            self.ubar = [self.ubar_0]
            self.vbar = [self.vbar_0]

            # initialise parameters to store F_n, F_nm1 and F_nm2 to calculate xibar_np1 with AB3
            F_n = np.zeros_like(xibar_n)
            F_nm1 = np.zeros_like(xibar_n)
            F_nm2 = np.zeros_like(xibar_n)

            # calculate wind stress

            tau = [(-tau_0*np.cos((2*np.pi*y)/(self.Ly)))*np.ones(self.NxG) for y in self.YG]
            self.tau = xr.DataArray(
                np.array(tau),
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            self.calc_wind_stress()


            # time stepping function

            #def time_step(scheme,F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1):
            def time_step(scheme,**kw):
                def wrapper(*args,**kwargs):

                    try:
                        kw.get('ubar_n')
                        kw.get('vbar_n')
                        kw.get('xibar_n')
                        kw.get('F_nm1')
                        kw.get('F_nm2')
                    except TypeError:
                        print('None returned.')

                    else:
                        # calculate advection term
                        adv_n = self.advection_new(self.xibar[-1],self.psibar[-1])

                        # flux term

                        # dissiaption from bottom drag
                        BD_n = self.r_BD*self.xibar[-1]

                        # diffusion of vorticity
                        xibar_n = xr.DataArray(
                            self.xibar[-1],
                            dims = ['YG','XG'],
                            coords = {
                                'YG': self.YG,
                                'XG': self.XG
                            }
                        )

                        diff_n = self.r_diff*(xibar_n.differentiate(coord='XG').differentiate(coord='XG') + \
                            xibar_n.differentiate(coord='YG').differentiate(coord='YG'))

                        # F_n
                        F_n = -adv_n - BD_n - diff_n + self.wind_stress

                        # calculate new value of xibar
                        xibar_np1 = scheme(self.xibar[-1],self.dt,F_n,kw.get('F_nm1'),kw.get('F_nm2'))

                        # uncomment for solid body rotation
                        # psibar_np1 = self.psibar[-1]

                        # SOLVE FOR PSIBAR
                        # comment out psibar_np1, ubar_np1 and vbar_np1 for solid body rotation
                        psibar_np1 = self.LU.solve((-np.array(xibar_np1)).flatten())
                        psibar_np1 = xr.DataArray(
                            psibar_np1.reshape((self.NyG,self.NxG)),
                            dims = ['YG','XG'],
                            coords = {
                                'YG': self.YG,
                                'XG': self.XG
                            }
                        )

                        ubar_np1 = xr.DataArray(
                            -psibar_np1.differentiate(coord='YG'),
                            dims = ['YG','XG'],
                            coords = {
                                'YG': self.YG,
                                'XG': self.XG
                            }
                        )

                        vbar_np1 = xr.DataArray(
                            psibar_np1.differentiate(coord='XG'),
                            dims = ['YG','XG'],
                            coords = {
                                'YG': self.YG,
                                'XG': self.XG
                            }
                        )

                        # DUMP XIBAR AND PSIBAR AT CERTAIN INTERVALS

                        self.xibar = np.append(self.xibar,[xibar_np1],axis=0)
                        self.psibar = np.append(self.psibar,[psibar_np1],axis=0)
                        # comment out ubar and vbar for solid body rotation
                        self.ubar = np.append(self.ubar,[ubar_np1],axis=0)
                        self.vbar = np.append(self.vbar,[vbar_np1],axis=0)

                        # reset values
                        F_nm2 = kw.get('F_nm1').copy()
                        F_nm1 = F_n.copy()
                        F_n = np.zeros_like(F_n)
                        xibar_n = xibar_np1.copy()
                        xibar_np1 = np.zeros_like(xibar_np1)
                        psibar_n = psibar_np1.copy()
                        psibar_np1 = np.zeros_like(psibar_np1)
                        # comment out ubar and vbar for solid body rotation
                        ubar_n = ubar_np1.copy()
                        ubar_np1 = np.zeros_like(ubar_n)
                        vbar_n = vbar_np1.copy()
                        vbar_np1 = np.zeros_like(vbar_n)

                        # comment out ubar and vbar for solid body rotation
                        return F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1, ubar_n, ubar_np1, vbar_n, vbar_np1
                return wrapper

            def forward_Euler(xibar_n,dt,F_n,F_nm1,F_nm2):
                xibar_np1 = xibar_n + dt*F_n
                return xibar_np1

            def AB3(xibar_n,dt,F_n,F_nm1,F_nm2):
                xibar_np1 = xibar_n + (dt/12)*(23*F_n - 16*F_nm1 + 5*F_nm2)
                return xibar_np1

            # first two time steps with forward Euler
            print('ts=1')
            # comment out ubar and vbar for solid body rotation
            func_to_run = time_step(scheme=forward_Euler,\
                F_nm2=F_nm2, F_nm1=F_nm1, F_n=F_n, xibar_n=xibar_n, xibar_np1=xibar_np1, \
                    psibar_n=psibar_n, psibar_np1=psibar_np1,ubar_n=ubar_n,ubar_np1=ubar_np1,\
                        vbar_n=vbar_n,vbar_np1=vbar_np1)
            # comment out for solid body rotation
            F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1, ubar_n, ubar_np1, vbar_n, vbar_np1 = func_to_run()
            # un comment for solid body rotation
            #F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1 = func_to_run()

            print('ts=2')
            # comment out ubar and vbar for solid body rotation
            func_to_run = time_step(scheme=forward_Euler,\
                F_nm2=F_nm2, F_nm1=F_nm1, F_n=F_n, xibar_n=xibar_n, xibar_np1=xibar_np1, \
                    psibar_n=psibar_n, psibar_np1=psibar_np1,ubar_n=ubar_n,ubar_np1=ubar_np1,\
                        vbar_n=vbar_n,vbar_np1=vbar_np1)
            # comment out for solid body rotation
            F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1, ubar_n, ubar_np1, vbar_n, vbar_np1 = func_to_run()
            # un comment for solid body rotation
            #F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1 = func_to_run()

            # all other time steps with AB3:
            for t in range(3,Nt):
                print('ts='+str(t))
                # comment out ubar and vbar for solid body rotation
                func_to_run = time_step(scheme=AB3,\
                F_nm2=F_nm2, F_nm1=F_nm1, F_n=F_n, xibar_n=xibar_n, xibar_np1=xibar_np1, \
                    psibar_n=psibar_n, psibar_np1=psibar_np1,ubar_n=ubar_n,ubar_np1=ubar_np1,\
                        vbar_n=vbar_n,vbar_np1=vbar_np1)
                # comment out for solid body rotation
                F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1, ubar_n, ubar_np1, vbar_n, vbar_np1 = func_to_run()
                # un comment for solid body rotation
                #F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1 = func_to_run()

            return psibar_n, xibar_n

    def advection_new(self,xi,psi):

        # calculate absolute velocity
        zeta = xi + self.f
        zeta = np.array(zeta)

        # interpolate psi onto cell centres
        f_psi = interp.interp2d(self.XG,self.YG,psi)
        psi_YCXC = np.array(f_psi(self.XC,self.YC))

        # get bathy as np array
        h = np.array(self.bathy)

        '''v_YGXC = np.array((psi_YCXC[:,1:] - psi_YCXC[:,:-1])/self.d) 
        u_YCXG = np.array((psi_YCXC[:-1,:] - psi_YCXC[1:,:])/self.d) 

        adv = (1/(2*self.d*self.bathy[1:-1,1:-1]))*((zeta[1:-1,1:-1] + zeta[2:,1:-1])*v_YGXC[1:,:] + \
            (zeta[1:-1,1:-1] + zeta[1:-1,2:])*u_YCXG[:,1:] - \
                (zeta[1:-1,1:-1] + zeta[:-2,1:-1])*v_YGXC[:-1,:] - \
                    (zeta[1:-1,1:-1] + zeta[1:-1,:-2])*u_YCXG[:,:-1]) 

        adv = np.pad(adv,((1,1)),constant_values=0)'''

        # calculate area average of advection term 
        # area average is take over the grid cell centred at the vorticity point, away from the boundaries
        adv = (1/(self.d**2))*(((psi_YCXC[1:,1:] - psi_YCXC[1:,:-1])*(zeta[1:-1,1:-1] + zeta[2:,1:-1]))/(h[1:-1,1:-1] + h[2:,1:-1]) - \
            ((psi_YCXC[1:,1:] - psi_YCXC[:-1,1:])*(zeta[1:-1,1:-1] + zeta[1:-1,2:]))/(h[1:-1,1:-1] + h[1:-1,2:]) - \
                ((psi_YCXC[:-1,1:] - psi_YCXC[:-1,:-1])*(zeta[1:-1,1:-1] + zeta[:-2,1:-1]))/(h[1:-1,1:-1] + h[:-2,1:-1]) + \
                    ((psi_YCXC[1:,:-1] - psi_YCXC[:-1,:-1])*(zeta[1:-1,1:-1] + zeta[1:-1,:-2]))/(h[1:-1,1:-1] + h[1:-1,:-2]))

        # pad with zero values on boundaries
        adv = np.pad(adv,((1,1)),constant_values=0)
        
        self.adv = np.append(self.adv,[adv],axis=0)

        return adv

    def elliptic(self):

        # create coefficient matrices

        C2 = np.array([np.concatenate(([1],\
            [(-1/((self.d**2)*self.bathy[j,i]) + (self.bathy[j,i+1] - self.bathy[j,i-1])/(1*(self.d**2)*(self.bathy[j,i]**2))) \
                for i in range(1,self.NxG-1)],\
                [1])) for j in range(self.NyG)])

        C3 = np.array([np.concatenate(([1],\
            [(-1/((self.d**2)*self.bathy[j,i]) - (self.bathy[j,i+1] - self.bathy[j,i-1])/(1*(self.d**2)*(self.bathy[j,i]**2))) \
                for i in range(1,self.NxG-1)],\
                [1])) for j in range(self.NyG)])
        
        C4 = np.concatenate(([np.ones(self.NxG)],\
            [[(-1/((self.d**2)*self.bathy[j,i]) + (self.bathy[j+1,i] - self.bathy[j-1,i])/(1*(self.d**2)*(self.bathy[j,i]**2))) \
                for i in range(self.NxG)] for j in range(1,self.NyG-1)],\
                [np.ones(self.NxG)]))

        C5 = np.concatenate(([np.ones(self.NxG)],\
            [[(-1/((self.d**2)*self.bathy[j,i]) - (self.bathy[j+1,i] - self.bathy[j-1,i])/(1*(self.d**2)*(self.bathy[j,i]**2))) \
                for i in range(self.NxG)] for j in range(1,self.NyG-1)],\
                [np.ones(self.NxG)]))

        C6 = np.array([np.concatenate(([1],\
            [4/(self.bathy[j,i]*(self.d**2)) for i in range(1,self.NxG-1)],\
                [1])) for j in range(self.NyG)])
    
        # create matrix to solve for psibar

        centre_ones = np.concatenate(([-1],[0 for i in range(1,self.NxG-1)],[-1]))
        diags_ones = np.concatenate((-1*np.ones(self.NxG),np.tile(centre_ones,self.NyG-2),-1*np.ones(self.NxG)))
        diagonal_ones = np.diagflat(diags_ones)

        centre = np.concatenate(([0],[1 for i in range(1,self.NxG-1)],[0]))
        diags = np.concatenate((np.zeros(self.NxG),np.tile(centre,self.NyG-2),np.zeros(self.NxG)))*C6.flatten()
        diagonal = np.diagflat(diags)

        centre_pi = np.concatenate(([0],[1 for i in range(1,self.NxG-1)],[0]))
        diags_pi = np.concatenate((np.zeros(self.NxG),np.tile(centre_pi,self.NyG-2),np.zeros(self.NxG)))*C2.flatten()
        diag_pi = np.diagflat(diags_pi[:len(diags_pi)-1],k=1)

        centre_mi = np.concatenate(([0],[1 for i in range(1,self.NxG-1)],[0]))
        diags_mi = np.concatenate((np.zeros(self.NxG),np.tile(centre_mi,self.NyG-2),np.zeros(self.NxG)))*C3.flatten()
        diag_mi = np.diagflat(diags_mi[1:],k=-1)

        centre_pj = np.concatenate(([0],[1 for i in range(1,self.NxG-1)],[0]))
        diags_pj = np.concatenate((np.zeros(self.NxG),np.tile(centre_pj,self.NyG-2),np.zeros(self.NxG)))*C4.flatten()
        diag_pj = np.diagflat(diags_pj[:len(diags_pi)-self.NxG],k=self.NxG)

        centre_mj = np.concatenate(([0],[1 for i in range(1,self.NxG-1)],[0]))
        diags_mj = np.concatenate((np.zeros(self.NxG),np.tile(centre_mj,self.NyG-2),np.zeros(self.NxG)))*C5.flatten()
        diag_mj = np.diagflat(diags_mj[self.NxG:],k=-self.NxG)

        diagonal_total = diagonal + diag_pi + diag_mi + diag_pj + diag_mj + diagonal_ones
        self.diagonal = diagonal_total
        LU = linalg.splu(sp.sparse.csc_matrix(diagonal_total))

        self.LU = LU

    def calc_wind_stress(self):
        # calculate the wind stress to be added to F_n

        tau = np.array(self.tau)
        bathy = self.bathy

        tau_xi = (-tau[2:,1:-1] + tau[:-2,1:-1])/(2*self.d*self.rho_0*bathy[1:-1,1:-1])
        tau_xi = np.pad(tau_xi,((1,1)),constant_values=0)

        self.wind_stress = xr.DataArray(
            tau_xi,
            dims = ['YG','XG'],
            coords = {
                'YG': self.YG,
                'XG': self.XG
            }
        )
    
