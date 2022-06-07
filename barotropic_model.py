#%%

import numpy as np
import pyfftw.interfaces.numpy_fft as fftw
import matplotlib.pyplot as plt
import xarray as xr
import scipy as sp
import scipy.sparse.linalg as linalg
import matplotlib.colors as colors
import scipy.interpolate as interp
import time

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

            ubar_0,vbar_0 = self.calc_UV(self.psibar_0)
            self.ubar_0 = ubar_0
            self.vbar_0 = vbar_0

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

            ubar_0,vbar_0 = self.calc_UV(self.psibar_0)
            self.ubar_0 = ubar_0
            self.vbar_0 = vbar_0


    def xi_from_psi(self):
        try:
            self.psibar_0
        except NameError:
            print('Specify initial psi or run init_psi() to calculate inttial xi from initial psi.')
        else:
            #calculate initial xibar from initial psibar
            '''xibar_0 = np.array((self.psibar_0.differentiate(coord='XG').differentiate(coord='XG') + \
                self.psibar_0.differentiate(coord='YG').differentiate(coord='YG'))/self.bathy - \
                    (self.bathy.differentiate(coord='XG')*self.psibar_0.differentiate(coord='XG') + 
                    self.bathy.differentiate(coord='YG')*self.psibar_0.differentiate(coord='YG'))/(self.bathy**2))'''

            h = np.array(self.bathy)
            psi = np.array(self.psibar_0)

            xibar_0 = (psi[1:-1,2:] + psi[1:-1,:-2] + psi[2:,1:-1] + psi[:-2,1:-1] - 4*psi[1:-1,1:-1])/(h[1:-1,1:-1]*(self.d**2)) - \
                ((psi[1:-1,2:] - psi[1:-1,:-2])*(h[1:-1,2:] - h[1:-1,:-2]) + \
                    (psi[2:,1:-1] - psi[:-2,1:-1])*(h[2:,1:-1] - h[:-2,1:-1]))/((h[1:-1,1:-1]**2)*(self.d**2))
 
            xibar_0 = np.pad(xibar_0,((1,1)),constant_values=0)

            self.xibar_0 = xr.DataArray(
                xibar_0,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            ubar_0,vbar_0 = self.calc_UV(self.psibar_0)
            self.ubar_0 = ubar_0
            self.vbar_0 = vbar_0


    def model(self,dt,Nt,gamma_q,r_BD,r_diff,tau_0,rho_0,dumpFreq):

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
            self.dumpFreq = dumpFreq

            # initialise parameters pisbar, zetabar, and grad xibar components
            self.psibar_n = self.psibar_0
            self.psibar_np1 = np.zeros_like(self.psibar_n)
            self.xibar_n = self.xibar_0
            self.xibar_np1 = np.zeros_like(self.xibar_n)

            # create arrays to save the data
            self.xibar = [self.xibar_0]
            self.psibar = [self.psibar_0]
            self.ubar = [self.ubar_0]
            self.vbar = [self.vbar_0]

            # initialise parameters to store F_n, F_nm1 and F_nm2 to calculate xibar_np1 with AB3
            self.F_n = np.zeros_like(self.xibar_n)
            self.F_nm1 = np.zeros_like(self.xibar_n)
            self.F_nm2 = np.zeros_like(self.xibar_n)

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

            # calculate dump frequency in number of timesteps
            dumpFreqTS = int(self.dumpFreq/self.dt)

            # time stepping function

            #def time_step(scheme,F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1):
            def time_step(scheme,**kw):
                def wrapper(*args,**kwargs):

                    # calculate advection term
                    adv_n = self.advection(self.xibar_n,self.psibar_n)

                    # flux term

                    # dissiaption from bottom drag
                    BD_n = self.r_BD*self.xibar_n

                    # diffusion of vorticity
                    diff_xi = self.diffusion_xi(self.xibar_n)
                    diff_n = self.r_diff*diff_xi

                    # F_n
                    F_n = -adv_n - BD_n + diff_n + self.wind_stress

                    # calculate new value of xibar
                    xibar_np1 = scheme(xibar_n=self.xibar_n,dt=self.dt,F_n=F_n,F_nm1=self.F_nm1,F_nm2=self.F_nm2)

                    # uncomment for solid body rotation
                    # psibar_np1 = self.psibar_n

                    # SOLVE FOR PSIBAR
                    # comment out for solid body rotation
                    psibar_np1 = self.LU.solve((-np.array(xibar_np1)).flatten())
                    psibar_np1 = psibar_np1.reshape((self.NyG,self.NxG))

                    # DUMP XIBAR AND PSIBAR AT CERTAIN INTERVALS

                    if kw.get('t')%kw.get('dumpFreq') == 0:
                        # calculate u and v from psi
                        u,v = self.calc_UV(psibar_np1)

                        # save xi, psi, u and v
                        self.xibar = np.append(self.xibar,[xibar_np1],axis=0)
                        self.psibar = np.append(self.psibar,[psibar_np1],axis=0)
                        self.ubar = np.append(self.ubar,[u],axis=0)
                        self.vbar = np.append(self.vbar,[v],axis=0)
       
                    # reset values
                    self.F_nm2 = self.F_nm1.copy()
                    self.F_nm1 = self.F_n.copy()
                    self.F_n = np.zeros_like(self.F_n)
                    self.xibar_n = xibar_np1.copy()
                    self.psibar_n = psibar_np1.copy()

                return wrapper

            def forward_Euler(xibar_n,dt,F_n,F_nm1,F_nm2):
                xibar_np1 = xibar_n + dt*F_n
                return xibar_np1

            def AB3(xibar_n,dt,F_n,F_nm1,F_nm2):
                xibar_np1 = xibar_n + (dt/12)*(23*F_n - 16*F_nm1 + 5*F_nm2)
                return xibar_np1

            # first two time steps with forward Euler
            func_to_run = time_step(scheme=forward_Euler,t=1,dumpFreq=dumpFreqTS)#,\
            func_to_run()

            func_to_run = time_step(scheme=forward_Euler,t=2,dumpFreq=dumpFreqTS)#,\
            func_to_run()

            # all other time steps with AB3:
            for t in range(3,Nt+1):
                func_to_run = time_step(scheme=AB3,t=t,dumpFreq=dumpFreqTS)#,\
                func_to_run()
            
            dataset_return = xr.Dataset(
                data_vars = dict(
                    xi = xr.DataArray(
                        self.xibar,
                        dims = ['T','YG','XG'],
                        coords = {
                            'T': np.arange(0,int(self.Nt*self.dt)+1,self.dumpFreq),
                            'YG': self.YG,
                            'XG': self.XG
                        }
                    ),
                    psi = xr.DataArray(
                        self.psibar,
                        dims = ['T','YG','XG'],
                        coords = {
                            'T': np.arange(0,int(self.Nt*self.dt)+1,self.dumpFreq),
                            'YG': self.YG,
                            'XG': self.XG
                        }
                    ),
                    u = xr.DataArray(
                        self.ubar,
                        dims = ['T','YC','XG'],
                        coords = {
                            'T': np.arange(0,int(self.Nt*self.dt)+1,self.dumpFreq),
                            'YC': self.YC,
                            'XG': self.XG
                        }
                    ),
                    v = xr.DataArray(
                        self.vbar,
                        dims = ['T','YG','XC'],
                        coords = {
                            'T': np.arange(0,int(self.Nt*self.dt)+1,self.dumpFreq),
                            'YG': self.YG,
                            'XC': self.XC
                        }
                    )
                ),
                coords = dict(
                    T = np.arange(0,int(self.Nt*self.dt)+1,self.dumpFreq),
                    YG = self.YG,
                    XG = self.XG
                )

            )

            return dataset_return

    def calc_UV(self,psi):

        h = np.array(self.bathy)
        psi = np.array(psi)

        v = (2/self.d)*((psi[:,1:] - psi[:,:-1])/(h[:,:-1] + h[:,1:]))
        u = -(2/self.d)*((psi[1:,:] - psi[:-1,:])/(h[:-1,:] + h[1:,:]))

        return u,v


    def advection(self,xi,psi):

        # calculate absolute velocity
        zeta = xi + self.f
        zeta = np.array(zeta)

        # interpolate psi onto cell centres
        psi = np.array(psi)
        psi_YCXC = (psi[:-1,:-1] + psi[:-1,1:] + psi[1:,:-1] + psi[1:,1:])/4

        # get bathy as np array
        h = np.array(self.bathy)

        # calculate area average of advection term 
        # area average is take over the grid cell centred at the vorticity point, away from the boundaries
        adv = (1/(self.d**2))*(((psi_YCXC[1:,1:] - psi_YCXC[1:,:-1])*(zeta[1:-1,1:-1] + zeta[2:,1:-1]))/(h[1:-1,1:-1] + h[2:,1:-1]) - \
            ((psi_YCXC[1:,1:] - psi_YCXC[:-1,1:])*(zeta[1:-1,1:-1] + zeta[1:-1,2:]))/(h[1:-1,1:-1] + h[1:-1,2:]) - \
                ((psi_YCXC[:-1,1:] - psi_YCXC[:-1,:-1])*(zeta[1:-1,1:-1] + zeta[:-2,1:-1]))/(h[1:-1,1:-1] + h[:-2,1:-1]) + \
                    ((psi_YCXC[1:,:-1] - psi_YCXC[:-1,:-1])*(zeta[1:-1,1:-1] + zeta[1:-1,:-2]))/(h[1:-1,1:-1] + h[1:-1,:-2]))

        # pad with zero values on boundaries
        adv = np.pad(adv,((1,1)),constant_values=0)

        return adv

    def diffusion_xi(self,xi):

        xi = np.array(xi)

        diffusion = (1/self.d**2)*(xi[1:-1,2:] + xi[1:-1,:-2] + xi[2:,1:-1] + xi[:-2,1:-1] - 4*xi[1:-1,1:-1])

        diffusion = np.pad(diffusion,((1,1)),constant_values=0)

        return diffusion



    def elliptic(self):

        # create coefficient matrices
        h = np.array(self.bathy)

        C2 = (-1/(h[1:-1,1:-1]*self.d**2)) + (1/(4*(h[1:-1,1:-1]**2)*(self.d**2)))*(h[1:-1,2:] - h[1:-1,:-2])
        C2 = np.pad(C2,((1,1)),constant_values=1)

        C3 = (-1/(h[1:-1,1:-1]*self.d**2)) - (1/(4*(h[1:-1,1:-1]**2)*(self.d**2)))*(h[1:-1,2:] - h[1:-1,:-2])
        C3 = np.pad(C3,((1,1)),constant_values=1)

        C4 = (-1/(h[1:-1,1:-1]*self.d**2)) + (1/(4*(h[1:-1,1:-1]**2)*(self.d**2)))*(h[2:,1:-1] - h[:-2,1:-1])
        C4 = np.pad(C4,((1,1)),constant_values=1)

        C5 = (-1/(h[1:-1,1:-1]*self.d**2)) - (1/(4*(h[1:-1,1:-1]**2)*(self.d**2)))*(h[2:,1:-1] - h[:-2,1:-1])
        C5 = np.pad(C5,((1,1)),constant_values=1)

        C6 = 4/(h[1:-1,1:-1]*self.d**2)
        C6 = np.pad(C6,((1,1)),constant_values=1)

    
        # create matrix to solve for psibar

        centre_ones = np.concatenate(([-1],np.zeros(self.NxG-2),[-1]))
        diags_ones = np.concatenate((-1*np.ones(self.NxG),np.tile(centre_ones,self.NyG-2),-1*np.ones(self.NxG)))
        diagonal_ones = sp.sparse.diags(diags_ones)

        centre = np.concatenate(([0],np.ones(self.NxG-2),[0]))

        diags = np.concatenate((np.zeros(self.NxG),np.tile(centre,self.NyG-2),np.zeros(self.NxG)))*C6.flatten()
        diagonal = sp.sparse.diags(diags)

        diags_pi = np.concatenate((np.zeros(self.NxG),np.tile(centre,self.NyG-2),np.zeros(self.NxG)))*C2.flatten()
        diag_pi = sp.sparse.diags(diags_pi[:len(diags_pi)-1],offsets=1)

        diags_mi = np.concatenate((np.zeros(self.NxG),np.tile(centre,self.NyG-2),np.zeros(self.NxG)))*C3.flatten()
        diag_mi = sp.sparse.diags(diags_mi[1:],offsets=-1)

        diags_pj = np.concatenate((np.zeros(self.NxG),np.tile(centre,self.NyG-2),np.zeros(self.NxG)))*C4.flatten()
        diag_pj = sp.sparse.diags(diags_pj[:len(diags_pi)-self.NxG],offsets=self.NxG)

        diags_mj = np.concatenate((np.zeros(self.NxG),np.tile(centre,self.NyG-2),np.zeros(self.NxG)))*C5.flatten()
        diag_mj = sp.sparse.diags(diags_mj[self.NxG:],offsets=-self.NxG)

        diagonal_total = diagonal + diag_pi + diag_mi + diag_pj + diag_mj + diagonal_ones
        self.diagonal = diagonal_total

        LU = linalg.splu(sp.sparse.csc_matrix(diagonal_total))
        self.LU = LU

    def calc_wind_stress(self):
        # calculate the wind stress to be added to F_n

        '''tau = np.array(self.tau)
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
        )'''

        dtau_dy = [((2*self.tau_0*np.pi)/(self.Ly)*np.sin((2*np.pi*y)/self.Ly))*np.ones(self.NxG) for y in self.YG]*(1/self.bathy)

        tau_xi = (-1/self.rho_0)*(dtau_dy - (self.tau*self.bathy.differentiate(coord='YG'))/(self.bathy**2))

        self.wind_stress = xr.DataArray(
            tau_xi,
            dims = ['YG','XG'],
            coords = {
                'YG': self.YG,
                'XG': self.XG
            }
        )


    

# %%
Nx = 200
Ny = 400
example_wind = Barotropic(d=5000,Nx=Nx,Ny=Ny,bathy=5000*np.ones((Ny+1,Nx+1)),f0=0.7E-4,beta=2.E-11)

#%%
example_wind.gen_init_psi(k_peak=2,const=1000)
example_wind.xi_from_psi()

#%%

X,Y = np.meshgrid(example_wind.XG/1000,example_wind.YG/1000)
fig,axs = plt.subplots(1,1)
im = axs.contour(X,Y,example_wind.psibar_0,cmap='RdBu',norm=colors.TwoSlopeNorm(vcenter=0))
axs.set_aspect(1)
cbar = plt.colorbar(im)
cbar.set_label('$\overline{\psi}$')
plt.xlabel('x (km)')
plt.ylabel('y (km)')
plt.title('Initial $\overline{\psi}$')
plt.show()

X,Y = np.meshgrid(example_wind.XG/1000,example_wind.YG/1000)
fig,axs = plt.subplots(1,1)
im = axs.contourf(X,Y,example_wind.xibar_0,levels=50,cmap='RdBu',norm=colors.TwoSlopeNorm(vcenter=0))
axs.set_aspect(1)
cbar = plt.colorbar(im)
cbar.set_label('$\overline{\\xi}$')
plt.xlabel('x (km)')
plt.ylabel('y (km)')
plt.title('Initial $\overline{\\xi}$')
plt.show()

#%%

example_wind_data = example_wind.model(dt=400,Nt=10,gamma_q=0,r_BD=1.E-7,r_diff=1,tau_0=0.1,rho_0=1000,dumpFreq=4000)



# %%

# %%
