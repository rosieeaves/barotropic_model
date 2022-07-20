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

    ##########################################################################
    # CLASS INITIALISATION
    ##########################################################################

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

            self.diagnosticsDict()
        
    ##########################################################################
    # INITIAL CONDITIONS FUNCTIONS
    ##########################################################################

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

    ##########################################################################
    # TIME STEPPING FUNCTIONS
    ##########################################################################


    def model(self,dt,Nt,gamma_q,r_BD,r_diff,tau_0,rho_0,dumpFreq,meanDumpFreq,diags):

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
            self.meanDumpFreq = meanDumpFreq
            self.T = np.arange(0,int(self.Nt*self.dt)+1,self.dumpFreq)

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

            # create mean parameters
            self.xibar_sum = self.xibar_0
            self.psibar_sum = self.psibar_0
            self.ubar_sum = self.ubar_0
            self.vbar_sum = self.vbar_0
            self.xi_MEAN = [np.zeros_like(self.xibar_0)]
            self.psi_MEAN = [np.zeros_like(self.psibar_0)]
            self.u_MEAN = [np.zeros_like(self.ubar_0)]
            self.v_MEAN = [np.zeros_like(self.vbar_0)]

            # create diagnostic parameters
            # method for calculating diagnostics e.g. xi_u
            # variables:
            # self.xi_u for saving snapshot data every dumpFreq
            # self.xi_u_np1 for calculating the next time step
            # self.xi_u_MEAN for saving mean data in every meanDumpFreq
            # self.xi_u_sum for summing the snapshot data to be used to calculate the mean
            # method:
            # calculate xi_u at every timestep and save new value in self.xi_u_np1
            # add self.xi_u_np1 to self.xi_u_sum at every timestep
            # after every dumpFreq seconds, save self.xi_u_np1 in self.xi_u
            # after every meanDumpFreq seconds, divide self.xi_u_sum by meanDumpFreq, save to self.xi_u_MEAN, set self.xi_u_sum go zeroes
            for var in diags:
                # create array to save data in every dumpFreq e.g. self.xi_u. 
                # Should be added to every dumpFreq seconds.
                setattr(self,var,[self.diagnosticsFunctsionDict[var](xi=self.xibar_0,psi=self.psibar_0,u=self.ubar_0,v=self.vbar_0)])
                # create variable for calculating np1 time step, e.g. self.xi_u_np1. Initialise with zeros. 
                # Should be updated every timestep.
                np1_str = var + '_np1'
                setattr(self,np1_str,np.zeros((len(getattr(self,self.diagnosticsCoordDict[var][1])),\
                    len(getattr(self,self.diagnosticsCoordDict[var][2])))))
                # create array to save MEAN data in, e.g. self.xi_u_MEAN. Initialise with zeros. 
                # Should be added to every meanDumpFreq seconds.
                setattr(self,self.diagnosticsMeanDict[var],[np.zeros((len(getattr(self,self.diagnosticsCoordDict[var][1])),\
                    len(getattr(self,self.diagnosticsCoordDict[var][2]))))])
                # create mean variable for summing snapshot data for use in calculating mean, e.g. self.xi_u_sum. 
                # Should be updated every timestep. 
                sum_str = var + '_sum'
                setattr(self,sum_str, self.diagnosticsFunctsionDict[var](xi=self.xibar_0,psi=self.psibar_0,u=self.ubar_0,v=self.vbar_0))

            
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
                    ubar_np1,vbar_np1 = self.calc_UV(psibar_np1)

                    # add to mean data
                    self.xibar_sum = self.xibar_sum + xibar_np1
                    self.psibar_sum = self.psibar_sum + psibar_np1
                    self.ubar_sum = self.ubar_sum + ubar_np1
                    self.vbar_sum = self.vbar_sum + vbar_np1

                    # calculate diagnostics and add to summed data
                    for var in diags:
                        # get variable name for calculating np1 timestep, e.g. self.xi_u_np1.
                        np1_str = var + '_np1'
                        # calculate value at np1
                        value_np1 = self.diagnosticsFunctsionDict[var](xi=xibar_np1,psi=psibar_np1,u=ubar_np1,v=vbar_np1)
                        # set value of var_np1, e.g. self.xi_u_np1, to value calculated
                        setattr(self,np1_str,value_np1)

                        # get sum variable name, e.g. self.xi_u_sum
                        sum_str = var + '_sum'
                        # add value at np1 timestep to sum
                        sum_value = getattr(self,sum_str) + value_np1
                        # set variable value to new sum
                        setattr(self,sum_str,sum_value)

                    # DUMP XIBAR AND PSIBAR AT CERTAIN INTERVALS

                    if kw.get('t')%kw.get('dumpFreq') == 0:
                        # calculate u and v from psi
                        u_np1,v_np1 = self.calc_UV(psibar_np1)

                        # save xi, psi, u and v
                        self.xibar = np.append(self.xibar,[xibar_np1],axis=0)
                        self.psibar = np.append(self.psibar,[psibar_np1],axis=0)
                        self.ubar = np.append(self.ubar,[u_np1],axis=0)
                        self.vbar = np.append(self.vbar,[v_np1],axis=0)

                        # run diagnostics 
                        for var in diags:
                            # get variable name at timestep np1, e.g. self.xi_u_np1
                            np1_str = var + '_np1'
                            # append np1 value to var data, e.g. self.xi_u, and reset var, e.g. self.xi_u, to appended array
                            setattr(self,var,np.append(getattr(self,var),[getattr(self,np1_str)],axis=0))

                    if kw.get('t')%kw.get('meanDumpFreq') == 0:

                        # dump mean data
                        self.xi_MEAN = np.append(self.xi_MEAN,[(self.xibar_sum*self.dt)/self.meanDumpFreq],axis=0)
                        self.psi_MEAN = np.append(self.psi_MEAN,[(self.psibar_sum*self.dt)/self.meanDumpFreq],axis=0)
                        self.u_MEAN = np.append(self.u_MEAN,[(self.ubar_sum*self.dt)/self.meanDumpFreq],axis=0)
                        self.v_MEAN = np.append(self.v_MEAN,[(self.vbar_sum*self.dt)/self.meanDumpFreq],axis=0)

                        # reset sum for mean to zero
                        self.xibar_sum = np.zeros_like(self.xibar_sum)
                        self.psibar_sum = np.zeros_like(self.psibar_sum)
                        self.ubar_sum = np.zeros_like(self.ubar_sum)
                        self.vbar_sum = np.zeros_like(self.vbar_sum)

                        # run diagnostics 
                        for var in diags:
                            # get variable name for summed data, e.g. self.xi_u_sum
                            sum_str = var + '_sum'
                            # append new mean value to mean variable and reset mean variable to new appended variable
                            setattr(self,self.diagnosticsMeanDict[var],\
                                np.append(getattr(self,self.diagnosticsMeanDict[var]),[(getattr(self,sum_str)*self.dt)/self.meanDumpFreq],axis=0))
                            # set sum to zero
                            setattr(self,sum_str,np.zeros_like(getattr(self,sum_str)))

                        
       
                    # reset values
                    self.F_nm2 = self.F_nm1.copy()
                    self.F_nm1 = self.F_n.copy()
                    self.F_n = np.zeros_like(self.F_n)
                    self.xibar_n = xibar_np1.copy()
                    self.psibar_n = psibar_np1.copy()

                return wrapper

            # calculate dump frequency in number of timesteps
            dumpFreqTS = int(self.dumpFreq/self.dt)
            meanDumpFreqTS = int(self.meanDumpFreq/self.dt)

            def forward_Euler(xibar_n,dt,F_n,F_nm1,F_nm2):
                xibar_np1 = xibar_n + dt*F_n
                return xibar_np1

            def AB3(xibar_n,dt,F_n,F_nm1,F_nm2):
                xibar_np1 = xibar_n + (dt/12)*(23*F_n - 16*F_nm1 + 5*F_nm2)
                return xibar_np1

            # first two time steps with forward Euler
            func_to_run = time_step(scheme=forward_Euler,t=1,dumpFreq=dumpFreqTS,meanDumpFreq=meanDumpFreqTS)
            func_to_run()

            func_to_run = time_step(scheme=forward_Euler,t=2,dumpFreq=dumpFreqTS,meanDumpFreq=meanDumpFreqTS)
            func_to_run()

            # all other time steps with AB3:
            for t in range(3,Nt+1):
                print('ts = ' + str(t))
                func_to_run = time_step(scheme=AB3,t=t,dumpFreq=dumpFreqTS,meanDumpFreq=meanDumpFreqTS)
                func_to_run()
                end_time = time.time_ns()

            self.T = np.arange(0,int(self.Nt*self.dt)+1,self.dumpFreq)
            self.T_MEAN = np.arange(1,int((self.Nt*self.dt)/self.meanDumpFreq)+1)
            
            dataset_return = xr.Dataset(
                data_vars = dict(
                    xi = xr.DataArray(
                        self.xibar,
                        dims = ['T','YG','XG'],
                        coords = {
                            'T': self.T,
                            'YG': self.YG,
                            'XG': self.XG
                        }
                    ),
                    xi_MEAN = xr.DataArray(
                        self.xi_MEAN[1:,:,:],
                        dims = ['T_MEAN','YG','XG'],
                        coords = {
                            'T_MEAN': self.T_MEAN,
                            'YG': self.YG,
                            'XG': self.XG
                        }
                    ),
                    psi = xr.DataArray(
                        self.psibar,
                        dims = ['T','YG','XG'],
                        coords = {
                            'T': self.T,
                            'YG': self.YG,
                            'XG': self.XG
                        }
                    ),
                    psi_MEAN = xr.DataArray(
                        self.psi_MEAN[1:,:,:],
                        dims = ['T_MEAN','YG','XG'],
                        coords = {
                            'T_MEAN': self.T_MEAN,
                            'YG': self.YG,
                            'XG': self.XG
                        }
                    ),
                    u = xr.DataArray(
                        self.ubar,
                        dims = ['T','YC','XG'],
                        coords = {
                            'T': self.T,
                            'YC': self.YC,
                            'XG': self.XG
                        }
                    ),
                    u_MEAN = xr.DataArray(
                        self.u_MEAN[1:,:,:],
                        dims = ['T_MEAN','YC','XG'],
                        coords = {
                            'T_MEAN': self.T_MEAN,
                            'YC': self.YC,
                            'XG': self.XG
                        }
                    ),
                    v = xr.DataArray(
                        self.vbar,
                        dims = ['T','YG','XC'],
                        coords = {
                            'T': self.T,
                            'YG': self.YG,
                            'XC': self.XC
                        }
                    ),
                    v_MEAN = xr.DataArray(
                        self.v_MEAN[1:,:,:],
                        dims = ['T_MEAN','YG','XC'],
                        coords = {
                            'T_MEAN': self.T_MEAN,
                            'YG': self.YG,
                            'XC': self.XC
                        }
                    )
                ),                    
                coords = dict(
                    T = self.T,
                    YG = self.YG,
                    XG = self.XG,
                    YC = self.YC,
                    XC = self.XC,
                    T_MEAN = self.T_MEAN,
                ),
                attrs = dict(
                    dt = self.dt,
                    Nt = self.Nt,
                    gamma_q = self.gamma_q,
                    r_BD = self.r_BD,
                    r_diff = self.r_diff,
                    tau_0 = self.tau_0,
                    rho_0 = self.rho_0,
                    dumpFreq = self.dumpFreq,
                    meanDumpFreq = self.meanDumpFreq,
                    dx = self.dx,
                    dy = self.dy,
                    Nx = self.Nx,
                    Ny = self.Ny,
                    f0 = self.f0,
                    beta = self.beta
                ) 

            )

            dataset_return['bathy'] = self.bathy
            dataset_return['f'] = self.f

            # add diagnostics data to dataset_return
            for var in diags:
                # snapshot data every dumpFreq seconds, e.g. self.xi_u
                dataset_return[var] = xr.DataArray(
                    getattr(self,var),
                    dims = self.diagnosticsCoordDict[var],
                    coords = {
                        self.diagnosticsCoordDict[var][0]: getattr(self,self.diagnosticsCoordDict[var][0]),
                        self.diagnosticsCoordDict[var][1]: getattr(self,self.diagnosticsCoordDict[var][1]),
                        self.diagnosticsCoordDict[var][2]: getattr(self,self.diagnosticsCoordDict[var][2])
                    }
                )
                # mean data every meanDumpFreq seconds, e.g. self.xi_u_MEAN
                dataset_return[self.diagnosticsMeanDict[var]] = xr.DataArray(
                    getattr(self,self.diagnosticsMeanDict[var])[1:,:,:],
                    dims = ['T_MEAN',self.diagnosticsCoordDict[var][1],\
                        self.diagnosticsCoordDict[var][2]],
                    coords = {
                        'T_MEAN': self.T_MEAN,
                        self.diagnosticsCoordDict[var][1]: getattr(self,self.diagnosticsCoordDict[var][1]),
                        self.diagnosticsCoordDict[var][2]: getattr(self,self.diagnosticsCoordDict[var][2])
                    }
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

        print('ellip start')

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
        
        print('ellip end')

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

    ##########################################################################
    # DIAGNOSTICS FUNCTIONS
    ##########################################################################

    def diagnosticsDict(self):

        self.diagnosticsFunctsionDict = {
            'xi_u': self.calc_xi_u,
            'xi_v': self.calc_xi_v,
            'u_u': self.calc_uSquare,
            'v_v': self.calc_vSquare
        }

        self.diagnosticsCoordDict = {
            'xi_u': ['T','YG','XG'],
            'xi_v': ['T','YG','XG'],
            'u_u': ['T', 'YC', 'XG'],
            'v_v': ['T', 'YG', 'XC']
        }

        self.diagnosticsMeanDict = {
            'xi_u': 'xi_u_MEAN',
            'xi_v': 'xi_v_MEAN',
            'u_u': 'u_u_MEAN',
            'v_v': 'v_v_MEAN'
        }

    def calc_xi_u(self,xi,psi,u,v):

        xi = np.array(xi)
        u = np.array(u)

        xi_u = xi[1:-1,1:-1]*((u[1:,1:-1] + u[:-1,1:-1])/2)
        xi_u = np.pad(xi_u,((1,1)),constant_values=0)

        return xi_u

    def calc_xi_v(self,xi,psi,u,v):

        xi = np.array(xi)
        v = np.array(v)

        xi_v = xi[1:-1,1:-1]*((v[1:-1,1:] + v[1:-1,:-1])/2)
        xi_v = np.pad(xi_v,((1,1)),constant_values=0)

        return xi_v
            
    def calc_uSquare(self,xi,psi,u,v):

        return u**2

    def calc_vSquare(self,xi,psi,u,v):

        return v**2



# %%
'''H = 5000
h = 500
dx = 2000
dy = 2000
Nx = 500
Ny = 500
Lx = dx*Nx 
Ly = dy*Ny

bathy_mount = [[(H-h*np.sin((np.pi*(i*dx))/Lx)*np.sin((np.pi*(j*dy))/Ly)) for i in range(Nx+1)] for j in range(Ny+1)]

test_diags = Barotropic(d=2000,Nx=Nx,Ny=Ny,bathy=bathy_mount,f0=0.7E-4,beta=2.E-11)

test_diags.gen_init_psi(k_peak=2,const=1E12)
test_diags.xi_from_psi()

X,Y = np.meshgrid(test_diags.XG/1000,test_diags.YG/1000)

plt.contourf(X,Y,test_diags.xibar_0)
plt.colorbar()
plt.show()

plt.contourf(X,Y,test_diags.psibar_0)
plt.colorbar()
plt.show()

plt.contourf(test_diags.XG/1000,test_diags.YC/1000,test_diags.ubar_0)
plt.colorbar()
plt.show()

#%%

diagnostics = ['xi_u','xi_v','u_u','v_v']
data = test_diags.model(dt=900,Nt=10,gamma_q=0,r_BD=0,r_diff=2,tau_0=0,rho_0=1000,dumpFreq=1800,meanDumpFreq=1800,diags=diagnostics)
# %%

print(data)
# %%

plt.contourf(data.XG/1000,data.YC/1000,data.u[5])
plt.colorbar()
plt.show()


# %%

print(np.array_equal(data.u_u,data.u**2))
print(np.array_equal(data.v_v,data.v**2))


# %%
u = np.array(data.u)
v = np.array(data.v)
xi = np.array(data.xi)

for t in range(len(u)):
    xi_u = xi[t,1:-1,1:-1]*((u[t,1:,1:-1] + u[t,:-1,1:-1])/2)
    xi_u = np.pad(xi_u,((1,1)),constant_values=0)
    print(np.array_equal(xi_u,data.xi_u[t]))

for t in range(len(v)):
    xi_v = xi[t,1:-1,1:-1]*((v[t,1:-1,1:] + v[t,1:-1,:-1])/2)
    xi_v = np.pad(xi_v,((1,1)),constant_values=0)
    print(np.array_equal(xi_v,data.xi_v[t]))'''


# %%
