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
            self.NxC = self.Nx 
            self.NyC = self.Ny
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

            self.bathy_np = np.array(self.bathy)

            self.bathy_YGXG = (self.bathy_np[:-1,:-1] + self.bathy_np[1:,:-1] + \
                
                self.bathy_np[:-1,1:] + self.bathy_np[1:,1:])/4

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

        self.psibar_0 = stream
        
        '''self.psibar_0 = xr.DataArray(
            stream,
            dims = ['YG','XG'],
            coords = {
                'YG': self.YG,
                'XG': self.XG
            }
        )'''

    def init_psi(self,psi):
        try:
            np.shape(psi) == ((self.NyG,self.NxG))
        except ValueError:
            print('Initial psi must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
        else:
            # set zero on boundaries
            psi_remove = psi[1:-1,1:-1]
            psi_pad = np.pad(psi_remove,((1,1)),constant_values=0)
            self.psibar_0 = psi_pad
            self.ubar_0,self.vbar_0 = self.calc_UV(self.psibar_0)
            #self.ubar_0 = self.ubar_np1
            #self.vbar_0 = self.vbar_np1

    def init_xi(self,xi):
        try:
            np.shape(xi) == ((self.NyG,self.NxG))
        except ValueError:
            print('Initial xi must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
        else:
            # set zero on boundaries
            xi_remove = xi[1:-1,1:-1]
            xi_pad = np.pad(xi_remove,((1,1)),constant_values=0)
            self.xibar_0 = xi_pad
            '''self.xibar_0 = xr.DataArray(
                xi_pad,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )'''

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

            self.psibar_0 = psibar

            '''self.psibar_0 = xr.DataArray(
                psibar,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )'''

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

            print(self.psibar_0)

            xibar_0 = (self.psibar_0[1:-1,2:] + self.psibar_0[1:-1,:-2] + \
                self.psibar_0[2:,1:-1] + self.psibar_0[:-2,1:-1] - 4*self.psibar_0[1:-1,1:-1])/(self.bathy_np[1:-1,1:-1]*(self.d**2)) - \
                    ((self.psibar_0[1:-1,2:] - self.psibar_0[1:-1,:-2])*(self.bathy_np[1:-1,2:] - self.bathy_np[1:-1,:-2]) + \
                        (self.psibar_0[2:,1:-1] - self.psibar_0[:-2,1:-1])*\
                            (self.bathy_np[2:,1:-1] - self.bathy_np[:-2,1:-1]))/((self.bathy_np[1:-1,1:-1]**2)*(self.d**2))
 
            xibar_0 = np.pad(xibar_0,((1,1)),constant_values=0)

            self.xibar_0 = xibar_0

            '''self.xibar_0 = xr.DataArray(
                xibar_0,
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )'''

            self.ubar_0,self.vbar_0 = self.calc_UV(self.psibar_0)

    ##########################################################################
    # TIME STEPPING FUNCTIONS
    ##########################################################################


    def model(self,dt,Nt,gamma_q,r_BD,kappa_xi,tau_0,rho_0,dumpFreq,meanDumpFreq,kappa_q=None,kappa_K=None,\
        scheme=False,init_K=None,init_Q=None,diags=[],tracer_init=None):

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

            if scheme != False and (init_K.any() == None or init_Q.any() == None):
                raise ValueError('init_K and init_Q parameter must be specified when scheme = True')

            if scheme == 'EGECAD' or scheme == 'EGECAD_TEST':
                if kappa_q == None or kappa_K == None:
                    raise ValueError('kappa_q and kappa_K parameters must be set when using scheme woth advection and diffsion.')

            # store model parameters in object
            self.dt = dt
            self.Nt = Nt
            self.gamma_q = gamma_q
            self.r_BD = r_BD
            self.kappa_xi = kappa_xi
            self.tau_0 = tau_0 
            self.rho_0 = rho_0
            self.dumpFreq = dumpFreq
            self.meanDumpFreq = meanDumpFreq
            self.T = np.arange(0,int(self.Nt*self.dt)+1,self.dumpFreq)
            self.kappa_q = kappa_q
            self.kappa_K = kappa_K
            self.scheme = scheme

            # if initial u and v do not already exist, calculate them
            try:
                self.ubar_0
            except:
                self.ubar_0,self.vbar_0 = self.calc_UV(self.psibar_0)

            # initialise parameters pisbar, zetabar, and grad xibar components
            self.psibar_n = self.psibar_0
            self.psibar_np1 = np.zeros_like(self.psibar_n)
            self.xibar_n = self.xibar_0
            self.xibar_np1 = np.zeros_like(self.xibar_n)
            self.ubar_n = self.ubar_0
            self.ubar_np1 = np.zeros_like(self.ubar_n)
            self.vbar_n = self.vbar_0
            self.vbar_np1 = np.zeros_like(self.vbar_n)

            # calculate length of T and T_MEAN arrays
            len_T = int((self.Nt*self.dt)/self.dumpFreq)
            len_T_MEAN = int((self.Nt*self.dt)/self.meanDumpFreq)

            # create arrays to save the data
            # NOTE: len_T+1 so that you can add the value at time = 0 in
            self.xibar = np.zeros((len_T+1,self.NyG,self.NxG),dtype=object)
            self.psibar = np.zeros((len_T+1,self.NyG,self.NxG),dtype=object)
            self.ubar = np.zeros((len_T+1,self.NyC,self.NxG),dtype=object)
            self.vbar = np.zeros((len_T+1,self.NyG,self.NxC),dtype=object)
            self.xibar[0] = self.xibar_0
            self.psibar[0] = self.psibar_0
            self.ubar[0] = self.ubar_0
            self.vbar[0] = self.vbar_0

            # create mean parameters
            self.xibar_sum = self.xibar_0
            self.psibar_sum = self.psibar_0
            self.ubar_sum = self.ubar_0
            self.vbar_sum = self.vbar_0
            self.xi_MEAN = np.zeros((len_T_MEAN,self.NyG,self.NxG),dtype=object)
            self.psi_MEAN = np.zeros((len_T_MEAN,self.NyG,self.NxG),dtype=object)
            self.u_MEAN = np.zeros((len_T_MEAN,self.NyC,self.NxG),dtype=object)
            self.v_MEAN = np.zeros((len_T_MEAN,self.NyG,self.NxC),dtype=object)

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
                num_Y = len(getattr(self,self.diagnosticsCoordDict[var][1]))
                num_X = len(getattr(self,self.diagnosticsCoordDict[var][2]))
                setattr(self,var,np.zeros((len_T+1,num_Y,num_X),dtype=object))
                value_0 = self.diagnosticsFunctionsDict[var](xi=self.xibar_0,psi=self.psibar_0,u=self.ubar_0,v=self.vbar_0)
                getattr(self,var)[0] = value_0
                # create variable for calculating np1 time step, e.g. self.xi_u_np1. Initialise with zeros. 
                # Should be updated every timestep.
                setattr(self,self.diagnosticsNP1Dict[var],np.zeros((num_Y,num_X)))
                # create array to save MEAN data in, e.g. self.xi_u_MEAN. Initialise with zeros. 
                # Should be added to every meanDumpFreq seconds.
                setattr(self,self.diagnosticsMeanDict[var],np.zeros((len_T_MEAN,num_Y,num_X),dtype=object))
                # create variable for summing snapshot data for use in calculating mean, e.g. self.xi_u_sum. 
                # Should be updated every timestep. 
                setattr(self,self.diagnosticsSumDict[var],value_0)
            
            # initialise parameters for time stepping
            self.F_n = np.zeros_like(self.xibar_n)
            self.F_nm1 = np.zeros_like(self.xibar_n)
            self.F_nm2 = np.zeros_like(self.xibar_n)
            self.adv_n = np.zeros_like(self.xibar_n)
            self.BD_n = np.zeros_like(self.xibar_n)
            self.diff_n = np.zeros_like(self.xibar_n)
            self.zeta_n = np.zeros_like(self.xibar_n)
            self.psi_YCXC = np.zeros((self.Ny,self.Nx))

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

            if scheme != False:
                # enstrophy variables
                self.Q_0 = init_Q
                self.Q_n = self.Q_0
                self.Q_np1 = np.zeros_like(self.Q_0)
                self.Q = np.zeros((len_T+1,self.NyG,self.NxG),dtype=object)
                self.Q[0] = self.Q_0
                self.Q_F_n = np.zeros_like(self.Q_0)
                self.Q_F_nm1 = np.zeros_like(self.Q_0)
                self.Q_F_nm2 = np.zeros_like(self.Q_0)
                self.Q_sum = self.Q_0
                self.Q_MEAN = np.zeros((len_T_MEAN,self.NyG,self.NxG),dtype=object)
                self.QAdv_n = np.zeros_like(self.Q_0)
                self.QDiff_n = np.zeros_like(self.Q_0)

                # energy variables
                self.K_0 = init_K
                self.K_n = self.K_0
                self.K_np1 = np.zeros_like(self.K_n)
                self.K = np.zeros((len_T+1,self.NyG,self.NxG),dtype=object)
                self.K[0] = self.K_0
                self.K_F_n = np.zeros_like(self.K_n)
                self.K_F_nm1 = np.zeros_like(self.K_n)
                self.K_F_nm2 = np.zeros_like(self.K_n)
                self.K_sum = self.K_0
                self.K_MEAN = np.zeros((len_T_MEAN,self.NyG,self.NxG),dtype=object)
                self.KAdv_n = np.zeros_like(self.K_n)
                self.KDiff_n = np.zeros_like(self.K_n)

                # variables for parameterization calculation 
                self.qbar_n = np.zeros_like(self.Q_0)
                self.qbar_dx_n = np.zeros_like(self.Q_0)
                self.qbar_dy_n = np.zeros_like(self.Q_0)
                self.mod_grad_qbar_n = np.zeros_like(self.Q_0)
                self.alpha_n = np.zeros_like(self.Q_0)
                self.psi_dx_n = np.zeros_like(self.psibar_0)
                self.psi_dy_n = np.zeros_like(self.psibar_0)
                self.mod_grad_qbar = np.zeros((len_T+1,self.NyG,self.NxG),dtype=object)
                self.alpha = np.zeros((len_T+1,self.NyG,self.NxG),dtype=object)
                self.qbar_dx = np.zeros((len_T+1,self.NyG,self.NxG),dtype=object)
                self.qbar_dy = np.zeros((len_T+1,self.NyG,self.NxG),dtype=object)

                self.flux_u_n = np.zeros_like(self.xibar_n)
                self.flux_v_n = np.zeros_like(self.xibar_n)
            else:
                self.Q_0 = np.zeros_like(self.xibar_n)
                self.Q_n = np.zeros_like(self.xibar_n)
                self.Q_np1 = np.zeros_like(self.xibar_n)
                self.K_0 = np.zeros_like(self.xibar_n)
                self.K_n = np.zeros_like(self.xibar_n)
                self.K_np1 = np.zeros_like(self.xibar_n)
            # run scheemeDict function to get schemeFunctionsDict which contains functions to run if scheme is True or False
            self.schemeDict()
            self.eddyFluxes_n = np.zeros_like(self.xibar_n)


            # time stepping function

            #def time_step(scheme,F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1):
            def time_step(scheme,**kw):

                def wrapper(*args,**kwargs):

                    # CALCULATE VORTICITY AT NEXT TIME STEP
                    # NOTE: advection_xi must always happen first so that psi_YCXC_n can be calculated and used in other functions
                    # calculate advection term
                    self.advection_xi()
                    # dissiaption from bottom drag
                    self.BD_n = self.r_BD*self.xibar_n
                    # diffusion of vorticity
                    self.diffusion(var=self.xibar_n,kappa=self.kappa_xi,var_return='diff_n')
                    # eddy fluxes
                    self.schemeFunctionsDict['calc_K_Q'](scheme=scheme)
                    self.schemeFunctionsDict['calc_eddyFluxes']()
                    # F_n
                    self.F_n = -self.adv_n - self.eddyFluxes_n - self.BD_n + self.diff_n + np.array(self.wind_stress)
                    # calculate new value of xibar
                    self.xibar_np1 = scheme(var_n=self.xibar_n,dt=self.dt,F_n=self.F_n,F_nm1=self.F_nm1,F_nm2=self.F_nm2)

                    # uncomment for solid body rotation
                    # self.psibar_np1 = self.psibar_n
                    # self.calc_UV(self.psibar_np1)

                    # SOLVE FOR PSI AT NEXT TIME STEP
                    # comment out for solid body rotation
                    self.psibar_np1 = self.LU.solve((-np.array(self.xibar_np1)).flatten())
                    self.psibar_np1 = self.psibar_np1.reshape((self.NyG,self.NxG))
                    self.ubar_np1,self.vbar_np1 = self.calc_UV(self.psibar_np1)

                    # ADD TO SUM FOR MEAN DATA
                    self.xibar_sum = self.xibar_sum + self.xibar_np1
                    self.psibar_sum = self.psibar_sum + self.psibar_np1
                    self.ubar_sum = self.ubar_sum + self.ubar_np1
                    self.vbar_sum = self.vbar_sum + self.vbar_np1

                    # DIAGNOSTICS
                    for var in diags:
                        # set np1 variable, e.g. self.xi_u_np1, to value at next timestep 
                        setattr(self,self.diagnosticsNP1Dict[var],\
                            self.diagnosticsFunctionsDict[var](xi=self.xibar_np1,psi=self.psibar_np1,u=self.ubar_np1,v=self.vbar_np1))
                        # set sum variable, e.g. self.xi_u_sum, to sum + np1 e.g. self.xi_u_sum + self.xi_u_np1
                        setattr(self,self.diagnosticsSumDict[var],getattr(self,self.diagnosticsSumDict[var])+getattr(self,self.diagnosticsNP1Dict[var]))
                    

                    # DUMP DATA EVERY dumpFreq SECONDS
                    if kw.get('t')%kw.get('dumpFreq') == 0:
                        index = int(kw.get('t')/kw.get('dumpFreq'))
                        # save xi, psi, u and v
                        self.xibar[index] = self.xibar_np1
                        self.psibar[index] = self.psibar_np1
                        self.ubar[index] = self.ubar_np1
                        self.vbar[index] = self.vbar_np1

                        # run diagnostics 
                        for var in diags:
                            # set snapshot data variable, e.g. self.xi_u, to append(old snapshot data, np1 value), e.g. append self.xi_u with self.xi_u_np1
                            getattr(self,var)[index] = getattr(self,self.diagnosticsNP1Dict[var])

                        if self.scheme != False:

                            self.Q[index] = self.Q_np1
                            self.K[index] = self.K_np1
                            self.mod_grad_qbar[index] = self.mod_grad_qbar_n
                            self.alpha[index] = self.alpha_n
                            self.qbar_dx[index] = self.qbar_dx_n
                            self.qbar_dy[index] = self.qbar_dy_n

                    # DUMP MEAN DATA
                    if kw.get('t')%kw.get('meanDumpFreq') == 0:
                        index_MEAN = int(kw.get('t')/kw.get('meanDumpFreq')) - 1
                        print(index_MEAN)
                        # dump mean data
                        self.xi_MEAN[index_MEAN] = (self.xibar_sum*self.dt)/self.meanDumpFreq
                        self.psi_MEAN[index_MEAN] = (self.psibar_sum*self.dt)/self.meanDumpFreq
                        self.u_MEAN[index_MEAN] = (self.ubar_sum*self.dt)/self.meanDumpFreq
                        self.v_MEAN[index_MEAN] = (self.vbar_sum*self.dt)/self.meanDumpFreq

                        # reset sum for mean to zero
                        self.xibar_sum = np.zeros_like(self.xibar_sum)
                        self.psibar_sum = np.zeros_like(self.psibar_sum)
                        self.ubar_sum = np.zeros_like(self.ubar_sum)
                        self.vbar_sum = np.zeros_like(self.vbar_sum)

                        if self.scheme != False:
                            self.Q_MEAN[index_MEAN] = (self.Q_sum*self.dt)/self.meanDumpFreq
                            self.K_MEAN[index_MEAN] = (self.K_sum*self.dt)/self.meanDumpFreq

                            self.Q_sum = np.zeros_like(self.Q_sum)
                            self.K_sum = np.zeros_like(self.K_sum)


                        # run diagnostics 
                        for var in diags:
                            # set mean data, e.g. self.xi_u_MEAN, to append(old mean data, new mean data), e.g. append self.xi_u_MEAN with self.xi_u_sum/time
                            getattr(self,self.diagnosticsMeanDict[var])[index_MEAN] = (getattr(self,self.diagnosticsSumDict[var])*self.dt)/self.meanDumpFreq
                            # set sum variable, e.g. self.xi_u_sum, to zero
                            setattr(self,self.diagnosticsSumDict[var],np.zeros_like(getattr(self,self.diagnosticsSumDict[var])))

                        
       
                    # reset values
                    self.F_nm2 = self.F_nm1.copy()
                    self.F_nm1 = self.F_n.copy()
                    self.F_n = np.zeros_like(self.F_n)
                    self.xibar_n = self.xibar_np1.copy()
                    self.psibar_n = self.psibar_np1.copy()

                    if self.scheme != False:

                        self.Q_F_nm2 = self.Q_F_nm1.copy()
                        self.Q_F_nm1 = self.Q_F_n.copy()
                        self.Q_F_n = np.zeros_like(self.Q_F_n)
                        self.Q_n = self.Q_np1.copy()
                        self.Q_np1 = np.zeros_like(self.Q_np1)

                        self.K_F_nm2 = self.K_F_nm1.copy()
                        self.K_F_nm1 = self.K_F_n.copy()
                        self.K_F_n = np.zeros_like(self.K_F_n)
                        self.K_n = self.K_np1.copy()
                        self.K_np1 = np.zeros_like(self.K_np1)
                        

                return wrapper

            # calculate dump frequency in number of timesteps
            dumpFreqTS = int(self.dumpFreq/self.dt)
            meanDumpFreqTS = int(self.meanDumpFreq/self.dt)

            def forward_Euler(var_n,dt,F_n,F_nm1,F_nm2):
                var_np1 = var_n + dt*F_n
                return var_np1

            def AB3(var_n,dt,F_n,F_nm1,F_nm2):
                var_np1 = var_n + (dt/12)*(23*F_n - 16*F_nm1 + 5*F_nm2)
                return var_np1

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

            self.T = np.arange(0,int(self.Nt*self.dt)+1,self.dumpFreq)
            self.T_MEAN = np.arange(1,int((self.Nt*self.dt)/self.meanDumpFreq)+1)

            
            dataset_return = xr.Dataset(                    
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
            if kappa_q != None:
                dataset_return['kappa_q'] = self.kappa_q
            if kappa_K != None:
                dataset_return['kappa_K'] = self.kappa_K

            dataset_return['bathy'] = self.bathy
            dataset_return['f'] = self.f

            dataset_return['xi'] = xr.DataArray(
                self.xibar,
                dims = ['T','YG','XG'],
                coords = {
                    'T': self.T,
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['xi_MEAN'] = xr.DataArray(
                self.xi_MEAN,
                dims = ['T_MEAN','YG','XG'],
                coords = {
                    'T_MEAN': self.T_MEAN,
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['psi'] = xr.DataArray(
                self.psibar,
                dims = ['T','YG','XG'],
                coords = {
                    'T': self.T,
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['psi_MEAN'] = xr.DataArray(
                self.psi_MEAN,
                dims = ['T_MEAN','YG','XG'],
                coords = {
                    'T_MEAN': self.T_MEAN,
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['u'] = xr.DataArray(
                self.ubar,
                dims = ['T','YC','XG'],
                coords = {
                    'T': self.T,
                    'YC': self.YC,
                    'XG': self.XG
                }
            )

            dataset_return['u_MEAN'] = xr.DataArray(
                self.u_MEAN,
                dims = ['T_MEAN','YC','XG'],
                coords = {
                    'T_MEAN': self.T_MEAN,
                    'YC': self.YC,
                    'XG': self.XG
                }
            )

            dataset_return['v'] = xr.DataArray(
                self.vbar,
                dims = ['T','YG','XC'],
                coords = {
                    'T': self.T,
                    'YG': self.YG,
                    'XC': self.XC
                }
            )

            dataset_return['v_MEAN'] = xr.DataArray(
                self.v_MEAN,
                dims = ['T_MEAN','YG','XC'],
                coords = {
                    'T_MEAN': self.T_MEAN,
                    'YG': self.YG,
                    'XC': self.XC
                }
            )
            

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
                    getattr(self,self.diagnosticsMeanDict[var]),
                    dims = ['T_MEAN',self.diagnosticsCoordDict[var][1],\
                        self.diagnosticsCoordDict[var][2]],
                    coords = {
                        'T_MEAN': self.T_MEAN,
                        self.diagnosticsCoordDict[var][1]: getattr(self,self.diagnosticsCoordDict[var][1]),
                        self.diagnosticsCoordDict[var][2]: getattr(self,self.diagnosticsCoordDict[var][2])
                    }
                )
            if self.scheme != False:

                dataset_return['Q'] = xr.DataArray(
                    self.Q,
                    dims = ['T','YG','XG'],
                    coords = {
                        'T': self.T,
                        'YG': self.YG,
                        'XG': self.XG
                    }
                )

                dataset_return['Q_MEAN'] = xr.DataArray(
                    self.Q_MEAN,
                    dims = ['T_MEAN','YG','XG'],
                    coords = {
                        'T_MEAN': self.T_MEAN,
                        'YG': self.YG,
                        'XG': self.XG
                    }
                )

                dataset_return['K'] = xr.DataArray(
                    self.K,
                    dims = ['T','YG','XG'],
                    coords = {
                        'T': self.T,
                        'YG': self.YG,
                        'XG': self.XG
                    }
                )

                dataset_return['K_MEAN'] = xr.DataArray(
                    self.K_MEAN,
                    dims = ['T_MEAN','YG','XG'],
                    coords = {
                        'T_MEAN': self.T_MEAN,
                        'YG': self.YG,
                        'XG': self.XG
                    }
                )

            return dataset_return

    def calc_UV(self,psi):

        vbar_np1 = (2/self.d)*((psi[:,1:] - psi[:,:-1])/(self.bathy_np[:,:-1] + self.bathy_np[:,1:]))
        ubar_np1 = -(2/self.d)*((psi[1:,:] - psi[:-1,:])/(self.bathy_np[:-1,:] + self.bathy_np[1:,:]))

        return ubar_np1,vbar_np1

    def dy_YGXG_pad0(self,var):
        var_dy = (var[2:,:] - var[:-2,:])/(2*self.dy)
        var_dy = np.pad(var_dy,((1,1),(0,0)),constant_values=0)
        return var_dy 

    def dx_YGXG_pad0(self,var):
        var_dx = (var[:,2:] - var[:,:-2])/(2*self.dx)
        var_dx = np.pad(var_dx,((0,0),(1,1)),constant_values=0)
        return var_dx 

    def advection_xi(self):

        # calculate absolute velocity
        self.zeta_n = np.array(self.xibar_n + self.f)

        # interpolate psi onto cell centres
        self.psiYCXC_n = (self.psibar_n[:-1,:-1] + self.psibar_n[:-1,1:] + self.psibar_n[1:,:-1] + self.psibar_n[1:,1:])/4

        # calculate area average of advection term 
        # area average is take over the grid cell centred at the vorticity point, away from the boundaries
        self.adv_n = (1/(self.d**2))*(((self.psiYCXC_n[1:,1:] - self.psiYCXC_n[1:,:-1])*\
            (self.zeta_n[1:-1,1:-1] + self.zeta_n[2:,1:-1]))/(self.bathy_np[1:-1,1:-1] + self.bathy_np[2:,1:-1]) - \
                ((self.psiYCXC_n[1:,1:] - self.psiYCXC_n[:-1,1:])*\
                    (self.zeta_n[1:-1,1:-1] + self.zeta_n[1:-1,2:]))/(self.bathy_np[1:-1,1:-1] + self.bathy_np[1:-1,2:]) - \
                        ((self.psiYCXC_n[:-1,1:] - self.psiYCXC_n[:-1,:-1])*\
                            (self.zeta_n[1:-1,1:-1] + self.zeta_n[:-2,1:-1]))/(self.bathy_np[1:-1,1:-1] + self.bathy_np[:-2,1:-1]) + \
                                ((self.psiYCXC_n[1:,:-1] - self.psiYCXC_n[:-1,:-1])*\
                                    (self.zeta_n[1:-1,1:-1] + self.zeta_n[1:-1,:-2]))/(self.bathy_np[1:-1,1:-1] + self.bathy_np[1:-1,:-2]))

        # pad with zero values on boundaries
        self.adv_n = np.pad(self.adv_n,((1,1)),constant_values=0)
    
    def diffusion(self,var,kappa,var_return): 
        var = np.array(var)

        diff_n = (1/self.d**2)*(var[1:-1,2:] + var[1:-1,:-2] + \
            var[2:,1:-1] + var[:-2,1:-1] - 4*var[1:-1,1:-1])

        diff_n = (np.pad(diff_n,((1,1)),constant_values=0))*kappa

        setattr(self,var_return,diff_n)

    def tracer_advect(self,var,var_return):
        var = np.array(var)

        adv_n = (1/(2*self.d**2))*(\
            (var[1:-1,1:-1] + var[2:,1:-1])*(self.psiYCXC_n[1:,1:] - self.psiYCXC_n[1:,:-1]) - \
                (var[1:-1,1:-1] + var[1:-1,2:])*(self.psiYCXC_n[1:,1:] - self.psiYCXC_n[:-1,1:]) - \
                    (var[1:-1,1:-1] + var[:-2,1:-1])*(self.psiYCXC_n[:-1,1:] - self.psiYCXC_n[:-1,:-1]) + \
                        (var[1:-1,1:-1] + var[1:-1,:-2])*(self.psiYCXC_n[1:,:-1] - self.psiYCXC_n[:-1,:-1]))

        setattr(self,var_return,np.pad(adv_n,((1,1)),constant_values=0))


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

        self.diagnosticsFunctionsDict = {
            'xi_u': self.calc_xi_u,
            'xi_v': self.calc_xi_v,
            'u_u': self.calc_uSquare,
            'v_v': self.calc_vSquare,
            'xi_xi': self.calc_xiSquare
        }

        self.diagnosticsCoordDict = {
            'xi_u': ['T','YG','XG'],
            'xi_v': ['T','YG','XG'],
            'u_u': ['T', 'YC', 'XG'],
            'v_v': ['T', 'YG', 'XC'],
            'xi_xi': ['T', 'YG', 'XG']
        }

        self.diagnosticsMeanDict = {
            'xi_u': 'xi_u_MEAN',
            'xi_v': 'xi_v_MEAN',
            'u_u': 'u_u_MEAN',
            'v_v': 'v_v_MEAN',
            'xi_xi': 'xi_xi_MEAN'
        }

        self.diagnosticsNP1Dict = {
            'xi_u': 'xi_u_np1',
            'xi_v': 'xi_v_np1',
            'u_u': 'u_u_np1',
            'v_v': 'v_v_np1',
            'xi_xi': 'xi_xi_np1'
        }

        self.diagnosticsSumDict = {
            'xi_u': 'xi_u_sum',
            'xi_v': 'xi_v_sum',
            'u_u': 'u_u_sum',
            'v_v': 'v_v_sum',
            'xi_xi': 'xi_xi_sum'
        }

    def calc_xi_u(self,xi,psi,u,v):

        xi_u = xi[1:-1,1:-1]*((u[1:,1:-1] + u[:-1,1:-1])/2)
        xi_u = np.pad(xi_u,((1,1)),constant_values=0)

        return xi_u

    def calc_xi_v(self,xi,psi,u,v):

        xi_v = xi[1:-1,1:-1]*((v[1:-1,1:] + v[1:-1,:-1])/2)
        xi_v = np.pad(xi_v,((1,1)),constant_values=0)

        return xi_v

    def calc_uSquare(self,xi,psi,u,v):

        return u**2

    def calc_vSquare(self,xi,psi,u,v):

        return v**2

    def calc_xiSquare(self,xi,psi,u,v):

        return xi**2

    ##########################################################################
    # PARAMETERIZATION
    ##########################################################################

    def schemeDict(self):
        if self.scheme != 'EGEC' and self.scheme != 'EGEC_TEST' and self.scheme != 'EGECAD' and self.scheme != 'EGECAD_TEST' \
            and self.scheme != 'Constant' and self.scheme != False:
            raise ValueError('scheme parameter should be set to \'EGEC\', \'EGECAD\' or False.')

        if self.scheme == 'EGEC':
            print('Energy conversion and enstrophy generation only. Calculates eddy fluxes.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_EGEC,
                'calc_eddyFluxes': self.eddyFluxes_scheme
            }
        elif self.scheme == 'EGEC_TEST':
            print('Energy conversion and enstrophy generation only. Doesn\'t calculate eddy fluxes.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_EGEC,
                'calc_eddyFluxes': self.noEddyFluxes
            }
        elif self.scheme == 'EGECAD':
            print('Full scheme with advection and diffusion. Calculates eddy fluxes.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_EGECAD,
                'calc_eddyFluxes': self.eddyFluxes_scheme
            }
        elif self.scheme == 'EGECAD_TEST':
            print('Full scheme with advection and diffusion. Doesn\'t calculate eddy fluxes.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_EGECAD,
                'calc_eddyFluxes': self.noEddyFluxes
            }
        elif self.scheme == 'Constant':
            print('Constant value of K and Q used (i.e. their initial values).')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_constant, 
                'calc_eddyFluxes': self.eddyFluxes_scheme
            }
        else:
            print('No scheme.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.no_K_Q,
                'calc_eddyFluxes': self.noEddyFluxes
            }

    def K_Q_EGEC(self,scheme):

        # NOTE: we use the values at time step n here to calculate dQ/dt_n and dK/dt_n so that we can claculate Q_np1 and K_np1
        # parameterized terms 
        self.qbar_n = np.array((self.f + self.xibar_n)/self.bathy_np)
        self.qbar_dx_n = self.dx_YGXG_pad0(var=self.qbar_n)
        self.qbar_dy_n = self.dy_YGXG_pad0(var=self.qbar_n)
        self.mod_grad_qbar_n = np.sqrt(self.qbar_dx_n**2 + self.qbar_dy_n**2)
        self.mod_grad_qbar_n = np.where(self.mod_grad_qbar_n < 1.E-18, np.ones_like(self.mod_grad_qbar_n)*1.E-18, self.mod_grad_qbar_n)

        self.psi_dx_n = self.dx_YGXG_pad0(var=self.psibar_n)
        self.psi_dy_n = self.dy_YGXG_pad0(var=self.psibar_n)

        self.alpha_n = (-2*self.gamma_q*np.sqrt(self.Q_n*self.K_n))    

        # dQdt and dKdt
        self.Q_F_n = -self.alpha_n*self.mod_grad_qbar_n 
        self.K_F_n = (self.alpha_n/self.mod_grad_qbar_n)*(self.qbar_dx_n*self.psi_dx_n + self.qbar_dy_n*self.psi_dy_n)  
        self.K_F_n = np.pad(self.K_F_n[1:-1,1:-1],((1,1)),constant_values=0)
        
        # step forward Q and K
        self.Q_np1 = scheme(var_n=self.Q_n,dt=self.dt,F_n=self.Q_F_n,F_nm1=self.Q_F_nm1,F_nm2=self.Q_F_nm2)
        self.K_np1 = scheme(var_n=self.K_n,dt=self.dt,F_n=self.K_F_n,F_nm1=self.K_F_nm1,F_nm2=self.K_F_nm2)
        # set K = 0 where K is negative
        self.K_np1 = np.where(self.K_np1 < 0,0,self.K_np1)

        # add to sum variables
        self.Q_sum = self.Q_sum + self.Q_np1
        self.K_sum = self.K_sum + self.K_np1

    def K_Q_EGECAD(self,scheme):
        # advect enstrophy and energy
        self.tracer_advect(var=self.Q_n,var_return='QAdv_n')
        self.tracer_advect(var=self.K_n,var_return='KAdv_n')

        # diffuse energy and enstrophy
        self.diffusion(var=self.Q_n,kappa=self.kappa_q,var_return='QDiff_n')
        self.diffusion(var=self.K_n,kappa=self.kappa_K,var_return='KDiff_n')

        # NOTE: we use the values at time step n here to calculate dQ/dt_n and dK/dt_n so that we can claculate Q_np1 and K_np1
        # parameterized terms 
        self.qbar_n = np.array((self.f + self.xibar_n)/self.bathy_np)
        self.qbar_dx_n = self.dx_YGXG_pad0(var=self.qbar_n)
        self.qbar_dy_n = self.dy_YGXG_pad0(var=self.qbar_n)
        self.mod_grad_qbar_n = np.sqrt(self.qbar_dx_n**2 + self.qbar_dy_n**2)
        self.mod_grad_qbar_n = np.where(self.mod_grad_qbar_n < 1.E-18, np.ones_like(self.mod_grad_qbar_n)*1.E-18, self.mod_grad_qbar_n)

        self.psi_dx_n = self.dx_YGXG_pad0(var=self.psibar_n)
        self.psi_dy_n = self.dy_YGXG_pad0(var=self.psibar_n)

        self.alpha_n = (-2*self.gamma_q*np.sqrt(self.Q_n*self.K_n))    

        # dQdt and dKdt
        self.Q_F_n = -self.alpha_n*self.mod_grad_qbar_n  - self.QAdv_n/self.bathy_np + self.QDiff_n
        self.K_F_n = (self.alpha_n/self.mod_grad_qbar_n)*(self.qbar_dx_n*self.psi_dx_n + self.qbar_dy_n*self.psi_dy_n)  - self.KAdv_n/self.bathy_np + self.KDiff_n
        self.K_F_n = np.pad(self.K_F_n[1:-1,1:-1],((1,1)),constant_values=0)
        
        # step forward Q and K
        self.Q_np1 = scheme(var_n=self.Q_n,dt=self.dt,F_n=self.Q_F_n,F_nm1=self.Q_F_nm1,F_nm2=self.Q_F_nm2)
        self.K_np1 = scheme(var_n=self.K_n,dt=self.dt,F_n=self.K_F_n,F_nm1=self.K_F_nm1,F_nm2=self.K_F_nm2)
        # set K = 0 where K is negative
        self.K_np1 = np.where(self.K_np1 < 0,0,self.K_np1)

        # add to sum variables
        self.Q_sum = self.Q_sum + self.Q_np1
        self.K_sum = self.K_sum + self.K_np1

    def no_K_Q(self,scheme):
        self.K_np1 = np.zeros_like(self.K_n) 
        self.Q_np1 = np.zeros_like(self.Q_n) 

    def K_Q_constant(self,scheme):
        self.K_np1 = self.K_n 
        self.Q_np1 = self.Q_n

        self.qbar_n = np.array((self.f + self.xibar_n)/self.bathy_np)
        self.qbar_dx_n = self.dx_YGXG_pad0(var=self.qbar_n)
        self.qbar_dy_n = self.dy_YGXG_pad0(var=self.qbar_n)
        self.mod_grad_qbar_n = np.sqrt(self.qbar_dx_n**2 + self.qbar_dy_n**2)
        self.mod_grad_qbar_n = np.where(self.mod_grad_qbar_n < 1.E-18, np.ones_like(self.mod_grad_qbar_n)*1.E-18, self.mod_grad_qbar_n)

        self.alpha_n = (-2*self.gamma_q*np.sqrt(self.Q_n*self.K_n)) 

    def eddyFluxes_scheme(self):
        self.flux_u_n = np.array((self.alpha_n*self.bathy_np*self.qbar_dx_n)/self.mod_grad_qbar_n)
        self.flux_v_n = np.array((self.alpha_n*self.bathy_np*self.qbar_dy_n)/self.mod_grad_qbar_n)

        self.eddyFluxes_n = (1/(2*self.d))*(self.flux_v_n[2:,1:-1] - self.flux_v_n[:-2,1:-1] + self.flux_u_n[1:-1,2:] - self.flux_u_n[1:-1,:-2])
        self.eddyFluxes_n = np.pad(self.eddyFluxes_n,((1,1)),constant_values=0)

    def noEddyFluxes(self):
        self.eddyFluxes_n = np.zeros_like(self.xibar_n)




# %%
'''H = 5000
h = 500
dx = 25000
dy = 25000
Nx = 40
Ny = 40
Lx = dx*Nx 
Ly = dy*Ny

bathy_mount = [[(H-h*np.sin((np.pi*(i*dx))/Lx)*np.sin((np.pi*(j*dy))/Ly)) for i in range(Nx+1)] for j in range(Ny+1)]
#bathy_flat = H*np.ones((Ny+1,Nx+1))

test_diags = Barotropic(d=dx,Nx=Nx,Ny=Ny,bathy=bathy_mount,f0=0.7E-4,beta=0)

test_diags.gen_init_psi(k_peak=2,const=1E11)
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

print(test_diags.xibar_0)

#%%
mu_Q = 1.E-11
sigma_Q = 1.E-12
mu_K = 1.E-4
sigma_K = 0.5E-3
init_Q = np.pad(np.random.normal(loc=mu_Q,scale=sigma_Q,size=(Ny+1,Nx+1))[1:-1,1:-1],((1,1)),constant_values=0)
init_K = np.random.normal(loc=mu_K,scale=sigma_K,size=(Ny+1,Nx+1))
init_K = np.where(init_K < 0,np.zeros_like(init_K),init_K)


#%%
plt.contourf(test_diags.XG,test_diags.YG,init_Q)
plt.colorbar()
plt.show()

plt.contourf(test_diags.XG,test_diags.YG,init_K)
plt.colorbar()
plt.show()

#%%
start_time = time.time_ns()
diagnostics = ['xi_u','xi_v','u_u','v_v','xi_xi']
data = test_diags.model(dt=900,Nt=10,gamma_q=0.0001,r_BD=0,kappa_xi=1,tau_0=0,rho_0=1000,dumpFreq=900,meanDumpFreq=9000,\
    kappa_q=1.E6,kappa_K=1.E3,diags=diagnostics,scheme='EGECAD',init_K=init_K,init_Q=init_Q)
end_time = time.time_ns()
print('finished')
print((end_time - start_time)/1.E9)


#%%
data.to_netcdf('./model_data/tests/schemeTest_f_PEG-KEC-Adv-Diff_gammaq-00001_kappaq-1000_kappaK-10')

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
    print(np.array_equal(xi_v,data.xi_v[t]))


# %%
for t in range(len(data.T)):
    plt.contourf(data.XG,data.YG,test_diags.K[t])
    plt.colorbar()
    plt.show()




# %%

# %%

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as anim
from matplotlib.animation import FuncAnimation
from IPython import display
import matplotlib.colors as colors
import matplotlib.cm as cm
plt.rcParams['animation.ffmpeg_path'] = '/Users/eavesr/opt/anaconda3/envs/mitgcm_env/bin/ffmpeg'

# %%
def plot_animate(t,X,Y,fig,axs,psi,xi,psi_levels,xi_levels,f,h):
    PV = f/h
    axs[0].cla()
    im_psi = axs[0].contour(X,Y,psi[t],cmap='RdBu',levels=psi_levels,extend='both')
    axs[0].set_xlabel('x (km)')
    axs[0].set_ylabel('y (km)')
    axs[0].title.set_text('psi')
    axs[0].set_aspect(1)
    axs[0].contour(X,Y,PV,colors='grey',linewidths=0.5)
    
    axs[1].cla()
    im_xi = axs[1].contourf(X,Y,xi[t],cmap='RdBu',levels=xi_levels,extend='both')
    axs[1].set_xlabel('x (km)')
    axs[1].set_ylabel('y (km)')
    axs[1].title.set_text('xi')
    axs[1].set_aspect(1)
    axs[1].contour(X,Y,PV,colors='grey',linewidths=0.5)
    
    fig.suptitle(str(t) + ' Days')
# %%

fig,axs = plt.subplots(1,2)
fig.set_size_inches(10,4)
X,Y = np.meshgrid(test_diags.XG/1000,test_diags.YG/1000)


ts = np.arange(100).astype(int)

levels_xi = np.linspace(np.nanquantile(data.xi,0.01),\
                        np.nanquantile(data.xi,0.99),20)
levels_psi = np.linspace(np.nanmin(data.psi),\
                         np.nanmax(data.psi),10)

cbar_psi = fig.colorbar(cm.ScalarMappable(norm=colors.TwoSlopeNorm(vmin=levels_psi[0],vmax=levels_psi[-1],vcenter=0),\
                                          cmap='RdBu'),ax=axs[0])
cbar_psi.set_label('$\psi$')
cbar_xi = fig.colorbar(cm.ScalarMappable(norm=colors.TwoSlopeNorm(vmin=levels_xi[0],vmax=levels_xi[-1],vcenter=0),\
                                         cmap='RdBu'),ax=axs[1])
cbar_xi.set_label('$\\xi$')


plt.tight_layout()

animation = FuncAnimation(fig, func=plot_animate, frames=ts, interval=100,fargs=[X,Y,fig,axs,data.psi,\
                                                                                 data.xi,levels_psi,\
                                                                                 levels_xi,test_diags.f,\
                                                                                test_diags.bathy],repeat=False)

video = animation.to_html5_video()
html = display.HTML(video)
display.display(html)
plt.close()
# %%

html_data = html.data
with open('./animations/tests/5km_visc_test_lowK_1.html', 'w') as f:
    f.write(html_data)

# %%

plt.contourf(data.XG,data.YG,test_diags.K[0])
plt.colorbar()
plt.show()

plt.contourf(data.XG,data.YG,test_diags.K[-1])
plt.colorbar()
plt.show()

# %%
print(np.nanmax(test_diags.K[99]))

# %%
print(np.array_equal(data.K[1],data.K[-1]))
# %%
print(np.max(test_diags.diff_n))
print(np.max(test_diags.diff_n_old))

# %%
print(test_diags.xibar_n)'''

# %%
