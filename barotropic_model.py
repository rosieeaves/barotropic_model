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
            self.XGm1 = np.array([i*self.dx for i in range(self.Nx+1)])[1:-1]
            self.YGm1 = np.array([j*self.dy for j in range(self.Ny+1)])[1:-1]
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

            self.bathy_YCXC = (self.bathy_np[:-1,:-1] + self.bathy_np[1:,:-1] + \
                
                self.bathy_np[:-1,1:] + self.bathy_np[1:,1:])/4
            
            self.bathy_YCXC = np.array(self.bathy_YCXC)

            self.bathy_YGXC = (self.bathy_np[:,1:] + self.bathy_np[:,:-1])/2
            self.bathy_YCXG = (self.bathy_np[1:,:] + self.bathy_np[:-1,:])/2

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
        self.start = 'FE'
        

    def init_psi(self,psi):
        try:
            np.shape(psi) == ((self.NyG,self.NxG))
        except ValueError:
            print('Initial psi must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
        else:
            self.start = 'FE'
            # set zero on boundaries
            psi_remove = psi[1:-1,1:-1]
            psi_pad = np.pad(psi_remove,((1,1)),constant_values=0)
            self.psibar_0 = psi_pad
            self.calc_UV(psi=self.psibar_0,u_return='ubar_0',v_return='vbar_0')
            self.xi_from_psi()

    def init_from_previous(self,xi,F_nm1,F_nm2):
        try:
            np.shape(xi) == ((self.NyG,self.NxG))
        except ValueError: 
            print('Initial xi must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
        else:
            try:
                np.shape(F_nm1) == ((self.NyG,self.NxG))
            except ValueError: 
                print('F_nm1 must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
            else: 
                try:
                    np.shape(F_nm2) == ((self.NyG,self.NxG))
                except ValueError: 
                    print('F_nm2 must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
                else:
                    self.start = 'AB3'
                    self.xibar_0 = xi
                    self.psi_from_xi()


    def init_xi(self,xi):
        try:
            np.shape(xi) == ((self.NyG,self.NxG))
        except ValueError:
            print('Initial xi must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
        else:
            self.start = 'FE'
            # set zero on boundaries
            xi_remove = xi[1:-1,1:-1]
            xi_pad = np.pad(xi_remove,((1,1)),constant_values=0)
            self.xibar_0 = xi_pad

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
            self.calc_UV(psi=self.psibar_0,u_return='ubar_0',v_return='vbar_0')


    def xi_from_psi(self):
        try:
            self.psibar_0
        except NameError:
            print('Specify initial psi or run init_psi() to calculate inttial xi from initial psi.')
        else:
            #calculate initial xibar from initial psibar

            xibar_0 = (self.psibar_0[1:-1,2:] + self.psibar_0[1:-1,:-2] + \
                self.psibar_0[2:,1:-1] + self.psibar_0[:-2,1:-1] - 4*self.psibar_0[1:-1,1:-1])/(self.bathy_np[1:-1,1:-1]*(self.d**2)) - \
                    ((self.psibar_0[1:-1,2:] - self.psibar_0[1:-1,:-2])*(self.bathy_np[1:-1,2:] - self.bathy_np[1:-1,:-2]) + \
                        (self.psibar_0[2:,1:-1] - self.psibar_0[:-2,1:-1])*\
                            (self.bathy_np[2:,1:-1] - self.bathy_np[:-2,1:-1]))/((self.bathy_np[1:-1,1:-1]**2)*(self.d**2))
 
            xibar_0 = np.pad(xibar_0,((1,1)),constant_values=0)

            self.xibar_0 = xibar_0

            self.calc_UV(psi=self.psibar_0,u_return='ubar_0',v_return='vbar_0')

    ##########################################################################
    # TIME STEPPING FUNCTIONS
    ##########################################################################


    def model(self,dt,Nt,dumpFreq,meanDumpFreq,\
        r_BD,mu_xi,\
            tau_0,rho_0,\
                eddy_scheme=False,init_K=None,init_Q=None,gamma_q=None,mu_PAR=None,r_Q=None,\
                    r_K=None,min_val=None,max_val=None,kappa_q=None,K_min=None,Q_min=None,backscatter_frac=None,diags=[],dissipationTerm='biharmonic'):

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

            # check corect parameters are set if using parameterization

            if eddy_scheme != False and eddy_scheme != 'constant' and (init_K.any() == None or init_Q.any() == None):
                raise ValueError('init_K and init_Q parameter must be specified when using scheme.')

            if eddy_scheme != False and eddy_scheme != 'constant' and (min_val == None):
                raise ValueError('min_val parameter must be specified when using scheme.')

            if eddy_scheme != False and eddy_scheme != 'constant' and (max_val == None):
                raise ValueError('max_val parameter must be specified when using scheme.')

            if eddy_scheme == 'EGEC' or eddy_scheme == 'EGEC_TEST':
                if r_Q == None or r_K == None or Q_min == None or K_min == None:
                    raise ValueError('r_Q, R_K, Q_min and K_min parameters should be set when using scheme.')

            if eddy_scheme == 'EGECAD' or eddy_scheme == 'EGECAD_TEST':
                if mu_PAR == None or r_Q == None or r_K == None or K_min == None or Q_min == None:
                    raise ValueError('mu_PAR, r_Q, r_K, Q_min and K_min parameters must be set when using scheme with advection and diffusion.')
                if np.shape(init_K) != (self.NyC,self.NxC):
                    raise ValueError('init_K should have shape (Ny,Nx) to be defined on the cell centre points.')
                if np.shape(init_Q) != (self.NyC,self.NxC):
                    raise ValueError('init_Q should have shape (Ny,Nx) to be defined on the cell centre points.')

            if eddy_scheme == 'constant' and kappa_q == None:
                raise ValueError('kappa_q must be specified when using constant scheme version.')
            
            if eddy_scheme == 'EGECAD_backscatter' and backscatter_frac == None:
                raise ValueError('backscatter_frac must be specified when using backscatter scheme version.')
            
            if dissipationTerm != 'biharmonic' and dissipationTerm != 'laplacian' and dissipationTerm != 'laplacian_enstrophy' and dissipationTerm != 'laplacian_energy':
                raise ValueError('dissipationTerm can be biharmonic, laplacian, laplacian_enstrophy or laplacian_energy')

            # store model parameters in object
            self.dt = dt
            self.Nt = Nt
            self.gamma_q = gamma_q
            self.r_BD = r_BD
            self.mu_xi = mu_xi
            self.tau_0 = tau_0 
            self.rho_0 = rho_0
            self.dumpFreq = dumpFreq
            self.meanDumpFreq = meanDumpFreq
            self.T = np.arange(0,int(self.Nt*self.dt)+1,self.dumpFreq)
            self.mu_PAR = mu_PAR
            self.r_Q = r_Q
            self.r_K = r_K
            self.Q_min = Q_min 
            self.K_min = K_min
            self.eddy_scheme = eddy_scheme
            self.min_val = min_val
            self.max_val = max_val
            self.kappa_q = kappa_q
            self.backscatter_frac = backscatter_frac
            self.dissipationTerm = dissipationTerm

            # if initial u and v do not already exist, calculate them
            try:
                self.ubar_0
            except:
                self.calc_UV(psi=self.psibar_0,u_return='ubar_0',v_return='vbar_0')

            # initialise parameters pisbar, zetabar, and grad xibar components
            self.psibar_n = self.psibar_0
            self.psibar_np1 = np.zeros_like(self.psibar_n)
            self.xibar_n = self.xibar_0
            self.xibar_np1 = np.zeros_like(self.xibar_n)
            self.ubar_n = self.ubar_0
            self.ubar_np1 = np.zeros_like(self.ubar_n)
            self.vbar_n = self.vbar_0
            self.vbar_np1 = np.zeros_like(self.vbar_n)
            self.index = 0
            self.index_MEAN = 0
            self.zeta_n = np.zeros((self.NyG,self.NxG))
            self.qbar_n = np.zeros((self.NyG,self.NxG))

            # calculate length of T and T_MEAN arrays
            len_T = int((self.Nt*self.dt)/self.dumpFreq)
            len_T_MEAN = int((self.Nt*self.dt)/self.meanDumpFreq)

            # create arrays to save the data
            # NOTE: len_T+1 so that you can add the value at time = 0 in
            self.xibar = np.zeros((len_T+1,self.NyG,self.NxG))
            self.psibar = np.zeros((len_T+1,self.NyG,self.NxG))
            self.ubar = np.zeros((len_T+1,self.NyC,self.NxG))
            self.vbar = np.zeros((len_T+1,self.NyG,self.NxC))
            self.xibar[0] = self.xibar_0
            self.psibar[0] = self.psibar_0
            self.ubar[0] = self.ubar_0
            self.vbar[0] = self.vbar_0

            # create mean parameters
            self.xibar_sum = self.xibar_0
            self.psibar_sum = self.psibar_0
            self.ubar_sum = self.ubar_0
            self.vbar_sum = self.vbar_0
            self.xi_MEAN = np.zeros((len_T_MEAN,self.NyG,self.NxG))
            self.psi_MEAN = np.zeros((len_T_MEAN,self.NyG,self.NxG))
            self.u_MEAN = np.zeros((len_T_MEAN,self.NyC,self.NxG))
            self.v_MEAN = np.zeros((len_T_MEAN,self.NyG,self.NxC))

            # parameters for energy checking
            self.q_biharm = np.zeros((len_T+1,self.NyG,self.NxG))
            self.psi_biharm = np.zeros((len_T+1,self.NyG,self.NxG))

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
                setattr(self,var,np.zeros((len_T+1,num_Y,num_X)))
                # create variable for calculating np1 time step, e.g. self.xi_u_np1. Initialise with zeros. 
                # Should be updated every timestep.
                setattr(self,self.diagnosticsNP1Dict[var],np.zeros((num_Y,num_X)))
                # calculate value at time zero, set it to var_np1
                self.diagnosticsFunctionsDict[var](xi=self.xibar_0,psi=self.psibar_0,u=self.ubar_0,v=self.vbar_0,var_return=self.diagnosticsNP1Dict[var])
                # put value at time zero in 0th index of saved data array
                getattr(self,var)[0] = getattr(self,self.diagnosticsNP1Dict[var]).copy()
                # reset var_np1 to zero
                setattr(self,self.diagnosticsNP1Dict[var],np.zeros((num_Y,num_X)))
                # create array to save MEAN data in, e.g. self.xi_u_MEAN. Initialise with zeros. 
                # Should be added to every meanDumpFreq seconds.
                setattr(self,self.diagnosticsMeanDict[var],np.zeros((len_T_MEAN,num_Y,num_X)))
                # create variable for summing snapshot data for use in calculating mean, e.g. self.xi_u_sum. 
                # Should be updated every timestep. 
                setattr(self,self.diagnosticsSumDict[var],getattr(self,var)[0])

            # initialise dissipation terms
            self.dissipationDict()
            if self.dissipationTerm == 'laplacian_enstrophy':
                self.dxidx = np.zeros((self.NyG,self.NxG))
                self.dxidy = np.zeros((self.NyG,self.NxG))
            elif self.dissipationTerm == 'laplacian_energy':
                self.dHxidx = np.zeros((self.NyG,self.NxG))
                self.dHxidy = np.zeros((self.NyG,self.NxG))
            self.dissipation_n = np.zeros((self.NyG,self.NxG))
            
            # initialise parameters for time stepping
            self.F_n = np.zeros_like(self.xibar_n)
            self.F_nm1 = np.zeros_like(self.xibar_n)
            self.F_nm2 = np.zeros_like(self.xibar_n)
            self.adv_n = np.zeros_like(self.xibar_n)
            self.BD_n = np.zeros_like(self.xibar_n)
            self.zeta_n = np.zeros_like(self.xibar_n)
            self.psiYCXC_n = np.zeros((self.Ny,self.Nx))

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

            self.energyCheck = np.zeros((len_T+1,self.NyG,self.NxG))

            if self.eddy_scheme != False:
                if self.eddy_scheme == 'constant':
                    self.dqdx_n = np.zeros((self.NyG,self.NxG))
                    self.dqdy_n = np.zeros((self.NyG,self.NxG))

                    self.flux_u_n = np.zeros((self.NyG,self.NxG))
                    self.flux_v_n = np.zeros((self.NyG,self.NxG))
                    self.eddyFluxes_n = np.zeros((self.NyG,self.NxG))

                    self.Q_0 = np.zeros((self.NyC,self.NxC))
                    self.Q_n = np.zeros((self.NyC,self.NxC))
                    self.Q_np1 = np.zeros((self.NyC,self.NxC))

                    self.K_0 = np.zeros((self.NyC,self.NxC))
                    self.K_n = np.zeros((self.NyC,self.NxC))
                    self.K_np1 = np.zeros((self.NyC,self.NxC))

                    self.Q_F_n = np.zeros((self.NyC,self.NxC))
                    self.Q_F_nm1 = np.zeros((self.NyC,self.NxC))
                    self.Q_F_nm2 = np.zeros((self.NyC,self.NxC))

                    self.K_F_n = np.zeros((self.NyC,self.NxC))
                    self.K_F_nm1 = np.zeros((self.NyC,self.NxC))
                    self.K_F_nm2 = np.zeros((self.NyC,self.NxC))
                    
                else:
                    # enstrophy variables
                    self.Q_0 = init_Q
                    self.Q_n = self.Q_0
                    self.Q_np1 = np.zeros((self.NyC,self.NxC))
                    self.Q = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.Q[0] = self.Q_0
                    self.Q_F_n = np.zeros((self.NyC,self.NxC))
                    self.Q_F_nm1 = np.zeros((self.NyC,self.NxC))
                    self.Q_F_nm2 = np.zeros((self.NyC,self.NxC))
                    self.Q_sum = self.Q_0
                    self.Q_MEAN = np.zeros((len_T_MEAN,self.NyC,self.NxC))
                    self.QAdv_n = np.zeros((self.NyC,self.NxC))
                    self.QDiff_L_n = np.zeros((self.NyC,self.NxC))
                    self.Q_n_YGXG = np.zeros((self.NyG,self.NxG))
                    self.Q_n_ghost = np.zeros((self.NyC+2,self.NxC+2))
                    self.Q_n_YGXC = np.zeros((self.NyG,self.NxC))
                    self.Q_n_YCXG = np.zeros((self.NyC,self.NxG))


                    # energy variables
                    self.K_0 = init_K
                    self.K_n = self.K_0
                    self.K_np1 = np.zeros((self.NyC,self.NxC))
                    self.K = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.K[0] = self.K_0
                    self.K_F_n = np.zeros((self.NyC,self.NxC))
                    self.K_F_nm1 = np.zeros((self.NyC,self.NxC))
                    self.K_F_nm2 = np.zeros((self.NyC,self.NxC))
                    self.K_sum = self.K_0
                    self.K_MEAN = np.zeros((len_T_MEAN,self.NyC,self.NxC))
                    self.KAdv_n = np.zeros((self.NyC,self.NxC))
                    self.KDiff_L_n = np.zeros((self.NyC,self.NxC))
                    self.K_n_YGXG = np.zeros((self.NyG,self.NxG))
                    self.K_n_ghost = np.zeros((self.NyC+2,self.NxC+2))
                    self.K_n_YGXC = np.zeros((self.NyG,self.NxC))
                    self.K_n_YCXG = np.zeros((self.NyC,self.NxG))

                    # variables for parameterization calculation 
                    self.dqdx_n_YCXC = np.zeros((self.NyC,self.NxC))
                    self.dqdx_n_YGXC = np.zeros((self.NyG,self.NxC))
                    self.dqdx_n_YCXG = np.zeros((self.NyC,self.NxG))

                    self.dqdy_n_YCXC = np.zeros((self.NyC,self.NxC))
                    self.dqdy_n_YGXC = np.zeros((self.NyG,self.NxC))
                    self.dqdy_n_YCXG = np.zeros((self.NyC,self.NxG))

                    self.mod_grad_qbar_n_YGXC = np.zeros((self.NyG,self.NxC))
                    self.mod_grad_qbar_n_YCXG = np.zeros((self.NyC,self.NxG))

                    self.kappa_n = np.zeros((self.NyC,self.NxC))
                    self.kappa_n_YGXC = np.zeros((self.NyG,self.NxC))
                    self.kappa_n_YCXG = np.zeros((self.NyC,self.NxG))
                    self.kappa_sum = np.zeros((self.NyC,self.NxC))
                    self.kappa_MEAN = np.zeros((len_T_MEAN,self.NyC,self.NxC))

                    self.dpsidx_n_YGXC = np.zeros((self.NyG,self.NxC))
                    self.dpsidy_n_YCXG = np.zeros((self.NyC,self.NxG))

                    self.mod_grad_qbar_n_YCXC = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.kappa = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.Q_min_array = self.Q_min*np.ones((self.NyC,self.NxC))
                    self.K_min_array = self.K_min*np.ones((self.NyC,self.NxC))

                    self.qu_EDDY_n_YGXC = np.zeros((self.NyG,self.NxC))
                    self.qv_EDDY_n_YCXG = np.zeros((self.NyC,self.NxG))
                    self.eddyFluxes = np.zeros((len_T+1,self.NyG,self.NxG))
                    self.eddyFluxes_n = np.zeros((self.NyG,self.NxG))

                    # scheme diagnostics 
                    self.enstrophyGen_n = np.zeros((self.NyC,self.NxC))
                    self.enstrophyGen_n_x_YGXC = np.zeros((self.NyG,self.NxC))
                    self.enstrophyGen_n_y_YCXG = np.zeros((self.NyC,self.NxG))
                    self.enstrophyGen = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.enstrophyGen_sum = np.zeros((self.NyC,self.NxC))
                    self.enstrophyGen_MEAN = np.zeros((len_T_MEAN,self.NyC,self.NxC))

                    self.energyConv_n = np.zeros((self.NyC,self.NxC))
                    self.energyConv_n_x_YGXC = np.zeros((self.NyG,self.NxC))
                    self.energyConv_n_y_YCXG = np.zeros((self.NyC,self.NxG))
                    self.energyConv = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.energyConv_sum = np.zeros((self.NyC,self.NxC))
                    self.energyConv_MEAN = np.zeros((len_T_MEAN,self.NyC,self.NxC))

                    # backscatter variables
                    self.volume_YCXC = np.sum(self.bathy_YCXC*self.dx*self.dy)
                    self.KE_backscatter = np.zeros(len_T+1)
                    self.KE_backscatter_n = 0
                    self.KE_backscatter_sum = 0
                    self.KE_backscatter_MEAN = np.zeros(len_T_MEAN)

                    # parameters for checking consistencies
                    self.Q_diffusion = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.K_diffusion = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.Q_damping = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.K_damping = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.Q_advection = np.zeros((len_T+1,self.NyC,self.NxC))
                    self.K_advection = np.zeros((len_T+1,self.NyC,self.NxC))

                
            else:
                self.Q_0 = np.zeros((self.NyC,self.NxC))
                self.Q_n = np.zeros((self.NyC,self.NxC))
                self.Q_np1 = np.zeros((self.NyC,self.NxC))
                self.K_0 = np.zeros((self.NyC,self.NxC))
                self.K_n = np.zeros((self.NyC,self.NxC))
                self.K_np1 = np.zeros((self.NyC,self.NxC))

            # run scheemeDict function to get schemeFunctionsDict which contains functions to run if scheme is True or False
            self.schemeDict()
            


            # time stepping function

            #def time_step(scheme,F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1):
            def time_step(scheme,**kw):

                def wrapper(*args,**kwargs):
                    self.zeta_n = np.array((self.f + self.xibar_n))
                    self.qbar_n = np.array((self.f + self.xibar_n)/self.bathy_np)

                    # CALCULATE VORTICITY AT NEXT TIME STEP
                    # calculate advection term
                    self.advection_xi()
                    # dissiaption from bottom drag
                    self.BD_n = self.r_BD*self.xibar_n
                    # diffusion of vorticity
                    #self.laplacian(var=self.xibar_n,mu=self.mu_xi_L,var_return='diffusion_L_n')
                    #self.biharmonic(var=self.xibar_n,mu=self.mu_xi_B,var_return='diffusion_B_n')

                    self.dissipationFunctionDict['dissipationFunc'](var=self.xibar_n,mu=self.mu_xi,var_return='dissipation_n')

                    '''self.xi_pad = np.pad(self.xibar_n,((1,1)),mode='edge')
                    self.diff_xi = self.diffusion(var=self.xi_pad)[1:-1,1:-1]
                    self.diff_xi_pad = np.pad(self.diff_xi,((1,1)),mode='edge')
                    self.diffusion_B_n = self.mu_xi_B*self.bathy_np*self.diffusion(var=self.diff_xi_pad)[1:-1,1:-1]'''

                    # eddy fluxes
                    self.schemeFunctionsDict['calc_K_Q']()
                    self.schemeFunctionsDict['KQ_timestep'](scheme=scheme)
                    self.schemeFunctionsDict['calc_eddyFluxes']()
                    # F_n
                    self.F_n = self.adv_n - self.eddyFluxes_n - self.BD_n + self.dissipation_n + np.array(self.wind_stress)
                    # calculate new value of xibar
                    self.xibar_np1 = scheme(var_n=self.xibar_n,dt=self.dt,F_n=self.F_n,F_nm1=self.F_nm1,F_nm2=self.F_nm2)
                    # set xi to zero on boundaries
                    #self.xibar_np1 = np.pad(self.xibar_np1[1:-1,1:-1],((1,1)),constant_values=0)

                    # uncomment for solid body rotation
                    # self.psibar_np1 = self.psibar_n
                    # self.calc_UV(self.psibar_np1)

                    # SOLVE FOR PSI AT NEXT TIME STEP
                    # comment out for solid body rotation
                    self.psibar_np1 = self.LU.solve((-np.array(self.xibar_np1)).flatten())
                    self.psibar_np1 = self.psibar_np1.reshape((self.NyG,self.NxG))
                    self.psibar_np1 = np.pad(self.psibar_np1[1:-1,1:-1],((1,1)),constant_values=0)
                    self.calc_UV(psi=self.psibar_np1,u_return='ubar_np1',v_return='vbar_np1')

                    # ADD TO SUM FOR MEAN DATA
                    self.xibar_sum += self.xibar_np1
                    self.psibar_sum += self.psibar_np1
                    self.ubar_sum += self.ubar_np1
                    self.vbar_sum += self.vbar_np1

                    # DIAGNOSTICS
                    for var in diags:
                        # Run function to set np1 variable, e.g. self.xi_u_np1, to new value
                        self.diagnosticsFunctionsDict[var](xi=self.xibar_np1,psi=self.psibar_np1,u=self.ubar_np1,\
                                                           v=self.vbar_np1,var_return=self.diagnosticsNP1Dict[var])

                        # set sum variable, e.g. self.xi_u_sum, to sum + np1 e.g. self.xi_u_sum + self.xi_u_np1
                        setattr(self,self.diagnosticsSumDict[var],getattr(self,self.diagnosticsSumDict[var])+getattr(self,self.diagnosticsNP1Dict[var]))
                        #self.__dict__[self.diagnosticsSumDict[var]] += self.__dict__[self.diagnosticsNP1Dict[var]]

                    

                    # DUMP DATA EVERY dumpFreq SECONDS
                    if kw.get('t')%kw.get('dumpFreq') == 0:
                        self.index = int(kw.get('t')/kw.get('dumpFreq'))
                        # save xi, psi, u and v
                        self.xibar[self.index] = self.xibar_np1
                        self.psibar[self.index] = self.psibar_np1
                        self.ubar[self.index] = self.ubar_np1
                        self.vbar[self.index] = self.vbar_np1
                        self.q_biharm[self.index] = self.qbar_n*self.dissipation_n 
                        self.psi_biharm[self.index] = self.psibar_n*self.dissipation_n

                        # run diagnostics 
                        for var in diags:
                            # set snapshot data variable, e.g. self.xi_u, to append(old snapshot data, np1 value), e.g. append self.xi_u with self.xi_u_np1
                            self.__dict__[var][self.index] = self.__dict__[self.diagnosticsNP1Dict[var]]
                            #getattr(self,var)[self.index] = getattr(self,self.diagnosticsNP1Dict[var])

                        if self.eddy_scheme != False and eddy_scheme != 'constant':

                            self.Q[self.index] = self.Q_np1
                            self.K[self.index] = self.K_np1
                            self.kappa[self.index] = self.kappa_n
                            self.enstrophyGen[self.index] = self.enstrophyGen_n 
                            self.energyConv[self.index] = self.energyConv_n
                            self.eddyFluxes[self.index] = self.eddyFluxes_n
                            self.KE_backscatter[self.index] = self.KE_backscatter_n

                            # variables for enstrophy issue 
                            # delete once problem solved
                            self.Q_diffusion[self.index] = self.QDiff_L_n
                            self.K_diffusion[self.index] = self.KDiff_L_n 
                            self.Q_advection[self.index] = self.QAdv_n
                            self.K_advection[self.index] = self.KAdv_n
                            self.Q_damping[self.index] = (self.Q_n - self.Q_min_array)*self.r_Q
                            self.K_damping[self.index] = (self.K_n - self.K_min_array)*self.r_K

                    # DUMP MEAN DATA
                    if kw.get('t')%kw.get('meanDumpFreq') == 0:
                        self.index_MEAN = int(kw.get('t')/kw.get('meanDumpFreq')) - 1
                        # dump mean data
                        self.xi_MEAN[self.index_MEAN] = (self.xibar_sum*self.dt)/self.meanDumpFreq
                        self.psi_MEAN[self.index_MEAN] = (self.psibar_sum*self.dt)/self.meanDumpFreq
                        self.u_MEAN[self.index_MEAN] = (self.ubar_sum*self.dt)/self.meanDumpFreq
                        self.v_MEAN[self.index_MEAN] = (self.vbar_sum*self.dt)/self.meanDumpFreq


                        # reset sum for mean to zero
                        self.xibar_sum = np.zeros_like(self.xibar_sum)
                        self.psibar_sum = np.zeros_like(self.psibar_sum)
                        self.ubar_sum = np.zeros_like(self.ubar_sum)
                        self.vbar_sum = np.zeros_like(self.vbar_sum)

                        if self.eddy_scheme != False and eddy_scheme != 'constant':
                            self.Q_MEAN[self.index_MEAN] = (self.Q_sum*self.dt)/self.meanDumpFreq
                            self.K_MEAN[self.index_MEAN] = (self.K_sum*self.dt)/self.meanDumpFreq
                            self.enstrophyGen_MEAN[self.index_MEAN] = (self.enstrophyGen_sum*self.dt)/self.meanDumpFreq
                            self.energyConv_MEAN[self.index_MEAN] = (self.energyConv_sum*self.dt)/self.meanDumpFreq
                            self.kappa_MEAN[self.index_MEAN] = (self.kappa_sum*self.dt)/self.meanDumpFreq
                            self.KE_backscatter_MEAN[self.index_MEAN] = (self.KE_backscatter_sum*self.dt)/self.meanDumpFreq

                            self.Q_sum = np.zeros_like(self.Q_sum)
                            self.K_sum = np.zeros_like(self.K_sum)
                            self.enstrophyGen_sum = np.zeros_like(self.enstrophyGen_sum)
                            self.energyConv_sum = np.zeros_like(self.energyConv_sum)
                            self.kappa_sum = np.zeros_like(self.kappa_sum)
                            self.KE_backscatter_sum = 0



                        # run diagnostics 
                        for var in diags:
                            # set mean data, e.g. self.xi_u_MEAN, to append(old mean data, new mean data), e.g. append self.xi_u_MEAN with self.xi_u_sum/time
                            self.__dict__[self.diagnosticsMeanDict[var]][self.index_MEAN] = (self.__dict__[self.diagnosticsSumDict[var]]*self.dt)/self.meanDumpFreq
                            #getattr(self,self.diagnosticsMeanDict[var])[self.index_MEAN] = (getattr(self,self.diagnosticsSumDict[var])*self.dt)/self.meanDumpFreq
                            # set sum variable, e.g. self.xi_u_sum, to zero
                            self.__dict__[self.diagnosticsSumDict[var]] = np.zeros_like(self.__dict__[self.diagnosticsSumDict[var]])
                            #setattr(self,self.diagnosticsSumDict[var],np.zeros_like(getattr(self,self.diagnosticsSumDict[var])))

                        
       
                    # reset values 
                    self.F_nm2 = self.F_nm1.copy()
                    self.F_nm1 = self.F_n.copy()
                    self.F_n = np.zeros_like(self.F_n)
                    self.xibar_n = self.xibar_np1.copy()
                    self.psibar_n = self.psibar_np1.copy()
                    self.ubar_n = self.ubar_np1.copy()
                    self.vbar_n = self.vbar_np1.copy()

                    if self.eddy_scheme != False:

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
                return var_n + dt*F_n

            def AB3(var_n,dt,F_n,F_nm1,F_nm2):
                return var_n + (dt/12)*(23*F_n - 16*F_nm1 + 5*F_nm2)
            
            t_0 = 1
            # first two time steps with forward Euler
            if self.start == 'FE':
                func_to_run = time_step(scheme=forward_Euler,t=1,dumpFreq=dumpFreqTS,meanDumpFreq=meanDumpFreqTS)
                func_to_run()

                func_to_run = time_step(scheme=forward_Euler,t=2,dumpFreq=dumpFreqTS,meanDumpFreq=meanDumpFreqTS)
                func_to_run()

                t_0 = 3

            # all other time steps with AB3:
            for t in range(t_0,self.Nt+1):
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
                    YGm1 = self.YGm1,
                    XGm1 = self.XGm1,
                    T_MEAN = self.T_MEAN,
                ),
                attrs = dict(
                    dt = self.dt,
                    Nt = self.Nt,
                    r_BD = self.r_BD,
                    mu_xi = self.mu_xi,
                    tau_0 = self.tau_0,
                    rho_0 = self.rho_0,
                    dumpFreq = self.dumpFreq,
                    meanDumpFreq = self.meanDumpFreq,
                    dx = self.dx,
                    dy = self.dy,
                    Nx = self.Nx,
                    Ny = self.Ny,
                    f0 = self.f0,
                    beta = self.beta,
                    dissipationTerm = self.dissipationTerm
                ) 

            )
            dataset_return['eddy_scheme'] = self.eddy_scheme
            if self.eddy_scheme != False:
                dataset_return['mu_PAR'] = self.mu_PAR
                dataset_return['r_Q'] = self.r_Q
                dataset_return['r_K'] = self.r_K
                dataset_return['Q_min'] = self.Q_min
                dataset_return['K_min'] = self.K_min
                dataset_return['gamma_q'] = self.gamma_q
                dataset_return['kappa_q'] = self.kappa_q

            dataset_return['bathy'] = self.bathy
            dataset_return['f'] = self.f

            dataset_return['xi'] = xr.DataArray(
                self.xibar.astype('float64'),
                dims = ['T','YG','XG'],
                coords = {
                    'T': self.T,
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['xi_MEAN'] = xr.DataArray(
                self.xi_MEAN.astype('float64'),
                dims = ['T_MEAN','YG','XG'],
                coords = {
                    'T_MEAN': self.T_MEAN,
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['psi'] = xr.DataArray(
                self.psibar.astype('float64'),
                dims = ['T','YG','XG'],
                coords = {
                    'T': self.T,
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['psi_MEAN'] = xr.DataArray(
                self.psi_MEAN.astype('float64'),
                dims = ['T_MEAN','YG','XG'],
                coords = {
                    'T_MEAN': self.T_MEAN,
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['u'] = xr.DataArray(
                self.ubar.astype('float64'),
                dims = ['T','YC','XG'],
                coords = {
                    'T': self.T,
                    'YC': self.YC,
                    'XG': self.XG
                }
            )

            dataset_return['u_MEAN'] = xr.DataArray(
                self.u_MEAN.astype('float64'),
                dims = ['T_MEAN','YC','XG'],
                coords = {
                    'T_MEAN': self.T_MEAN,
                    'YC': self.YC,
                    'XG': self.XG
                }
            )

            dataset_return['v'] = xr.DataArray(
                self.vbar.astype('float64'),
                dims = ['T','YG','XC'],
                coords = {
                    'T': self.T,
                    'YG': self.YG,
                    'XC': self.XC
                }
            )

            dataset_return['v_MEAN'] = xr.DataArray(
                self.v_MEAN.astype('float64'),
                dims = ['T_MEAN','YG','XC'],
                coords = {
                    'T_MEAN': self.T_MEAN,
                    'YG': self.YG,
                    'XC': self.XC
                }
            )

            dataset_return['xi_F_nm1'] = xr.DataArray(
                self.F_nm1.astype('float64'),
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['xi_F_nm2'] = xr.DataArray(
                self.F_nm2.astype('float64'),
                dims = ['YG','XG'],
                coords = {
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['q_biharmonic'] = xr.DataArray(
                self.q_biharm.astype('float64'),
                dims = ['T','YG','XG'],
                coords = {
                    'T': self.T,
                    'YG': self.YG,
                    'XG': self.XG
                }
            )

            dataset_return['psi_biharmonic'] = xr.DataArray(
                self.psi_biharm.astype('float64'),
                dims = ['T','YG','XG'],
                coords = {
                    'T': self.T,
                    'YG': self.YG,
                    'XG': self.XG
                }
            )
            

            # add diagnostics data to dataset_return
            for var in diags:
                # snapshot data every dumpFreq seconds, e.g. self.xi_u
                dataset_return[var] = xr.DataArray(
                    getattr(self,var).astype('float64'),
                    dims = self.diagnosticsCoordDict[var],
                    coords = {
                        self.diagnosticsCoordDict[var][0]: getattr(self,self.diagnosticsCoordDict[var][0]),
                        self.diagnosticsCoordDict[var][1]: getattr(self,self.diagnosticsCoordDict[var][1]),
                        self.diagnosticsCoordDict[var][2]: getattr(self,self.diagnosticsCoordDict[var][2])
                    }
                )

                # mean data every meanDumpFreq seconds, e.g. self.xi_u_MEAN
                dataset_return[self.diagnosticsMeanDict[var]] = xr.DataArray(
                    getattr(self,self.diagnosticsMeanDict[var]).astype('float64'),
                    dims = ['T_MEAN',self.diagnosticsCoordDict[var][1],\
                        self.diagnosticsCoordDict[var][2]],
                    coords = {
                        'T_MEAN': self.T_MEAN,
                        self.diagnosticsCoordDict[var][1]: getattr(self,self.diagnosticsCoordDict[var][1]),
                        self.diagnosticsCoordDict[var][2]: getattr(self,self.diagnosticsCoordDict[var][2])
                    }
                )
            if self.eddy_scheme != False and eddy_scheme != 'constant':
         
                dataset_return['Q'] = xr.DataArray(
                    self.Q.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )
       
                dataset_return['Q_MEAN'] = xr.DataArray(
                    self.Q_MEAN.astype('float64'),
                    dims = ['T_MEAN','YC','XC'],
                    coords = {
                        'T_MEAN': self.T_MEAN,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )
 
                dataset_return['K'] = xr.DataArray(
                    self.K.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )
         
                dataset_return['K_MEAN'] = xr.DataArray(
                    self.K_MEAN.astype('float64'),
                    dims = ['T_MEAN','YC','XC'],
                    coords = {
                        'T_MEAN': self.T_MEAN,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )
     
                dataset_return['kappa'] = xr.DataArray(
                    self.kappa.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )

                dataset_return['kappa_MEAN'] = xr.DataArray(
                    self.kappa_MEAN.astype('float64'),
                    dims = ['T_MEAN','YC','XC'],
                    coords = {
                        'T_MEAN': self.T_MEAN,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )
     
                dataset_return['enstrophyGen'] = xr.DataArray(
                    self.enstrophyGen.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )
   
                dataset_return['enstrophyGen_MEAN'] = xr.DataArray(
                    self.enstrophyGen_MEAN.astype('float64'),
                    dims = ['T_MEAN','YC','XC'],
                    coords = {
                        'T_MEAN': self.T_MEAN,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )

                dataset_return['energyConv'] = xr.DataArray(
                    self.energyConv.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )
   
                dataset_return['energyConv_MEAN'] = xr.DataArray(
                    self.energyConv_MEAN.astype('float64'),
                    dims = ['T_MEAN','YC','XC'],
                    coords = {
                        'T_MEAN': self.T_MEAN,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )
        
                dataset_return['eddyFluxes'] = xr.DataArray(
                    self.eddyFluxes.astype('float64'),
                    dims = ['T','YG','XG'],
                    coords = {
                        'T': self.T,
                        'YG': self.YG,
                        'XG': self.XG
                    }
                )

                dataset_return['KE_backscatter'] = xr.DataArray(
                    self.KE_backscatter.astype('float64'),
                    dims=['T'],
                    coords = {
                        'T': self.T
                    }
                )

                dataset_return['KE_backscatter_MEAN'] = xr.DataArray(
                    self.KE_backscatter_MEAN.astype('float64'),
                    dims = ['T_MEAN'],
                    coords = {
                        'T_MEAN': self.T_MEAN
                    }
                )

                dataset_return['Q_damping'] = xr.DataArray(
                    self.Q_damping.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )

                dataset_return['K_damping'] = xr.DataArray(
                    self.K_damping.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )

                dataset_return['Q_diffusion'] = xr.DataArray(
                    self.Q_diffusion.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )

                dataset_return['K_diffusion'] = xr.DataArray(
                    self.K_diffusion.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )

                dataset_return['Q_advection'] = xr.DataArray(
                    self.Q_advection.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )

                dataset_return['K_advection'] = xr.DataArray(
                    self.K_advection.astype('float64'),
                    dims = ['T','YC','XC'],
                    coords = {
                        'T': self.T,
                        'YC': self.YC,
                        'XC': self.XC
                    }
                )

            return dataset_return

    ##########################################################################
    # DIFFUSION FUNCTIONS
    ##########################################################################   

    def laplacian(self,var,mu,var_return):
        setattr(self,var_return,self.diffusion(var=np.array(var))*mu)

    def laplacian_enstrophy(self,var,mu,var_return):
        # dissipation term is H div((1/H) * grad xi) with boundary condition grad xi = 0

        # calculate dxi/dx and dxi/dy using centred difference
        # set to zero on the boundary
        self.dxidx  = np.pad((self.xibar_n[1:-1,2:] - self.xibar_n[1:-1,:-2])/(2*self.dx),((1,1)),constant_values=0)
        self.dxidy = np.pad((self.xibar_n[2:,1:-1] - self.xibar_n[:-2,1:-1])/(2*self.dy),((1,1)),constant_values=0)

        setattr(self,var_return,mu*np.pad(self.bathy_np[1:-1,1:-1]*\
                                       (((self.dxidx[1:-1,2:]/self.bathy_np[1:-1,2:]) - (self.dxidx[1:-1,:-2]/self.bathy_np[1:-1,:-2]))/(2*self.dx) + \
                                        ((self.dxidy[2:,1:-1]/self.bathy_np[2:,1:-1]) - (self.dxidy[:-2,1:-1]/self.bathy_np[2:,1:-1]))/(self.dy)),\
                                            ((1,1)),constant_values=0))
    
    def laplacian_energy(self,var,mu,var_return):
        # dissipation term is div((1/H) * grad(H*xi)) with boundary conditions psi = 0 and xi = 0

        # calculate d H*xi /dx and d H*xi /dy calculating values at boundaries
        self.dHxidx = self.dx_YGXG(var=self.bathy_np*self.xibar_n)
        self.dHxidy = self.dy_YGXG(var=self.bathy_np*self.xibar_n)

        setattr(self,var_return,mu*np.pad((((self.dHxidx[1:-1,2:]/self.bathy_np[1:-1,2:]) - (self.dHxidx[1:-1,:-2]/self.bathy_np[1:-1,:-2]))/(2*self.dx) + \
                                           ((self.dHxidy[2:,1:-1]/self.bathy_np[2:,1:-1]) - (self.dHxidy[:-2,1:-1]/self.bathy_np[:-2,1:-1]))/(2*self.dy)),\
                                            ((1,1)),constant_values=0))

    def biharmonic(self,var,mu,var_return):
        setattr(self,var_return,-mu*self.diffusion(var=self.diffusion(var=np.array(var))))

    def diffusion(self,var): 

        return np.pad((1/self.d**2)*(var[1:-1,2:] + var[1:-1,:-2] + var[2:,1:-1] + var[:-2,1:-1] - 4*var[1:-1,1:-1]),((1,1)),constant_values=0)
    
    def dissipationDict(self):

        if self.dissipationTerm == 'biharmonic':
            print('Biharmonic diffusion')
            self.dissipationFunctionDict = {
                'dissipationFunc': self.biharmonic
            }
        elif self.dissipationTerm == 'laplacian':
            print('Laplacian diffusion')
            self.dissipationFunctionDict = {
                'dissipationFunc': self.laplacian
            }
        elif self.dissipationTerm == 'laplacian_enstrophy':
            print('Enstrophy sink form of Laplacian diffusion')
            self.dissipationFunctionDict = {
                'dissipationFunc': self.laplacian_enstrophy
            }
        elif self.dissipationTerm == 'laplacian_energy':
            print('Energy sink form of Laplacian diffusion')
            self.dissipationFunctionDict = {
                'dissipationFunc': self.laplacian_energy
            }

    ##########################################################################
    # UV FUNCTIONS
    ##########################################################################  

    def calc_UV(self,psi,u_return,v_return):

        setattr(self,u_return,-(2/self.d)*((psi[1:,:] - psi[:-1,:])/(self.bathy_np[:-1,:] + self.bathy_np[1:,:])))
        setattr(self,v_return,(2/self.d)*((psi[:,1:] - psi[:,:-1])/(self.bathy_np[:,:-1] + self.bathy_np[:,1:])))

    ##########################################################################
    # DIFFERENTIATE FUNCTIONS
    ##########################################################################  

    def dy_YGXG_pad0(self,var):
        var_dy = (var[2:,:] - var[:-2,:])/(2*self.dy)
        var_dy = np.pad(var_dy,((1,1),(0,0)),constant_values=0)
        return var_dy 

    def dx_YGXG_pad0(self,var):
        var_dx = (var[:,2:] - var[:,:-2])/(2*self.dx)
        var_dx = np.pad(var_dx,((0,0),(1,1)),constant_values=0)
        return var_dx 

    def dy_YGXG(self,var):

        var_dy = (var[2:,:] - var[:-2,:])/(2*self.dy)
        var_dy_pad = np.pad(var_dy,((1,1),(0,0)),constant_values=0)

        var_dy_edge = np.zeros_like(var_dy_pad)
        var_dy_edge[0,:] = (var[1,:] - var[0,:])/self.dy 
        var_dy_edge[-1,:] = (var[-1,:] - var[-2,:])/self.dy

        var_dy_BC = var_dy_pad + var_dy_edge

        return var_dy_BC

    def dx_YGXG(self,var):

        var_dx = (var[:,2:] - var[:,:-2])/(2*self.dx)
        var_dx_pad = np.pad(var_dx,((0,0),(1,1)),constant_values=0)

        var_dx_edge = np.zeros_like(var_dx_pad)
        var_dx_edge[:,0] = (var[:,1] - var[:,0])/self.dx 
        var_dx_edge[:,-1] = (var[:,-1] - var[:,-2])/self.dx

        var_dx_BC = var_dx_pad + var_dx_edge

        return var_dx_BC
    

    ##########################################################################
    # ADVECTION FUNCTIONS
    ##########################################################################  

    def advection_xi(self):
        
        self.adv_n = (-1/(12*self.d**2))*((self.psibar_n[:-2,1:-1] + self.psibar_n[:-2,2:] - self.psibar_n[2:,1:-1] - self.psibar_n[2:,2:])*(self.qbar_n[1:-1,2:] - self.qbar_n[1:-1,1:-1]) + \
                                         (self.psibar_n[:-2,:-2] + self.psibar_n[:-2,1:-1] - self.psibar_n[2:,:-2] - self.psibar_n[2:,1:-1])*(self.qbar_n[1:-1,1:-1] - self.qbar_n[1:-1,:-2]) + \
                                            (self.psibar_n[1:-1,2:] + self.psibar_n[2:,2:] - self.psibar_n[1:-1,:-2] - self.psibar_n[2:,:-2])*(self.qbar_n[2:,1:-1] - self.qbar_n[1:-1,1:-1]) + \
                                                (self.psibar_n[:-2,2:] + self.psibar_n[1:-1,2:] - self.psibar_n[:-2,:-2] - self.psibar_n[1:-1,:-2])*(self.qbar_n[1:-1,1:-1] - self.qbar_n[:-2,1:-1]) + \
                                                    (self.psibar_n[1:-1,2:] - self.psibar_n[2:,1:-1])*(self.qbar_n[2:,2:]  - self.qbar_n[1:-1,1:-1]) + \
                                                        (self.psibar_n[:-2,1:-1] - self.psibar_n[1:-1,:-2])*(self.qbar_n[1:-1,1:-1] - self.qbar_n[:-2,:-2]) + \
                                                            (self.psibar_n[2:,1:-1] - self.psibar_n[1:-1,:-2])*(self.qbar_n[2:,:-2] - self.qbar_n[1:-1,1:-1]) + \
                                                                (self.psibar_n[1:-1,2:] - self.psibar_n[:-2,1:-1])*(self.qbar_n[1:-1,1:-1] - self.qbar_n[:-2,2:]))
        

        # pad with zero values on boundaries
        self.adv_n = np.pad(self.adv_n,((1,1)),constant_values=0)

    def tracer_advect(self,var,var_return):

        var = np.array(var)

        # interpolate psi onto cell centres
        self.psiYCXC_n = (self.psibar_n[:-1,:-1] + self.psibar_n[:-1,1:] + self.psibar_n[1:,:-1] + self.psibar_n[1:,1:])/4

        adv_n = (1/(2*self.d**2))*(\
            (var[1:-1,1:-1] + var[2:,1:-1])*(self.psiYCXC_n[1:,1:] - self.psiYCXC_n[1:,:-1]) - \
                (var[1:-1,1:-1] + var[1:-1,2:])*(self.psiYCXC_n[1:,1:] - self.psiYCXC_n[:-1,1:]) - \
                    (var[1:-1,1:-1] + var[:-2,1:-1])*(self.psiYCXC_n[:-1,1:] - self.psiYCXC_n[:-1,:-1]) + \
                        (var[1:-1,1:-1] + var[1:-1,:-2])*(self.psiYCXC_n[1:,:-1] - self.psiYCXC_n[:-1,:-1]))

        setattr(self,var_return,np.pad(adv_n,((1,1)),constant_values=0))

    ##########################################################################
    # ELLIPTIC FUNCTIONS
    ##########################################################################  

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

    ##########################################################################
    # WIND STRESS FUNCTIONS
    ##########################################################################  

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
        tau_xi = np.roll(tau_xi,len(tau_xi[0]))

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
            'xi_xi': self.calc_xiSquare,
            'q': self.calc_q,
            'q_q': self.calc_qSquare,
            'enstrophy': self.calc_enstrophy,
            'u_YGXG': self.calc_uYGXG,
            'v_YGXG': self.calc_vYGXG,
            'u_u_YGXG': self.calc_uuYGXG,
            'v_v_YGXG': self.calc_vvYGXG
        }

        self.diagnosticsCoordDict = {
            'xi_u': ['T','YG','XG'],
            'xi_v': ['T','YG','XG'],
            'u_u': ['T', 'YC', 'XG'],
            'v_v': ['T', 'YG', 'XC'],
            'xi_xi': ['T', 'YG', 'XG'],
            'q': ['T', 'YG', 'XG'],
            'q_q': ['T', 'YG', 'XG'],
            'enstrophy': ['T', 'YG', 'XG'],
            'u_YGXG': ['T','YGm1','XG'],
            'v_YGXG': ['T','YG','XGm1'],
            'u_u_YGXG': ['T','YGm1','XG'],
            'v_v_YGXG': ['T','YG','XGm1']
        }

        self.diagnosticsMeanDict = {
            'xi_u': 'xi_u_MEAN',
            'xi_v': 'xi_v_MEAN',
            'u_u': 'u_u_MEAN',
            'v_v': 'v_v_MEAN',
            'xi_xi': 'xi_xi_MEAN',
            'q': 'q_MEAN',
            'q_q': 'q_q_MEAN',
            'enstrophy': 'enstrophy_MEAN',
            'u_YGXG': 'u_YGXG_MEAN',
            'v_YGXG': 'v_YGXG_MEAN',
            'u_u_YGXG': 'u_u_YGXG_MEAN',
            'v_v_YGXG': 'v_v_YGXG_MEAN'
        }

        self.diagnosticsNP1Dict = {
            'xi_u': 'xi_u_np1',
            'xi_v': 'xi_v_np1',
            'u_u': 'u_u_np1',
            'v_v': 'v_v_np1',
            'xi_xi': 'xi_xi_np1',
            'q': 'q_np1',
            'q_q': 'q_q_np1',
            'enstrophy': 'enstrophy_np1',
            'u_YGXG': 'u_YGXG_np1',
            'v_YGXG': 'v_YGXG_np1',
            'u_u_YGXG': 'u_u_YGXG_np1',
            'v_v_YGXG': 'v_v_YGXG_np1'
        }

        self.diagnosticsSumDict = {
            'xi_u': 'xi_u_sum',
            'xi_v': 'xi_v_sum',
            'u_u': 'u_u_sum',
            'v_v': 'v_v_sum',
            'xi_xi': 'xi_xi_sum',
            'q': 'q_sum',
            'q_q': 'q_q_sum',
            'enstrophy': 'enstrophy_sum',
            'u_YGXG': 'u_YGXG_sum',
            'v_YGXG': 'v_YGXG_sum',
            'u_u_YGXG': 'u_u_YGXG_sum',
            'v_v_YGXG': 'v_v_YGXG_sum'
        }

    def calc_xi_u(self,xi,psi,u,v,var_return):

        setattr(self,var_return,np.pad(xi[1:-1,1:-1]*((u[1:,1:-1] + u[:-1,1:-1])/2),((1,1)),constant_values=0))

    def calc_xi_v(self,xi,psi,u,v,var_return):

        setattr(self,var_return,np.pad(xi[1:-1,1:-1]*((v[1:-1,1:] + v[1:-1,:-1])/2),((1,1)),constant_values=0))

    def calc_uSquare(self,xi,psi,u,v,var_return):

        setattr(self,var_return,u**2)

    def calc_vSquare(self,xi,psi,u,v,var_return):

        setattr(self,var_return,v**2)

    def calc_xiSquare(self,xi,psi,u,v,var_return):

        setattr(self,var_return,xi**2)

    def calc_q(self,xi,psi,u,v,var_return):

        setattr(self,var_return,np.array((self.f+xi)/self.bathy))

    def calc_qSquare(self,xi,psi,u,v,var_return):

        setattr(self,var_return,np.array((self.f+xi)/self.bathy)**2)

    def calc_enstrophy(self,xi,psi,u,v,var_return):

        setattr(self,var_return,(np.array((self.f+xi)/self.bathy)**2)/2)

    def calc_uYGXG(self,xi,psi,u,v,var_return):
        setattr(self,var_return,(u[1:] + u[:-1])/2)

    def calc_vYGXG(self,xi,psi,u,v,var_return):
        setattr(self,var_return,(v[:,1:] + v[:,:-1])/2)

    def calc_uuYGXG(self,xi,psi,u,v,var_return):
        setattr(self,var_return,((u[1:] + u[:-1])/2)**2)

    def calc_vvYGXG(self,xi,psi,u,v,var_return):
        setattr(self,var_return,(((v[:,1:] + v[:,:-1])/2)**2))

    ##########################################################################
    # PARAMETERIZATION
    ##########################################################################

    def schemeDict(self):
        if self.eddy_scheme != 'EGEC' and self.eddy_scheme != 'EGEC_TEST' and self.eddy_scheme != 'EGECAD' and self.eddy_scheme != 'EGECAD_TEST' \
            and self.eddy_scheme != 'constant' and self.eddy_scheme != 'EGECADB' and self.eddy_scheme != False and self.eddy_scheme != 'AdvectionOnly':
            raise ValueError('scheme parameter should be set to \'EGEC\', \'EGECAD\' , \'EGECADB\' or False.')

        if self.eddy_scheme == 'EGEC':
            print('Energy conversion and enstrophy generation only. Calculates eddy fluxes.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_EGEC,
                'KQ_timestep' : self.timestep_KQ,
                'calc_eddyFluxes': self.eddyFluxes_scheme
            }
        elif self.eddy_scheme == 'EGEC_TEST':
            print('Energy conversion and enstrophy generation only. Doesn\'t calculate eddy fluxes.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_EGEC,
                'KQ_timestep' : self.noTimestep,
                'calc_eddyFluxes': self.noEddyFluxes
            }
        elif self.eddy_scheme == 'EGECAD':
            print('Full scheme with advection and diffusion. Calculates eddy fluxes.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_EGECAD,
                'KQ_timestep' : self.timestep_KQ,
                'calc_eddyFluxes': self.eddyFluxes_scheme
            }
        elif self.eddy_scheme == 'EGECAD_TEST':
            print('Full scheme with advection and diffusion. Doesn\'t calculate eddy fluxes.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_EGECAD,
                'KQ_timestep' : self.noTimestep,
                'calc_eddyFluxes': self.noEddyFluxes
            }
        elif self.eddy_scheme == 'constant':
            print('Scheme with constant value of kappa_q.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.no_K_Q,
                'KQ_timestep' : self.noTimestep,
                'calc_eddyFluxes': self.eddyFluxes_constant
            }
        elif self.eddy_scheme == 'AdvectionOnly':
            print('Adection of K and Lambda only. Does not calculate eddy fluxes.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_AdvectionOnly, 
                'KQ_timestep' : self.timestep_KQ,
                'calc_eddyFluxes': self.noEddyFluxes
            }
        elif self.eddy_scheme == 'EGECADB':
            print('Full scheme. Includes EKE backscatter term.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.K_Q_EGECADB,
                'KQ_timestep' : self.timestep_KQ, 
                'calc_eddyFluxes': self.eddyFluxes_scheme
            }
        else:
            print('No scheme.')
            self.schemeFunctionsDict = {
                'calc_K_Q': self.no_K_Q,
                'KQ_timestep' : self.noTimestep,
                'calc_eddyFluxes': self.noEddyFluxes
            }

    def advection_YCXC(self,var,var_return):

        var_pad = np.pad(var,((1,1)),constant_values=0)
        
        var_adv = (1/(4*self.d))*\
            ((self.bathy_np[1:,:-1] + self.bathy_np[1:,1:])*(var_pad[1:-1,1:-1] + var_pad[2:,1:-1])*self.vbar_n[1:,:] + \
                ((self.bathy_np[1:,1:] + self.bathy_np[:-1,1:])*(var_pad[1:-1,1:-1] + var_pad[1:-1,2:])*self.ubar_n[:,1:]) - \
                    ((self.bathy_np[:-1,1:] + self.bathy_np[:-1,:-1])*(var_pad[1:-1,1:-1] + var_pad[:-2,1:-1])*self.vbar_n[:-1,:]) - \
                        ((self.bathy_np[1:,:-1] + self.bathy_np[:-1,:-1])*(var_pad[1:-1,1:-1] + var_pad[1:-1,:-2])*self.ubar_n[:,:-1]))

        setattr(self,var_return,var_adv)      


    def K_Q_AdvectionOnly(self):

        # advect enstrophy and energy
        self.advection_YCXC(var=self.Q_n,var_return='QAdv_n')
        self.advection_YCXC(var=self.K_n,var_return='KAdv_n')

        # diffuse enstrophy
        self.Q_n_ghost = np.pad(self.Q_n,((1,1)),mode='edge')
        self.laplacian(var=self.Q_n_ghost,mu=self.mu_PAR,var_return='QDiff_L_n')
        self.QDiff_L_n = self.QDiff_L_n[1:-1,1:-1]

        # diffuse energy
        self.K_n_ghost = np.pad(self.K_n,((1,1)),mode='edge')
        self.laplacian(var=self.K_n_ghost,mu=self.mu_PAR,var_return='KDiff_L_n')
        self.KDiff_L_n = self.KDiff_L_n[1:-1,1:-1]

        # dQdt and dKdt
        self.Q_F_n = -self.QAdv_n/self.bathy_YCXC + self.QDiff_L_n - (self.Q_n - self.Q_min_array)*self.r_Q 
        self.K_F_n = -self.KAdv_n/self.bathy_YCXC + self.KDiff_L_n - (self.K_n - self.K_min_array)*self.r_K

    def EGEC_test(self):
        # NOTE: we use the values at time step n here to calculate dQ/dt_n and dK/dt_n so that we can claculate Q_np1 and K_np1

        # interpolate Lambda and K to YGXC and YCXG points. do not calculate at boundaries. 
        self.Q_n_YGXC = (self.Q_n[1:,:] + self.Q_n[:-1,:])/2 # YGXC [1:-1,:]
        self.Q_n_YCXG = (self.Q_n[:,1:] + self.Q_n[:,:-1])/2 # YCXG [:,1:-1]
        self.K_n_YGXC = (self.K_n[1:,:] + self.K_n[:-1,:])/2 # YGXC [1:-1,:]
        self.K_n_YCXG = (self.K_n[:,1:] + self.K_n[:,:-1])/2 # YCXG [:,1:-1]

        # calculate dqdx at YGXC points
        self.dqdx_n_YGXC = (self.qbar_n[:,1:] - self.qbar_n[:,:-1])/self.dx # YGXC
        # calculate dqdy at YCXG points
        self.dqdy_n_YCXG = (self.qbar_n[1:,:] - self.qbar_n[:-1,:])/self.dy # YCXG
        # 4 point average of dqdx to YCXG points. pad with zeros on E & W boundaries.
        self.dqdx_n_YCXG = np.pad((self.dqdx_n_YGXC[1:,:-1] + self.dqdx_n_YGXC[1:,1:] + self.dqdx_n_YGXC[:-1,1:] + self.dqdx_n_YGXC[:-1,:-1])/4,\
                                  ((0,0),(1,1)),constant_values=0) # YCXG
        # 4 point average of dqdy to YGXC points. pad with zeros at N & S boundaries.
        self.dqdy_n_YGXC = np.pad((self.dqdy_n_YCXG[1:,:-1] + self.dqdy_n_YCXG[1:,1:] + self.dqdy_n_YCXG[:-1,:-1] + self.dqdy_n_YCXG[:-1,1:])/4,\
                                  ((1,1),(0,0)),constant_values=0) # YGXC
        # calculate mod grad qbar on YGXC
        self.mod_grad_qbar_n_YGXC = np.sqrt(self.dqdx_n_YGXC**2 + self.dqdy_n_YGXC**2) # YGXC [1:-1,:]
        # set minimum value
        self.mod_grad_qbar_n_YGXC = np.where(self.mod_grad_qbar_n_YGXC<self.min_val,\
                                     self.min_val*np.ones_like(self.mod_grad_qbar_n_YGXC),self.mod_grad_qbar_n_YGXC)
        # calculate mod grad qbar on YCXG
        self.mod_grad_qbar_n_YCXG = np.sqrt(self.dqdx_n_YCXG**2 + self.dqdy_n_YCXG**2) # YCXG [:,1:-1]
        # set minimum value
        self.mod_grad_qbar_n_YCXG = np.where(self.mod_grad_qbar_n_YCXG<self.min_val,\
                                     self.min_val*np.ones_like(self.mod_grad_qbar_n_YCXG),self.mod_grad_qbar_n_YCXG)
        # calculate kappa_q at YGXC and YCXG points. Set to zero on boundaries for zero flux BC. 
        self.kappa_n_YGXC = 2*self.gamma_q*np.sqrt(self.Q_n_YGXC*self.K_n_YGXC)/self.mod_grad_qbar_n_YGXC[1:-1,:] # YGXC [1:-1,:]
        self.kappa_n_YCXG = 2*self.gamma_q*np.sqrt(self.Q_n_YCXG*self.K_n_YCXG)/self.mod_grad_qbar_n_YCXG[:,1:-1] # YCXG [:,1:-1]
        # set kappa_n_YGXC to zero on the northern and southern boundaries
        self.kappa_n_YGXC = np.pad(self.kappa_n_YGXC,((1,1),(0,0)),constant_values=0) # YGXC [:,:]
        # set kappa_n_YCXG to zero on the eastern and western boundaries 
        self.kappa_n_YCXG = np.pad(self.kappa_n_YCXG,((0,0),(1,1)),constant_values=0) # YCXG [:,:]

        # calculate qu_EDDY = bar{q'u'} at YCXG points. 
        self.qu_EDDY_n_YCXG = -self.kappa_n_YCXG*self.dqdx_n_YCXG # YCXG [:,:]
        # calculate qv_EDDY = bar{q'v'} at YCXG points
        self.qv_EDDY_n_YGXC = -self.kappa_n_YGXC*self.dqdy_n_YGXC # YGXC [:,:]
        # calculate bar{q'u'}dqdx at YGXC points
        self.enstrophyGen_n_x_YCXG = self.qu_EDDY_n_YCXG*self.dqdx_n_YCXG # YGXC
        # calculate bqr{q'v'}dqdy at YCXG points
        self.enstrophyGen_n_y_YGXC = self.qv_EDDY_n_YGXC*self.dqdy_n_YGXC # YCXG
        # calculate enstrophy generation at YCXC points 
        self.enstrophyGen_n = (self.enstrophyGen_n_x_YCXG[:,1:] + self.enstrophyGen_n_x_YCXG[:,:-1] + self.enstrophyGen_n_y_YGXC[1:,:] + self.enstrophyGen_n_y_YGXC[:-1,:])/2 # YCXC

        # calculate dpsidx at YGXC points 
        self.dpsidx_n_YGXC = (self.psibar_n[:,1:] - self.psibar_n[:,:-1])/self.dx # YGXC
        # calculate dpsidy at YCXG points
        self.dpsidy_n_YCXG = (self.psibar_n[1:,:] - self.psibar_n[:-1,:])/self.dy # YCXG
        # 4 point average of dpsidx to YCXG points. pad zeros on E & W boundaries.
        self.dpsidx_n_YCXG = np.pad((self.dpsidx_n_YGXC[1:,:-1] + self.dpsidx_n_YGXC[1:,1:] + self.dpsidx_n_YGXC[:-1,1:] + self.dpsidx_n_YGXC[:-1,:-1])/4,\
                                    ((0,0),(1,1)),constant_values=0) # YCXG
        # 4 point average of dpsidy to YGXC points. pad zeros on N and boundaries
        self.dpsidy_n_YGXC = np.pad((self.dpsidy_n_YCXG[1:,:-1] + self.dpsidy_n_YCXG[1:,1:] + self.dpsidy_n_YCXG[:-1,:-1] + self.dpsidy_n_YCXG[:-1,1:])/4,\
                                    ((1,1),(0,0)),constant_values=0) # YGXC
        # calculate bar{q'u'}dpsidx at YCXG points
        self.energyConv_n_x_YCXG = self.qu_EDDY_n_YCXG*self.dpsidx_n_YCXG  # YGXC
        # calculate bar{q'v'}dpsidy at YGXC points
        self.energyConv_n_y_YGXC = self.qv_EDDY_n_YGXC*self.dpsidy_n_YGXC # YCXG
        # calculate energy conversion at YCXC points 
        self.energyConv_n = (self.energyConv_n_x_YCXG[:,1:] + self.energyConv_n_x_YCXG[:,:-1] + self.energyConv_n_y_YGXC[1:,:] + self.energyConv_n_y_YGXC[:-1,:])/2

        # calculate kappa_n at YCXC points 
        self.dqdx_n_YCXC = (self.qbar_n[1:,1:] + self.qbar_n[:-1,1:] - self.qbar_n[1:,:-1] - self.qbar_n[:-1,:-1])/(2*self.dx) # YCXC
        self.dqdy_n_YCXC = (self.qbar_n[1:,1:] + self.qbar_n[1:,:-1] - self.qbar_n[:-1,1:] - self.qbar_n[:-1,:-1])/(2*self.dy) # YCXC
        self.mod_grad_qbar_n_YCXC = np.sqrt(self.dqdx_n_YCXC**2 + self.dqdy_n_YCXC**2) # YCXC
        self.kappa_n = 2*self.gamma_q*np.sqrt(self.Q_n*self.K_n)/self.mod_grad_qbar_n_YCXC # YCXC

        # sum variables
        self.kappa_sum += self.kappa_n
        self.enstrophyGen_sum += self.enstrophyGen_n 
        self.energyConv_sum += self.energyConv_n

        

    def EGEC(self):
        # NOTE: we use the values at time step n here to calculate dQ/dt_n and dK/dt_n so that we can claculate Q_np1 and K_np1

        # interpolate Lambda and K to YGXC and YCXG points. do not calculate at boundaries. 
        self.Q_n_YGXC = (self.Q_n[1:,:] + self.Q_n[:-1,:])/2 # YGXC [1:-1,:]
        self.Q_n_YCXG = (self.Q_n[:,1:] + self.Q_n[:,:-1])/2 # YCXG [:,1:-1]
        self.K_n_YGXC = (self.K_n[1:,:] + self.K_n[:-1,:])/2 # YGXC [1:-1,:]
        self.K_n_YCXG = (self.K_n[:,1:] + self.K_n[:,:-1])/2 # YCXG [:,1:-1]

        # calculate dqdx at YGXC points
        self.dqdx_n_YGXC = (self.qbar_n[:,1:] - self.qbar_n[:,:-1])/self.dx # YGXC
        # calculate dqdy at YCXG points
        self.dqdy_n_YCXG = (self.qbar_n[1:,:] - self.qbar_n[:-1,:])/self.dy # YCXG
        # 4 point average of dqdx to YCXG points. do not calculate at boundaries.
        self.dqdx_n_YCXG = (self.dqdx_n_YGXC[1:,:-1] + self.dqdx_n_YGXC[1:,1:] + self.dqdx_n_YGXC[:-1,1:] + self.dqdx_n_YGXC[:-1,:-1])/4 # YCXG
        # 4 point average of dqdy to YGXC points. do not calculate at boundaries. 
        self.dqdy_n_YGXC = (self.dqdy_n_YCXG[1:,:-1] + self.dqdy_n_YCXG[1:,1:] + self.dqdy_n_YCXG[:-1,:-1] + self.dqdy_n_YCXG[:-1,1:])/4 # YGXC
        # calculate mod grad qbar on YGXC
        self.mod_grad_qbar_n_YGXC = np.sqrt(self.dqdx_n_YGXC[1:-1,:]**2 + self.dqdy_n_YGXC**2) # YGXC [1:-1,:]
        # set minimum value
        self.mod_grad_qbar_n_YGXC = np.where(self.mod_grad_qbar_n_YGXC<self.min_val,\
                                     self.min_val*np.ones_like(self.mod_grad_qbar_n_YGXC),self.mod_grad_qbar_n_YGXC)
        # calculate mod grad qbar on YCXG
        self.mod_grad_qbar_n_YCXG = np.sqrt(self.dqdx_n_YCXG**2 + self.dqdy_n_YCXG[:,1:-1]**2) # YCXG [:,1:-1]
        # set minimum value
        self.mod_grad_qbar_n_YCXG = np.where(self.mod_grad_qbar_n_YCXG<self.min_val,\
                                     self.min_val*np.ones_like(self.mod_grad_qbar_n_YCXG),self.mod_grad_qbar_n_YCXG)
        # calculate kappa_q at YGXC and YCXG points. Set to zero on boundaries for zero flux BC. 
        self.kappa_n_YGXC = 2*self.gamma_q*np.sqrt(self.Q_n_YGXC*self.K_n_YGXC)/self.mod_grad_qbar_n_YGXC # YGXC [1:-1,:]
        self.kappa_n_YCXG = 2*self.gamma_q*np.sqrt(self.Q_n_YCXG*self.K_n_YCXG)/self.mod_grad_qbar_n_YCXG # YCXG [:,1:-1]
        # set kappa_n_YGXC to zero on the northern and southern boundaries
        self.kappa_n_YGXC = np.pad(self.kappa_n_YGXC,((1,1),(0,0)),constant_values=0) # YGXC [:,:]
        # set kappa_n_YCXG to zero on the eastern and western boundaries 
        self.kappa_n_YCXG = np.pad(self.kappa_n_YCXG,((0,0),(1,1)),constant_values=0) # YCXG [:,:]
        # calculate qu_EDDY = bar{q'u'} at YGXC points
        self.qu_EDDY_n_YGXC = -self.kappa_n_YGXC*self.dqdx_n_YGXC # YGXC [:,:]
        # calculate qv_EDDY = bar{q'v'} at YCXG points
        self.qv_EDDY_n_YCXG = -self.kappa_n_YCXG*self.dqdy_n_YCXG # YCXG [:,:]
        # calculate bar{q'u'}dqdx at YGXC points
        self.enstrophyGen_n_x_YGXC = self.qu_EDDY_n_YGXC*self.dqdx_n_YGXC # YGXC
        # calculate bqr{q'v'}dqdy at YCXG points
        self.enstrophyGen_n_y_YCXG = self.qv_EDDY_n_YCXG*self.dqdy_n_YCXG # YCXG
        # calculate enstrophy generation at YCXC points 
        self.enstrophyGen_n = (self.enstrophyGen_n_x_YGXC[1:,:] + self.enstrophyGen_n_x_YGXC[:-1,:] + self.enstrophyGen_n_y_YCXG[:,1:] + self.enstrophyGen_n_y_YCXG[:,:-1])/2 # YCXC

        # calculate dpsidx at YGXC points 
        self.dpsidx_n_YGXC = (self.psibar_n[:,1:] - self.psibar_n[:,:-1])/self.dx # YGXC
        # calculate dpsidy at YCXG points
        self.dpsidy_n_YCXG = (self.psibar_n[1:,:] - self.psibar_n[:-1,:])/self.dy # YCXG
        # calculate bar{q'u'}dpsidx at YGXC points
        self.energyConv_n_x_YGXC = self.qu_EDDY_n_YGXC*self.dpsidx_n_YGXC  # YGXC
        # calculate bar{q'v'}dpsidy at YCXG points
        self.energyConv_n_y_YCXG = self.qv_EDDY_n_YCXG*self.dpsidy_n_YCXG # YCXG
        # calculate energy conversion at YCXC points 
        self.energyConv_n = (self.energyConv_n_x_YGXC[1:,:] + self.energyConv_n_x_YGXC[:-1,:] + self.energyConv_n_y_YCXG[:,1:] + self.energyConv_n_y_YCXG[:,:-1])/2

        # calculate kappa_n at YCXC points 
        self.dqdx_n_YCXC = (self.qbar_n[1:,1:] + self.qbar_n[:-1,1:] - self.qbar_n[1:,:-1] - self.qbar_n[:-1,:-1])/(2*self.dx) # YCXC
        self.dqdy_n_YCXC = (self.qbar_n[1:,1:] + self.qbar_n[1:,:-1] - self.qbar_n[:-1,1:] - self.qbar_n[:-1,:-1])/(2*self.dy) # YCXC
        self.mod_grad_qbar_n_YCXC = np.sqrt(self.dqdx_n_YCXC**2 + self.dqdy_n_YCXC**2) # YCXC
        self.kappa_n = 2*self.gamma_q*np.sqrt(self.Q_n*self.K_n)/self.mod_grad_qbar_n_YCXC # YCXC

        # sum variables
        self.kappa_sum += self.kappa_n
        self.enstrophyGen_sum += self.enstrophyGen_n 
        self.energyConv_sum += self.energyConv_n


    def K_Q_EGEC(self):

        self.EGEC()
        
        # dQdt and dKdt
        self.Q_F_n = -self.enstrophyGen_n 
        self.K_F_n = self.energyConv_n 

    def K_Q_EGECAD(self):
        
        self.EGEC()

        # advect enstrophy and energy
        self.advection_YCXC(var=self.Q_n,var_return='QAdv_n')
        self.advection_YCXC(var=self.K_n,var_return='KAdv_n')

        # diffuse enstrophy
        # add ghost points of edge values to ensure no flux through boundaries 
        self.Q_n_ghost = np.pad(self.Q_n*self.bathy_YCXC,((1,1)),mode='edge')
        # calculate laplacian diffusion. outer (ghost) points will be set to zero. 
        self.laplacian(var=self.Q_n_ghost,mu=1,var_return='QDiff_L_n')
        # remove outer points and multiply by mu/H
        self.QDiff_L_n = np.array(self.mu_PAR/self.bathy_YCXC)*self.QDiff_L_n[1:-1,1:-1]

        # diffuse energy
        # add ghost points of edge values to ensure no flux through boundaries
        self.K_n_ghost = np.pad(self.K_n*self.bathy_YCXC,((1,1)),mode='edge')
        # calculate laplacian diffusion. outer (ghost) points will be set to zero. 
        self.laplacian(var=self.K_n_ghost,mu=1,var_return='KDiff_L_n')
        # remove outer points and multiply by mu/H
        self.KDiff_L_n = np.array(self.mu_PAR/self.bathy_YCXC)*self.KDiff_L_n[1:-1,1:-1]

        # dQdt and dKdt
        self.Q_F_n = -self.enstrophyGen_n  - self.QAdv_n/self.bathy_YCXC + self.QDiff_L_n - (self.Q_n - self.Q_min_array)*self.r_Q
        self.K_F_n = self.energyConv_n - self.KAdv_n/self.bathy_YCXC + self.KDiff_L_n - (self.K_n - self.K_min_array)*self.r_K

    def K_Q_EGECADB(self):
        
        self.K_Q_EGECAD()

        # backscatter term
        backscatter_integrand = -self.psibar_n*self.dissipation_n # YGXG
        backscatter_integrand_YCXC = (backscatter_integrand[1:,1:] + backscatter_integrand[1:,:-1] + \
            backscatter_integrand[:-1,1:] + backscatter_integrand[:-1,:-1])/4 # YCXC
        backscatter_integral = np.sum(backscatter_integrand_YCXC*self.dx*self.dy) # volume integral
        self.KE_backscatter_n = self.backscatter_frac*(backscatter_integral/self.volume_YCXC) # volume average


        # add to K_F_n
        self.K_F_n = self.K_F_n - self.KE_backscatter_n*np.ones_like(self.K_n)

        self.KE_backscatter_sum = self.KE_backscatter_sum + self.KE_backscatter_n*self.dt

    def timestep_KQ(self,scheme):

        # step forward Q and K
        self.Q_np1 = scheme(var_n=self.Q_n,dt=self.dt,F_n=self.Q_F_n,F_nm1=self.Q_F_nm1,F_nm2=self.Q_F_nm2)
        self.K_np1 = scheme(var_n=self.K_n,dt=self.dt,F_n=self.K_F_n,F_nm1=self.K_F_nm1,F_nm2=self.K_F_nm2)
        # set K = 0 where K is negative
        self.K_np1 = np.where(self.K_np1 < 0,0,self.K_np1)
        # set Q = 0 where Q is negative
        self.Q_np1 = np.where(self.Q_np1 < 0,0,self.Q_np1)

        # add to sum variables
        self.Q_sum += self.Q_np1
        self.K_sum += self.K_np1

    def noTimestep(self,scheme):
        return


    def no_K_Q(self):
        self.K_np1 = np.zeros_like(self.K_n) 
        self.Q_np1 = np.zeros_like(self.Q_n) 

    def eddyFluxes_scheme(self):

        self.eddyFluxes_n = (1/self.d)*(self.bathy_YCXG[1:,1:-1]*self.qv_EDDY_n_YCXG[1:,1:-1] + self.bathy_YGXC[1:-1,1:]*self.qu_EDDY_n_YGXC[1:-1,1:] - \
                                        self.bathy_YCXG[:-1,1:-1]*self.qv_EDDY_n_YCXG[:-1,1:-1] - self.bathy_YGXC[1:-1,:-1]*self.qu_EDDY_n_YGXC[1:-1,:-1])
        self.eddyFluxes_n = np.pad(self.eddyFluxes_n,((1,1)),constant_values=0)

    def eddyFluxes_scheme_TEST(self):

        # 4 point interpolation of qu_EDDY_YCXG to YGXC. Do not calculate at boundaries. 
        self.qu_EDDY_n_YGXC = (self.qu_EDDY_n_YCXG[1:,:-1] + self.qu_EDDY_n_YCXG[1:,1:] + self.qu_EDDY_n_YCXG[:-1,:-1] + self.qu_EDDY_n_YCXG[:-1,1:])/4 # YGXC

        # 4 point interpolation of qv_EDDY_YGXC to YCXG. Do not calculate at boundaries.
        self.qv_EDDY_n_YCXG = (self.qv_EDDY_n_YGXC[1:,:-1] + self.qv_EDDY_n_YGXC[1:,1:] + self.qv_EDDY_n_YGXC[:-1,1:] + self.qv_EDDY_n_YGXC[:-1,:-1])/4 # YCXG


        self.eddyFluxes_n = (1/self.d)*(self.bathy_YCXG[1:,1:-1]*self.qv_EDDY_n_YCXG[1:,:] + self.bathy_YGXC[1:-1,1:]*self.qu_EDDY_n_YGXC[:,1:] - \
                                        self.bathy_YCXG[:-1,1:-1]*self.qv_EDDY_n_YCXG[:-1,:] - self.bathy_YGXC[1:-1,:-1]*self.qu_EDDY_n_YGXC[:,:-1])
        self.eddyFluxes_n = np.pad(self.eddyFluxes_n,((1,1)),constant_values=0)

    def noEddyFluxes(self):
        self.eddyFluxes_n = np.zeros_like(self.xibar_n)

    def eddyFluxes_constant(self):
        self.dqdx_n = self.dx_YGXG(var=self.qbar_n)
        self.dqdy_n = self.dy_YGXG(var=self.qbar_n)

        # ZETA fluxes
        self.flux_u_n = -self.kappa_q*self.bathy_np*self.dqdx_n
        self.flux_v_n = -self.kappa_q*self.bathy_np*self.dqdy_n

        self.eddyFluxes_n = (1/(2*self.d))*(self.flux_v_n[2:,1:-1] - self.flux_v_n[:-2,1:-1] + self.flux_u_n[1:-1,2:] - self.flux_u_n[1:-1,:-2])
        self.eddyFluxes_n = np.pad(self.eddyFluxes_n,((1,1)),constant_values=0)




#%%

'''d = 5000 # m
Nx = 200
Ny = 200
Lx = d*Nx # m
Ly = d*Ny # m
f0 = 0.7E-4 # s-1
beta = 0 # m-1s-1
# VORTICITY PARAMETERS
r_BD = 0
mu_xi = 10
dissipationTerm = 'laplacian_energy'
# EDDY SCHEME PARAMETERS
mu_PAR = 500
r_Q = 5.E-8
r_K = 0
Q_min = 0
K_min = 0
gamma_q = 0.1
min_val = 1.E-16
max_val = 1.E4
#kappa_q = 20
# WIND PARAMETERS
tau_0 = 0
rho_0 = 1025
# TIME STEPPING
dt = 900
Nt = 9600
dumpFreq =  86400
meanDumpFreq = 864000
diagnostics = ['xi_u','xi_v','u_u','v_v','xi_xi','q','q_q']

bathy_random = np.load('./../barotropic_model_analysis/model_data/FDT_MOUNT_5km_f/randomTopography_5km.npy')
#bathy_flat = 500*np.ones((Ny+1,Nx+1))

domain = Barotropic(d=d,Nx=Nx,Ny=Ny,bathy=bathy_random,f0=f0,beta=beta)'''


#%%

'''init_K = 1.5E-4*np.ones((Ny,Nx))
init_Q = 1.E-20*np.ones((Ny,Nx))
print(np.shape(init_Q))'''

'''#init_psi = np.zeros((Ny+1,Nx+1))
init_psi = np.load('./../barotropic_model_analysis/model_data/FDT_MOUNT_5km_f/initPsi_5km.npy')
domain.init_psi(init_psi)
domain.xi_from_psi()'''


#%%

'''data = domain.model(dt=dt,\
    Nt=Nt,\
        r_BD=r_BD,\
            mu_xi=mu_xi,\
                tau_0=tau_0,\
                    rho_0=rho_0,\
                        dumpFreq=dumpFreq,\
                            meanDumpFreq=meanDumpFreq,\
                                diags=diagnostics,\
                                    dissipationTerm=dissipationTerm)'''

'''data = domain.model(dt=dt,\
    Nt=Nt,\
        dumpFreq=dumpFreq,\
            meanDumpFreq=meanDumpFreq,\
                r_BD=r_BD,\
                    mu_xi_L=mu_xi_L,\
                        mu_xi_B=mu_xi_B,\
                            tau_0=tau_0,\
                                rho_0=rho_0,\
                                    eddy_scheme='EGECAD',\
                                        init_K=init_K,\
                                            init_Q=init_Q,\
                                                gamma_q=gamma_q,\
                                                    mu_PAR=mu_PAR,\
                                                        r_Q=r_Q,\
                                                            r_K=r_K,\
                                                                Q_min=Q_min,\
                                                                    K_min=K_min,\
                                                                        min_val=min_val,\
                                                                            max_val = max_val,\
                                                                                backscatter_frac=1.,\
                                                                                    diags=diagnostics)'''

#%%
'''t = 0
plt.contourf(data.XG,data.YG,data.psi[t])
plt.colorbar()
plt.show()

t = -1
plt.contourf(data.XG,data.YG,data.psi[t])
plt.colorbar()
plt.show()'''



#%%

'''qbar = np.array((data.xi + data.f)/data.bathy)
biharm_enstr = qbar*domain.biharm_term

#%%
import sys
sys.path.insert(0,'./../barotropic_model_analysis/analysis/functions')
from derivatives import *
from integrals import *
from regrid import *

#%%

def area_int(var,dx,dy):

    # calculate the area integral of variable=var
    # NOTE: var is defined on the YG,XG points. 
    var = np.array(var)
    
    var_intX = (dx/2)*(var[:,1:] + var[:,:-1])
    var_intX = np.nansum(var_intX,axis=1)

    var_intXY = (dy/2)*(var_intX[1:] + var_intX[:-1])
    var_intXY = np.nansum(var_intXY,axis=0)

    return var_intXY

#%%

biharm_enstr_volAvg = np.zeros(len(biharm_enstr))
biharm_volAvg = np.zeros(len(domain.biharm_term))
enstrophyGen_volAvg = np.zeros(len(data.enstrophyGen))
enstrophy_volAvg = np.zeros((len(data.q_q)))
volume_YGXG = basin_volume_YGXG(data)
volume_YCXC = basin_volume_YCXC(data)

for t in range(len(biharm_enstr)):
    biharm_enstr_volAvg[t] = area_int(var=biharm_enstr[t],dx=data.dx,dy=data.dy)/volume_YGXG
    biharm_volAvg[t] = vol_int(var=domain.biharm_term[t],dx=data.dx,dy=data.dy,h=data.bathy)/volume_YGXG
    enstrophyGen_volAvg[t] = vol_int_YCXC(var=data.enstrophyGen[t],dx=data.dx,dy=data.dy,h=data.bathy)/volume_YCXC
    enstrophy_volAvg[t] = vol_int(var=data.q_q[t]/2,dx=data.dx,dy=data.dy,h=data.bathy)/volume_YGXG


# %%

plt.plot(np.arange(1,len(biharm_enstr_volAvg)),-biharm_enstr_volAvg[1:],label='biharmonic term')
plt.plot(np.arange(1,len(enstrophyGen_volAvg)),enstrophyGen_volAvg[1:],label='eddy to mean enstrophy generation')
plt.plot(np.arange(1,len(enstrophyGen_volAvg)),-biharm_enstr_volAvg[1:]+enstrophyGen_volAvg[1:],label='sum')
plt.legend()
plt.title('Contribution to d/dt of mean enstrophy')
plt.show()

#%%
plt.plot(np.arange(1,len(enstrophy_volAvg)),enstrophy_volAvg[1:])
plt.show()

#%%
enstrophy_dt = (enstrophy_volAvg[1:] - enstrophy_volAvg[:-1])/86400

plt.plot(np.arange(len(enstrophy_dt)),enstrophy_dt)
plt.title('d/dt enstrophy')
plt.show()

#%%
plt.plot(np.arange(1,2001),enstrophyGen_volAvg[1:],label='eddy to mean enstrophy generation')
plt.legend()
plt.show()

# %%
enstrophy = (qbar**2)/2
enstrophy_volAvg = np.zeros(len(enstrophy))
for t in range(len(enstrophy)):
    enstrophy_volAvg[t] = vol_int(var=enstrophy[t],dx=data.dx,dy=data.dy,h=data.bathy)/volume_YGXG
# %%
plt.plot(np.arange(2001),enstrophy_volAvg)
plt.show()

#%%

enstrophy_dt = (enstrophy_volAvg[1:] - enstrophy_volAvg[:-1])/86400

plt.plot(np.arange(1001),-biharm_enstr_volAvg,label='biharmonic term')
plt.plot(np.arange(1001),enstrophyGen_volAvg,label='eddy to mean enstrophy generation')
plt.plot(np.arange(1000),enstrophy_dt,label='enstrophy')
plt.legend()
plt.show()

# %%
plt.plot(np.arange(1000),enstrophy_dt,label='enstrophy')
plt.show()

# %%
adv_enstr = qbar*domain.advection
adv_enstr_volAvg = np.zeros(len(adv_enstr))

for t in range(len(adv_enstr)):
    adv_enstr_volAvg[t] = vol_int(var=adv_enstr[t],dx=data.dx,dy=data.dy,h=data.bathy)/volume_YGXG

plt.plot(np.arange(1001),adv_enstr_volAvg)

#%%

plt.plot(np.arange(1001),adv_enstr_volAvg - biharm_enstr_volAvg + enstrophyGen_volAvg)
plt.show()
plt.plot(np.arange(1001),enstrophy_volAvg)
plt.show()


#%%

KE = (u_YCXC(data.u_u) + v_YCXC(data.v_v))/2

KE_volAvg = np.zeros(len(KE))
enstrophyGen_volAvg = np.zeros(len(data.enstrophyGen))

volume = basin_volume_YCXC(data)

for t in range(len(KE)):
    KE_volAvg[t] = vol_int_YCXC(var=KE[t],dx=data.dx,dy=data.dy,h=data.bathy)/volume
    enstrophyGen_volAvg[t] = vol_int_YCXC(var=data.enstrophyGen[t],dx=data.dx,dy=data.dy,h=data.bathy)/volume

#%%
plt.plot(np.arange(1,len(KE_volAvg)),KE_volAvg[1:])
plt.show()

#%%
plt.plot(np.arange(1,len(enstrophyGen_volAvg)),-enstrophyGen_volAvg[1:])


# %%
plt.contourf(data.XC,data.YC,data.Q[4])
plt.colorbar()
plt.show()
# %%
print(np.shape(domain.mu_PAR/domain.bathy_YCXC))'''
# %%
