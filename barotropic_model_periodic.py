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
            np.shape(bathy) == (Ny,Nx)
        except ValueError:
            print('bathy must have shape (Ny,Nx) and be defined on the grid corners.')
        else:

            self.d = d
            self.dx = d
            self.dy = d
            self.Nx = Nx
            self.Ny = Ny 
            self.Lx = int(d*Nx)
            self.Ly = int(d*Ny)
            self.XG = np.array([i*self.dx for i in range(self.Nx)])
            self.YG = np.array([j*self.dy for j in range(self.Ny)])
            self.XC = np.array([(i+0.5)*self.dx for i in range(self.Nx)])
            self.YC = np.array([(j+0.5)*self.dy for j in range(self.Ny)])
            self.NxC = self.Nx 
            self.NyC = self.Ny
            self.NxG = self.Nx
            self.NyG = self.Ny

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

            self.bathy_YCXC = (np.roll(np.roll(self.bathy_np,-1,0),-1,1) + np.roll(self.bathy_np,-1,1) + \
                               np.roll(self.bathy_np,-1,0) + self.bathy_np)/4

            self.bathy_YGXC = (np.roll(self.bathy_np,-1,1) + self.bathy_np)/2
            self.bathy_YCXG = (np.roll(self.bathy_np,-1,0) + self.bathy_np)/2

            self.diagnosticsDict()

    #########################################################################
    # INITIAL CONDITIONS FUNCTIONS
    ##########################################################################
            
    def init_psi(self,psi,i_zero,j_zero):
        try:
            np.shape(psi) == ((self.NyG,self.NxG))
        except ValueError:
            print('Initial psi must have shape (Ny,Nx) and be defined on the grid corners.')
        else:
            self.start = 'FE'
            self.i_zero = i_zero
            self.j_zero = j_zero
            self.psibar_0 = psi
            self.calc_UV(psi=self.psibar_0,u_return='ubar_0',v_return='vbar_0')
            self.xi_from_psi()

    def init_xi(self,xi,i_zero,j_zero):
        try:
            np.shape(xi) == ((self.NyG,self.NxG))
        except ValueError:
            print('Initial xi must have shape (Ny+1,Nx+1) to be defined on the grid corners.')
        else:
            self.start = 'FE'
            self.i_zero = i_zero
            self.j_zero = j_zero
            self.xibar_0 = xi
            self.psi_from_xi()

    def psi_from_xi(self):
        try:
            self.xibar_0
        except NameError:
            print('Specifiy initial xi to calculate xi from psi.')
        else:
            # get elliptic solver
            self.elliptic()

            psibar = self.LU.solve((np.array(self.xibar_0)).flatten())
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
            self.elliptic()

            self.xibar_0 = (self.xi_matrix@self.psibar_0.flatten()).reshape((self.NyG,self.NxG))
            print(np.sum(self.xibar_0))
            print(abs(np.sum(self.xi_matrix, 0)).max())
            print(abs(np.sum(self.xi_matrix, 1)).max())
            print(abs(self.psibar_0).max())
            print(abs(self.xibar_0).max())

            '''xibar_0 = (1/((self.bathy_np**2)*(self.d**2)))*(self.bathy_np*\
                (np.roll(self.psibar_0,-1,1) + np.roll(self.psibar_0,-1,0) - 4*self.psibar_0 + \
                 np.roll(self.psibar_0,1,1) + np.roll(self.psibar_0,1,0)) - \
                    (np.roll(self.bathy_np,-1,1) - np.roll(self.bathy_np,1,1))*(np.roll(self.psibar_0,-1,1) - np.roll(self.psibar_0,1,1))/4 - \
                        (np.roll(self.bathy_np,-1,0) - np.roll(self.bathy_np,1,0))*(np.roll(self.psibar_0,-1,0) - np.roll(self.psibar_0,1,0))/4)'''
            
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

    ##########################################################################
    # TIME STEPPING FUNCTIONS
    ##########################################################################
                    
    def model(self,dt,Nt,dumpFreq,meanDumpFreq,\
        r_BD,mu_xi_L,mu_xi_B,\
            tau_0,rho_0,\
                eddy_scheme=False,init_K=None,init_Q=None,gamma_q=None,mu_PAR=None,r_Q=None,\
                    r_K=None,min_val=None,max_val=None,kappa_q=None,K_min=None,Q_min=None,backscatter_frac=None,diags=[]):

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

            # store model parameters in object
            self.dt = dt
            self.Nt = Nt
            self.gamma_q = gamma_q
            self.r_BD = r_BD
            self.mu_xi_L = mu_xi_L
            self.mu_xi_B = mu_xi_B
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
            self.q_n = np.zeros((self.NyG,self.NxG))
            self.B = np.zeros((self.NyG,self.NxG))

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
            
            # initialise parameters for time stepping
            self.F_n = np.zeros_like(self.xibar_n)
            self.F_nm1 = np.zeros_like(self.xibar_n)
            self.F_nm2 = np.zeros_like(self.xibar_n)
            self.adv_n = np.zeros_like(self.xibar_n)
            self.BD_n = np.zeros_like(self.xibar_n)
            self.diffusion_L_n = np.zeros_like(self.xibar_n)
            self.diffusion_B_n = np.zeros_like(self.xibar_n)
            self.diffusion_B_n_mid = np.zeros_like(self.xibar_n)
            self.zeta_n = np.zeros_like(self.xibar_n)

            # parameters for checking enstrophy issue 
            # delete once issue is solved
            self.biharm_term = np.zeros((len_T+1,self.NyG,self.NxG))
            self.advection = np.zeros((len_T+1,self.NyG,self.NxG))
            self.q_biharm = np.zeros((len_T+1,self.NyG,self.NxG))
            self.psi_biharm = np.zeros((len_T+1,self.NyG,self.NxG))
            self.Q_diffusion = np.zeros((len_T+1,self.NyC,self.NxC))
            self.K_diffusion = np.zeros((len_T+1,self.NyC,self.NxC))
            self.Q_damping = np.zeros((len_T+1,self.NyC,self.NxC))
            self.K_damping = np.zeros((len_T+1,self.NyC,self.NxC))
            self.Q_advection = np.zeros((len_T+1,self.NyC,self.NxC))
            self.K_advection = np.zeros((len_T+1,self.NyC,self.NxC))

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
                    self.qbar_n = np.zeros((self.NyG,self.NxG))
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
                    self.K_n_YGXC = np.zeros((self.NyG,self.NxC))
                    self.K_n_YCXG = np.zeros((self.NyC,self.NxG))

                    # variables for parameterization calculation 
                    self.qbar_n = np.zeros((self.NyG,self.NxG))
                    self.dqdx_n_YCXC = np.zeros((self.NyC,self.NxC))
                    self.dqdx_n_YCXG = np.zeros((self.NyC,self.NxG))
                    self.dqdx_n_YGXC = np.zeros((self.NyG,self.NxC))
                    self.dqdy_n_YCXC = np.zeros((self.NyC,self.NxC))
                    self.dqdy_n_YCXG = np.zeros((self.NyC,self.NxG))
                    self.dqdy_n_YGXC = np.zeros((self.NyG,self.NxC))

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

                
            else:
                self.Q_0 = np.zeros((self.NyC,self.NxC))
                self.Q_n = np.zeros((self.NyC,self.NxC))
                self.Q_np1 = np.zeros((self.NyC,self.NxC))
                self.K_0 = np.zeros((self.NyC,self.NxC))
                self.K_n = np.zeros((self.NyC,self.NxC))
                self.K_np1 = np.zeros((self.NyC,self.NxC))

            self.advection = np.zeros((self.Nt,self.NyG,self.NxG))

            self.nt = 0

            # run scheemeDict function to get schemeFunctionsDict which contains functions to run if scheme is True or False
            self.schemeDict()

            # time stepping function

            #def time_step(scheme,F_nm2, F_nm1, F_n, xibar_n, xibar_np1, psibar_n, psibar_np1):
            def time_step(scheme,**kw):

                def wrapper(*args,**kwargs):

                    # calculate terms used in functions
                    self.zeta_n = np.array(self.f) + self.xibar_n 
                    self.q_n = self.zeta_n/self.bathy_np

                    # CALCULATE VORTICITY AT NEXT TIME STEP
                    # calculate advection term
                    self.advection_xi()
                    self.advection[self.nt] = self.adv_n
                    # dissiaption from bottom drag
                    self.BD_n = self.r_BD*self.xibar_n
                    # diffusion of vorticity
                    self.laplacian(var=self.xibar_n,mu=self.mu_xi_L,var_return='diffusion_L_n')
                    self.biharmonic(var=self.xibar_n,mu=self.mu_xi_B,var_return='diffusion_B_n')

                    # eddy fluxes
                    self.schemeFunctionsDict['calc_K_Q']()
                    self.schemeFunctionsDict['KQ_timestep'](scheme=scheme)
                    self.schemeFunctionsDict['calc_eddyFluxes']()
                    # F_n
                    self.F_n = self.adv_n - self.eddyFluxes_n - self.BD_n + self.diffusion_L_n - self.diffusion_B_n + np.array(self.wind_stress)
                    # calculate new value of xibar
                    self.xibar_np1 = scheme(var_n=self.xibar_n,dt=self.dt,F_n=self.F_n,F_nm1=self.F_nm1,F_nm2=self.F_nm2)

                    # uncomment for solid body rotation
                    # self.psibar_np1 = self.psibar_n
                    # self.calc_UV(self.psibar_np1,u_return='ubar_np1',v_return='vbar_np1')

                    # SOLVE FOR PSI AT NEXT TIME STEP
                    # comment out for solid body rotation
                    #self.B = self.xibar_n.copy()
                    #self.B[self.j_zero,self.i_zero] = 0
                    self.psibar_np1 = self.LU.solve((-np.array(self.xibar_np1)).flatten())
                    self.psibar_np1 = self.psibar_np1.reshape((self.NyG,self.NxG))

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
                            self.energyCheck[self.index-1] = self.psibar_n*self.adv_n
                            self.biharm_term[self.index-1] = self.diffusion_B_n
                            self.advection[self.index-1] = self.adv_n
                            self.q_biharm[self.index-1] = self.qbar_n*self.diffusion_B_n 
                            self.psi_biharm[self.index-1] = self.psibar_n*self.diffusion_B_n
                            self.Q_diffusion[self.index-1] = self.QDiff_L_n
                            self.K_diffusion[self.index-1] = self.KDiff_L_n 
                            self.Q_advection[self.index-1] = self.QAdv_n
                            self.K_advection[self.index-1] = self.KAdv_n
                            self.Q_damping[self.index-1] = (self.Q_n - self.Q_min_array)*self.r_Q
                            self.K_damping[self.index-1] = (self.K_n - self.K_min_array)*self.r_K

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
                    self.B = np.zeros((self.NyG,self.NxG))

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

                    self.nt += 1
                        

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
                    T_MEAN = self.T_MEAN,
                ),
                attrs = dict(
                    dt = self.dt,
                    Nt = self.Nt,
                    r_BD = self.r_BD,
                    mu_xi_L = self.mu_xi_L,
                    mu_xi_B = self.mu_xi_B,
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

            return dataset_return
        
    ##########################################################################
    # ADVECTION FUNCTIONS
    ##########################################################################  

    def advection_xi(self):

        # enstrophy and energy conserving advection term from Arakawa 66
        self.adv_n = (-1/(12*self.d**2))*((np.roll(self.psibar_n,1,0) + np.roll(np.roll(self.psibar_n,1,0),-1,1) - np.roll(self.psibar_n,-1,0) - np.roll(np.roll(self.psibar_n,-1,0),-1,1))*(np.roll(self.q_n,-1,1) - self.q_n) + \
                                                (np.roll(np.roll(self.psibar_n,1,0),1,1) + np.roll(self.psibar_n,1,0) - np.roll(np.roll(self.psibar_n,-1,0),1,1) - np.roll(self.psibar_n,-1,0))*(self.q_n - np.roll(self.q_n,1,1)) + \
                                                        (np.roll(self.psibar_n,-1,1) + np.roll(np.roll(self.psibar_n,-1,0),-1,1) - np.roll(self.psibar_n,1,1) - np.roll(np.roll(self.psibar_n,-1,0),1,1))*(np.roll(self.q_n,-1,0) - self.q_n) + \
                                                                (np.roll(np.roll(self.psibar_n,1,0),-1,1) + np.roll(self.psibar_n,-1,1) - np.roll(np.roll(self.psibar_n,1,0),1,1) - np.roll(self.psibar_n,1,1))*(self.q_n - np.roll(self.q_n,1,0)) + \
                                                                        (np.roll(self.psibar_n,-1,1) - np.roll(self.psibar_n,-1,0))*(np.roll(np.roll(self.q_n,-1,0),-1,1) - self.q_n) + \
                                                                                (np.roll(self.psibar_n,1,0) - np.roll(self.psibar_n,1,1))*(self.q_n - np.roll(np.roll(self.q_n,1,0),1,1)) + \
                                                                                        (np.roll(self.psibar_n,-1,0) - np.roll(self.psibar_n,1,1))*(np.roll(np.roll(self.q_n,-1,0),1,1) - self.q_n) + \
                                                                                                (np.roll(self.psibar_n,-1,1) - np.roll(self.psibar_n,1,0))*(self.q_n - np.roll(np.roll(self.q_n,1,0),-1,1)))
        
    def advection_YCXC(self,var,var_return):
        
        setattr(self,var_return,(1/(4*self.d))*\
            ((np.roll(np.roll(self.bathy_np,-1,0),-1,1) + np.roll(self.bathy_np,-1,0))*(np.roll(var,-1,0) + var)*np.roll(self.vbar_n,-1,0) + \
                (np.roll(np.roll(self.bathy_np,-1,0),-1,1) + np.roll(self.bathy_np,-1,1))*(np.roll(var,-1,1) + var)*np.roll(self.ubar_n,-1,1) - \
                    (np.roll(self.bathy_np,-1,1) + self.bathy_np)*(var + np.roll(var,1,0))*self.vbar_n - \
                        (np.roll(self.bathy_np,-1,0) + self.bathy_np)*(var + np.roll(var,1,1))*self.ubar_n))
        
    ##########################################################################
    # DIFFUSION FUNCTIONS
    ##########################################################################   

    def laplacian(self,var,mu,var_return):
        setattr(self,var_return,self.diffusion(var=np.array(var))*mu)

    def biharmonic(self,var,mu,var_return):
        setattr(self,var_return,mu*self.diffusion(var=self.diffusion(var=np.array(var))))

    def diffusion(self,var): 
        return (1/(self.d**2))*(np.roll(var,-1,0) + np.roll(var,-1,1) + \
                                np.roll(var,1,0) + np.roll(var,1,1) - 4*var)
    
    ##########################################################################
    # UV FUNCTIONS
    ##########################################################################  

    def calc_UV(self,psi,u_return,v_return):

        setattr(self,u_return,-(2/self.d)*((np.roll(psi,-1,0) - psi)/self.bathy_YCXG))
        setattr(self,v_return,(2/self.d)*((np.roll(psi,-1,1) - psi)/self.bathy_YGXC))
    

    ##########################################################################
    # ELLIPTIC FUNCTIONS
    ##########################################################################  

    def elliptic(self):

        print('ellip start')

        # create coefficient matrices

        '''C2 = (-1/(self.bathy_np*self.d**2)) + (1/(4*(self.bathy_np**2)*(self.d**2)))*(np.roll(self.bathy_np,-1,1) - np.roll(self.bathy_np,1,1))

        C3 = (-1/(self.bathy_np*self.d**2)) - (1/(4*(self.bathy_np**2)*(self.d**2)))*(np.roll(self.bathy_np,-1,1) - np.roll(self.bathy_np,1,1))

        C4 = (-1/(self.bathy_np*self.d**2)) + (1/(4*(self.bathy_np**2)*(self.d**2)))*(np.roll(self.bathy_np,-1,0) - np.roll(self.bathy_np,1,0))

        C5 = (-1/(self.bathy_np*self.d**2)) - (1/(4*(self.bathy_np**2)*(self.d**2)))*(np.roll(self.bathy_np,-1,0) - np.roll(self.bathy_np,1,0))
        
        C6 = 4/(self.bathy_np*self.d**2)'''

        C2 = -2/((np.roll(self.bathy_np,-1,1) + self.bathy_np)*self.d**2)

        C3 = -2/((self.bathy_np + np.roll(self.bathy_np,1,1))*self.d**2)

        C4 = -2/((np.roll(self.bathy_np,-1,0) + self.bathy_np)*self.d**2)

        C5 = -2/((self.bathy_np + np.roll(self.bathy_np,1,0))*self.d**2)

        C6 = -C2-C3-C4-C5

        offset1 = np.ones((self.NxG))
        offset1[-1] = 0
        offset1 = np.tile(offset1,self.NyG)


        offset2 = np.zeros((self.NxG))
        offset2[0] = 1
        offset2 = np.tile(offset2,self.NyG)

        
        # create diagonals

        diags_C3_1 = sp.sparse.diags(np.roll(C3.flatten(),-1,0)*offset1,offsets=-1,shape=(self.NyG*self.NxG,self.NyG*self.NxG)) 
        diags_C3_2 = sp.sparse.diags(C3.flatten()*offset2,offsets=self.NxG-1,shape=(self.NyG*self.NxG,self.NyG*self.NxG))
        diags_C3 = diags_C3_1 + diags_C3_2

        diags_C2_1 = sp.sparse.diags(C2.flatten()*offset1,offsets=1,shape=(self.NyG*self.NxG,self.NyG*self.NxG))
        diags_C2_2 = sp.sparse.diags(np.roll(C2.flatten(),-self.NxG+1,0)*offset2,offsets=-self.NxG+1,shape=(self.NyG*self.NxG,self.NyG*self.NxG))
        diags_C2 = diags_C2_1 + diags_C2_2

        diags_C5_1 = sp.sparse.diags(np.roll(C5.flatten(),-self.NxG,0),offsets=-self.NxG,shape=(self.NyG*self.NxG,self.NyG*self.NxG)) 
        diags_C5_2 = sp.sparse.diags(C5.flatten(),offsets=self.NxG*self.NxG-self.NxG,shape=(self.NyG*self.NxG,self.NyG*self.NxG)) 
        diags_C5 = diags_C5_1 + diags_C5_2

        diags_C4_1 = sp.sparse.diags(C4.flatten(),offsets=self.NxG,shape=(self.NyG*self.NxG,self.NyG*self.NxG)) 
        diags_C4_2 = sp.sparse.diags(np.roll(C4.flatten(),self.NxG,0),offsets=-self.NxG*self.NxG+self.NxG,shape=(self.NyG*self.NxG,self.NyG*self.NxG)) 
        diags_C4 = diags_C4_1 + diags_C4_2

        diags_C6 = sp.sparse.diags(C6.flatten())

        # total matrix

        matrix = diags_C2 + diags_C3 + diags_C4 + diags_C5 + diags_C6

        self.xi_matrix = matrix.copy()

        # set zero point for uniqueness in a periodic domain

        '''zero_index = self.j_zero*self.NyG + self.i_zero + 1
        print(zero_index)
        matrix = sp.sparse.lil_matrix(matrix)
        matrix[zero_index-1,:] = 0
        matrix[:,zero_index-1] = 0
        matrix[zero_index-1,zero_index-1] = 1'''
        self.matrix = matrix

        print(matrix.todense())

        # create LU solver

        LU = linalg.splu(sp.sparse.csc_matrix(self.matrix))
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
            'enstrophy': self.calc_enstrophy
        }

        self.diagnosticsCoordDict = {
            'xi_u': ['T','YG','XG'],
            'xi_v': ['T','YG','XG'],
            'u_u': ['T', 'YC', 'XG'],
            'v_v': ['T', 'YG', 'XC'],
            'xi_xi': ['T', 'YG', 'XG'],
            'q': ['T', 'YG', 'XG'],
            'q_q': ['T', 'YG', 'XG'],
            'enstrophy': ['T', 'YG', 'XG']
        }

        self.diagnosticsMeanDict = {
            'xi_u': 'xi_u_MEAN',
            'xi_v': 'xi_v_MEAN',
            'u_u': 'u_u_MEAN',
            'v_v': 'v_v_MEAN',
            'xi_xi': 'xi_xi_MEAN',
            'q': 'q_MEAN',
            'q_q': 'q_q_MEAN',
            'enstrophy': 'enstrophy_MEAN'
        }

        self.diagnosticsNP1Dict = {
            'xi_u': 'xi_u_np1',
            'xi_v': 'xi_v_np1',
            'u_u': 'u_u_np1',
            'v_v': 'v_v_np1',
            'xi_xi': 'xi_xi_np1',
            'q': 'q_np1',
            'q_q': 'q_q_np1',
            'enstrophy': 'enstrophy_np1'
        }

        self.diagnosticsSumDict = {
            'xi_u': 'xi_u_sum',
            'xi_v': 'xi_v_sum',
            'u_u': 'u_u_sum',
            'v_v': 'v_v_sum',
            'xi_xi': 'xi_xi_sum',
            'q': 'q_sum',
            'q_q': 'q_q_sum',
            'enstrophy': 'enstrophy_sum'
        }

    def calc_xi_u(self,xi,psi,u,v,var_return):

        setattr(self,var_return,xi*self.interp_y(var=u))

    def calc_xi_v(self,xi,psi,u,v,var_return):

        setattr(self,var_return,xi*self.interp_x(var=v))

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

    ##########################################################################
    # INTERPOLATION FUNCTIONS
    ##########################################################################
        
    def interp_y(self,var):
        return (np.vstack((var,var[0])) + np.vstack((var[-1],var)))/2
    
    def interp_x(self,var):
        return (np.transpose(np.vstack((np.transpose(var),np.transpose(var)[0]))) + \
             np.transpose(np.vstack((np.transpose(var)[-1],np.transpose(var)))))/2
    
    def interp_y(self,var):
        # interpolates var (YCXC or YCXG) in y direction
        # (i.e. to YGXC or YGXG respectively)
        return (var + np.roll(var,1,0))/2
    
    def interp_x(self,var):
        # interpolates var (YCXC or YGXC) in x direction
        # (i.e. to YCXG or YGXG respectively)
        return (var + np.roll(var,1,1))/2
    
    ##########################################################################
    # DIFFERENTIATION FUNCTIONS
    ##########################################################################

    def dx_YCXC(self,var):
        # var defined at YCXC
        # returns d var/dx at YCXG
        return (var - np.roll(var,1,1))/self.dx
    
    def dy_YCXC(self,var):
        # var defined at YCXC
        # returns d var/dy at YGXC
        return (var - np.roll(var,1,0))/self.dy
    
    def dx_YGXG(self,var):
        # var defined at YGXG
        # returns d var/dx at YGXC
        return (np.roll(var,-1,1) - var)/self.dx
    
    def dy_YGXG(self,var):
        # var defined at YGXG
        # returns d var/dy at YCXG
        return (np.roll(var,-1,0) - var)/self.dy
    
    ##########################################################################
    # STACK FUNCTION
    ##########################################################################

    def stack_x(self,var):
        return np.transpose(np.vstack((np.transpose(var)[-1],np.transpose(var),np.transpose(var)[0])))
    
    def stack_y(self,var):
        return np.vstack((var[-1],var,var[0]))

    ##########################################################################
    # PARAMETERIZATION
    ##########################################################################

    def schemeDict(self):
        if self.eddy_scheme != 'EGEC' and self.eddy_scheme != 'EGEC_TEST' and self.eddy_scheme != 'EGECAD' and self.eddy_scheme != 'EGECAD_TEST' \
            and self.eddy_scheme != 'constant' and self.eddy_scheme != 'EGECAD_backscatter' and self.eddy_scheme != False and self.eddy_scheme != 'AdvectionOnly':
            raise ValueError('scheme parameter should be set to \'EGEC\', \'EGECAD\' or False.')

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
        elif self.eddy_scheme == 'A':
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

    def K_Q_AdvectionOnly(self):
        return
    
    def EGEC(self):
        # NOTE: we use the values at time step n here to calculate dQ/dt_n and dK/dt_n so that we can claculate Q_np1 and K_np1

        # interpolate Lambda and K to YGXC and YCXG points. do not calculate at boundaries. 
        self.Q_n_YGXC = self.interp_y(var=self.Q_n) # YGXC
        self.K_n_YGXC = self.interp_y(var=self.K_n) # YGXC

        self.Q_n_YCXG = self.interp_x(var=self.Q_n)  # YCXG 
        self.K_n_YCXG = self.interp_x(var=self.K_n)  # YCXG 

        # calculate dqdx at YGXC points
        self.dqdx_n_YGXC = self.dx_YGXG(var=self.qbar_n) # YGXC
        # calculate dqdy at YCXG points
        self.dqdy_n_YCXG = self.dy_YGXG(var=self.qbar_n) # YCXG
        # 4 point average of dqdx to YCXG points. 
        self.dqdx_n_YCXG = (np.roll(self.dqdx_n_YGXC,-1,0) + np.roll(self.dqdx_n_YGXC,1,1) + \
                            np.roll(np.roll(self.dqdx_n_YGXC,-1,0),1,1) + self.dqdx_n_YGXC)/4 # YCXG
        # 4 point average of dqdy to YGXC points. 
        self.dqdy_n_YGXC = (np.roll(self.dqdy_n_YCXG,-1,1) + np.roll(self.dqdy_n_YCXG,1,0) + \
                            np.roll(np.roll(self.dqdy_n_YCXG,1,0),-1,1) + self.dqdy_n_YCXG)/4 # YGXC
        # calculate mod grad qbar on YGXC
        self.mod_grad_qbar_n_YGXC = np.sqrt(self.dqdx_n_YGXC**2 + self.dqdy_n_YGXC**2) # YGXC 
        # calculate mod grad qbar on YCXG
        self.mod_grad_qbar_n_YCXG = np.sqrt(self.dqdx_n_YCXG**2 + self.dqdy_n_YCXG**2) # YCXG 
        # calculate kappa_q at YGXC and YCXG points. Set to zero on boundaries for zero flux BC. 
        self.kappa_n_YGXC = 2*self.gamma_q*np.sqrt(self.Q_n_YGXC*self.K_n_YGXC)/self.mod_grad_qbar_n_YGXC # YGXC 
        self.kappa_n_YCXG = 2*self.gamma_q*np.sqrt(self.Q_n_YCXG*self.K_n_YCXG)/self.mod_grad_qbar_n_YCXG # YCXG 
        # calculate qu_EDDY = bar{q'u'} at YGXC points
        self.qu_EDDY_n_YGXC = -self.kappa_n_YGXC*self.dqdx_n_YGXC # YGXC 
        # calculate qv_EDDY = bar{q'v'} at YCXG points
        self.qv_EDDY_n_YCXG = -self.kappa_n_YCXG*self.dqdy_n_YCXG # YCXG 
        # calculate bar{q'u'}dqdx at YGXC points
        self.enstrophyGen_n_x_YGXC = self.qu_EDDY_n_YGXC*self.dqdx_n_YGXC # YGXC
        # calculate bqr{q'v'}dqdy at YCXG points
        self.enstrophyGen_n_y_YCXG = self.qv_EDDY_n_YCXG*self.dqdy_n_YCXG # YCXG
        # calculate enstrophy generation at YCXC points 
        self.enstrophyGen_n = (np.roll(self.enstrophyGen_n_x_YGXC,-1,1) + self.enstrophyGen_n_x_YGXC + \
                               np.roll(self.enstrophyGen_n_y_YCXG,-1,0) + self.enstrophyGen_n_y_YCXG)/2 # YCXC

        # calculate dpsidx at YGXC points 
        self.dpsidx_n_YGXC = self.dx_YGXG(var=self.psibar_n) # YGXC
        # calculate dpsidy at YCXG points
        self.dpsidy_n_YCXG = self.dy_YGXG(var=self.psibar_n) # YCXG
        # calculate bar{q'u'}dpsidx at YGXC points
        self.energyConv_n_x_YGXC = self.qu_EDDY_n_YGXC*self.dpsidx_n_YGXC # YGXC
        # calculate bar{q'v'}dpsidy at YCXG points
        self.energyConv_n_y_YCXG = self.qv_EDDY_n_YCXG*self.dpsidy_n_YCXG # YCXG
        # calculate energy conversion at YCXC points 
        self.energyConv_n = (np.roll(self.energyConv_n_x_YGXC,-1,1) + self.energyConv_n_x_YGXC + \
                             np.roll(self.energyConv_n_y_YCXG,-1,0) + self.energyConv_n_y_YCXG)/2 # YCXC

        # calculate kappa_n at YCXC points 
        self.dqdx_n_YCXC = (np.roll(np.roll(self.qbar_n,-1,0),-1,1) + np.roll(self.qbar_n,-1,1) - \
                            np.roll(self.qbar_n,-1,0) - self.qbar_n)/(2*self.dx) # YCXC
        self.dqdy_n_YCXC = (np.roll(np.roll(self.qbar_n,-1,0),-1,1) + np.roll(self.qbar_n,-1,0) - \
                            np.roll(self.qbar_n,-1,1) - self.qbar_n)/(2*self.dy) # YCXC
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
        # calculate laplacian diffusion. 
        self.laplacian(var=self.Q_n,mu=1,var_return='QDiff_L_n')
        # multiply by mu/H
        self.QDiff_L_n = np.array(self.mu_PAR/self.bathy_YCXC)*self.QDiff_L_n

        # diffuse energy
        # calculate laplacian diffusion. 
        self.laplacian(var=self.K_n,mu=1,var_return='KDiff_L_n')
        # multiply by mu/H
        self.KDiff_L_n = np.array(self.mu_PAR/self.bathy_YCXC)*self.KDiff_L_n

        # dQdt and dKdt
        self.Q_F_n = -self.enstrophyGen_n  - self.QAdv_n/self.bathy_YCXC + self.QDiff_L_n - (self.Q_n - self.Q_min_array)*self.r_Q
        self.K_F_n = self.energyConv_n - self.KAdv_n/self.bathy_YCXC + self.KDiff_L_n - (self.K_n - self.K_min_array)*self.r_K

    def K_Q_EGECADB(self):
        self.K_Q_EGECAD()

        # backscatter term
        self.backscatter_integrand_YGXG = self.psibar_n*self.diffusion_B_n # YGXG
        self.backscatter_integrand_YCXC = (np.roll(np.roll(self.backscatter_integrand_YGXG,-1,0),-1,1) + \
                                           np.roll(self.backscatter_integrand_YGXG,-1,0) + \
                                            np.roll(self.backscatter_integrand_YGXG,-1,1) + \
                                                self.backscatter_integrand_YGXG)/4 # YCXC
        # volume integrate
        self.backscatter_integral = np.sum(self.backscatter_integrand_YCXC*self.dx*self.dy) # YCXC
        # volume average
        self.KE_backscatter_n = self.backscatter_integral/self.volume_YCXC # YCXC
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

        # calculate eddy fluxes term
        self.eddyFluxes_n = (1/self.d)*(self.qv_EDDY_n_YCXG*self.bathy_YCXG + self.qu_EDDY_n_YGXC*self.bathy_YGXC - \
                                        np.roll(self.qv_EDDY_n_YCXG*self.bathy_YCXG,1,0) - np.roll(self.qu_EDDY_n_YGXC*self.bathy_YGXC,1,1))
    
    def eddyFluxes_constant(self):
        self.dqdx_n = self.dx_YGXG(var=self.qbar_n) # YGXC
        self.dqdy_n = self.dy_YGXG(var=self.qbar_n) # YCXG

        # ZETA fluxes
        self.flux_u_n = -self.kappa_q*self.dqdx_n # YGXC
        self.flux_v_n = -self.kappa_q*self.dqdy_n # YCXG

        self.eddyFluxes_n = (1/self.d)*(self.flux_v_n*self.bathy_YCXG  + self.flux_u_n*self.bathy_YGXC - \
                                        np.roll(self.flux_v_n*self.bathy_YCXG,1,0) - np.roll(self.flux_u_n*self.bathy_YGXC,1,1))
    
    def noEddyFluxes(self):
        self.eddyFluxes_n = np.zeros_like(self.xibar_n)


#%%

'''L_0 = 1
d = 5000/L_0 # m
Nx = 200
Ny = 200
Lx = d*Nx # m
Ly = d*Ny # m
f0 = 0.7E-4 # s-1
beta = 0*L_0 # m-1s-1
# VORTICITY PARAMETERS
r_BD = 0
mu_xi_B = 1.E8/L_0**4
mu_xi_L = 0/L_0**2
# EDDY SCHEME PARAMETERS
mu_PAR = 500*5000
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
Nt = 960*2
dumpFreq =  86400
meanDumpFreq = 9000
diagnostics = ['xi_u','xi_v','u_u','v_v','xi_xi','q','q_q']

bathy_random = np.load('./../barotropic_model_analysis/model_data/periodic/FDT_5km/randomTopography_5km.npy')/L_0
#bathy_flat = 500*np.ones((Ny+1,Nx+1))

#bathy_random = np.ones((Ny,Nx))

domain = Barotropic(d=d,Nx=Nx,Ny=Ny,bathy=bathy_random,f0=f0,beta=beta)

#%%

init_psi = np.load('./../barotropic_model_analysis/model_data/periodic/FDT_5km/initPsi_5km.npy')/L_0**3

#init_psi = np.zeros((Ny,Nx))

i_zero = int(np.argmin(np.abs(init_psi))%(Ny+1))-1
j_zero = int((np.argmin(np.abs(init_psi))-i_zero)/(Ny+1))
print(i_zero)
print(j_zero)
domain.init_psi(init_psi,i_zero=i_zero,j_zero=j_zero)

#%%
data = domain.model(dt=dt,\
    Nt=Nt,\
        r_BD=r_BD,\
            mu_xi_L=mu_xi_L,\
                mu_xi_B=mu_xi_B,\
                    tau_0=tau_0,\
                        rho_0=rho_0,\
                            dumpFreq=dumpFreq,\
                                meanDumpFreq=meanDumpFreq)#,\
                                    #diags=diagnostics)


# %%
t = -1
plt.contourf(data.XG,data.YG,data.psi[t])
plt.colorbar()
plt.show()


#%%
t = -1
plt.contourf(data.XG,data.YG,data.xi[t])
plt.colorbar()
plt.show()

#%%
t = 960
plt.contourf(data.XG,data.YG,domain.advection[t])
plt.colorbar()
plt.show()

# %%
plt.contourf(data.XG,data.YG,domain.diffusion_B_n)
plt.colorbar()
plt.show()

#%%
plt.contourf(data.XG,data.YG,domain.adv_n)
plt.colorbar()
plt.show()

#%%
plt.contourf(data.XG,data.YG,domain.BD_n)
plt.colorbar()
plt.show()


# %%
xi = np.array(data.xi)

t = 0

print(np.shape(np.vstack((xi[0,-1],xi[0]))))

xi_periodic = np.transpose(np.vstack((np.transpose(np.vstack((xi[0,-1],xi[0]))),np.transpose(np.vstack((xi[0,-1],xi[0])))[0])))

print(np.array_equal(xi_periodic[0],xi_periodic[-1]))
print(np.array_equal(xi_periodic[:,0],xi_periodic[:,-1]))

plt.contourf(np.arange(Nx+1),np.arange(Ny+1),xi_periodic)
plt.colorbar()
plt.show()

xi_integral = np.sum(xi_periodic*data.dx*data.dy)
print(xi_integral)

v = np.ones(domain.xi_matrix.shape[1])
print(domain.xi_matrix.sum(0))
print(domain.xi_matrix @ v)
print(xi_integral)
print(domain.xi_matrix.getrow(1))
# %%

print(np.sum(domain.matrix.todense()))

# %%

print(np.abs(np.sum(domain.xi_matrix.todense(),0)).max())

# %%
print(np.abs(np.sum(domain.xi_matrix,0)).max())
print(np.abs(np.sum(domain.xi_matrix,1)).max())
# %%
print(np.abs(domain.xi_matrix - np.transpose(domain.xi_matrix)).max())'''
# %%
