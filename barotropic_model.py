#%%

import numpy as np

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
        self.f = [(f0+(i*beta*self.dy))*np.ones(Nx+1) for i in range(self.Ny+1)]

    def model(self,dt,Nt,init_psi,gamma_q,r):

        # store values in object

        self.dt = dt
        self.Nt = Nt
        self.init_psi = init_psi 
        self.gamma_q = gamma_q
        self.r = r

        # init_psi must have shape Ny+1 * Nx+1
        #calculate xibar from init_psi    
        xibar_n = 0
        # create xibar_new array
        xibar_np1 = np.zeros_like(xibar_n)

        # need to store F_n, F_n-1 and F_n-2 to calculate xibar_new with AB3
        F_n = np.zeros_like(xibar_n)
        F_nm1 = np.zeros_like(xibar_n)
        F_nm2 = np.zeros_like(xibar_n)

        psibar_n = init_psi
        zetabar_n = 0
        xibar_dx_n = 0
        xibar_dy_n = 0

        # first two time steps with forward Euler

        for t in range(2):

            # calculate values needed to step forward xiba
            # advection term
            adv_n = self.advection(zetabar_n,psibar_n)

            # flux term

            # dissipation 
            D_n = r*xibar_n

            # F_n
            F_n = -adv_n - div_flux_n - D_n

            # forward Euler
            xibar_np1 = xibar_n + dt*F_n

            # SOLVE FOR PSIBAR

            # DUMP XIBAR AND PSIBAR AT CERTAIN INTERVALS

            # reset values
            F_nm2 = F_nm1.copy()
            F_nm1 = F_n.copy()
            F_n = np.zeros_like(F_n)
            xibar_n = xibar_np1.copy()
            xibar_np1 = np.zeros_like(xibar_np1)

        # rest of the time steps with AB3
        for t in range(2,Nt): 

            # calculate values needed to step forward xibar
            # advection term
            adv_n = self.advection(zetabar_n,psibar_n)

            # flux term

            # dissipation 
            D_n = r*xibar_n

            # F_n
            F_n = -adv_n - div_flux_n - D_n
            
            # AB3
            xibar_np1 = xibar_n + (dt/12)*(23*F_n - 16*F_nm1 + 5*F_nm2)

            # SOLVE FOR PSIBAR

            # DUMP XIBAR AND PSIBAR AT CERTAIN INTERVALS

            # reset values
            F_nm2 = F_nm1.copy()
            F_nm1 = F_n.copy()
            F_n = np.zeros_like(F_n)
            xibar_n = xibar_np1.copy()
            xibar_np1 = np.zeros_like(xibar_np1)


    def advection(self,zetabar,psibar):

        # psibar and zetabar have coordinates [y,x]

        J = np.zeros((self.Ny,self.Nx))

        for i in range(self.Nx):
            for j in range(self.Ny):

                J[j,i] = (1/(12*self.d**2))*(psibar[j+1,i]*(3*zetabar[j,i+1] - 3*zetabar[j,i-1] + zetabar[j+1,i+1] - zetabar[j-1,i-1]) + \

                            psibar[j-1,i]*(-3*zetabar[j,i+1] + 3*zetabar[j,i-1]) + \

                            psibar[j,i+1]*(-3*zetabar[j+1,i] + 3*zetabar[j-1,i] - zetabar[j+1,i+1] + zetabar[j-1,i-1]) + \

                            psibar[j,i-1]*(3*zetabar[j+1,i] - 3*zetabar[j-1,i]) + \

                            psibar[j+1,i+1]*(zetabar[j,i+1] - zetabar[j+1,i]) + \

                            psibar[j-1,i-1]*(zetabar[j,i+1] - zetabar[j+1,i])
                
                )
        
        return J



#%%

test = Barotropic(d=5000,Nx=200,Ny=200,bathy=np.zeros((200,200)),f0=0.7E-4,beta=2E-11)
# %%

print(test.f[199])
# %%
