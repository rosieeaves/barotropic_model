#%% 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 

#%%  

dataset_1 = xr.load_dataset('./model_data/FDT_MOUNT_2km_516days')
dataset_2 = xr.load_dataset('./model_data/FDT_MOUNT_2km_516-1000days')

#%%
print(dataset_1)

#%%
print(dataset_2)


# %%

dataset_complete = xr.Dataset()
dataset_complete = dataset_complete.assign_coords(dataset_1.coords)
# %%
stitched_vars = ['xi','psi','u','v','xi_u','xi_v','u_u','v_v']

# %%
print(xr.concat([dataset_1.xi,dataset_2.xi[1:]],dim='T'))

# %%
dataset_complete['T'] = np.append(dataset_1.T,dataset_2.T[1:])
dataset_complete['T_MEAN'] = np.arange(1,len(np.append(dataset_1.T_MEAN,dataset_2.T_MEAN[1:])))
# %%
print(dataset_complete)

# %%
print(len(dataset_1.T_MEAN))
print(len(dataset_2.T_MEAN))

# %%
