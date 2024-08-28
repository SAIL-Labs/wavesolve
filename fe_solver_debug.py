# %%
import numpy as np
from wavesolve.fe_solver import solve_waveguide
from wavesolve import waveguide

rcore = 5
rclad = 10
ncore = 1.444+8.8e-3
nclad = 1.444

# 3-mode fiber
circular_fiber = waveguide.CircularFiber(rcore,rclad,ncore,nclad,64,128,core_mesh_size=0.5)
wl = 1.55 #um
IOR_dict = circular_fiber.assign_IOR()

circular_fiber.mesh_dist_scale = 0.25
mesh = circular_fiber.make_mesh(order=2,adaptive=True)
tris = mesh.cells[1].data[tuple(mesh.cell_sets["cladding"])][0,:,0,:]
solve_waveguide(mesh,wl,IOR_dict,plot=False,Nmax=3)

# %%
