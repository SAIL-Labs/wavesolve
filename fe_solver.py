""" use finite element method with quadratic triangular elements to solve for modes in the SCALAR approximation. 
    if this works, perhaps vector forthcoming. also, maybe fe-bpm?
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from mesher import plot_mesh,lantern_mesh,circ_points

def IOR_fiber(r,n,n0):
    def inner(x,y):
        if x*x+y*y <= r*r:
            return n
        return n0
    return inner

def construct_AB(mesh,IOR_dict,k):
    from shape_funcs import compute_dNdN, compute_NN
    
    points = mesh.points
    materials = mesh.cell_sets.keys()

    N = len(points)

    A = np.zeros((N,N))
    B = np.zeros((N,N))

    for material in materials:
        tris = mesh.cells[1].data[tuple(mesh.cell_sets[material])][0,:,0,:]
        for tri in tris:
            tri_points = points[tri]
            NN = compute_NN(tri_points)
            dNdN = compute_dNdN(tri_points)
            ix = np.ix_(tri,tri)
            A[ix] += k**2*IOR_dict[material]**2 * NN - dNdN
            B[ix] += NN

    return A,B

def solve(A,B,mesh,k,IOR_dict,plot=False):
    w,v = eigh(A,B)

    IORs = [ior[1] for ior in IOR_dict.items()]
    nmin,nmax = min(IORs) , max(IORs)

    if plot:
        for _w,_v in zip(w[::-1],v.T[::-1]):
            ne = np.sqrt(_w/k**2)
            if not (nmin <= ne <= nmax):
                print("warning: spurious mode! ")
            
            print("effective index: ",np.sqrt(_w/k**2))
            plot_eigenvector(mesh,_v)

def plot_eigenvector(mesh,v,plot_mesh = False):
    points = mesh.points
    fig,ax = plt.subplots(figsize=(5,5))
    plt.axis('equal')
    plt.tricontourf(points[:,0],points[:,1],v,levels=40)
    plt.colorbar()
    if plot_mesh:
        plot_mesh(mesh,show=False,ax=ax)
    try:
        circle = plt.Circle((0,0),mesh.cell_data['radius'],ec='white',fc='None',lw=2)
        ax.add_patch(circle)
    except:
        pass
    plt.show()

ncore = 1.444 + 0.01036
nclad = 1.444
njack = 1.444 - 5.5e-3

IOR_dict = {"core":ncore,"cladding":nclad,"jacket":njack}

k = 2*np.pi/1.55
res = 40

r = 10
w = 4*r

cores = [(0,0)] + circ_points(2*r/3,5)
mesh = lantern_mesh(w/2,r,cores,r/10,res)

print("mesh and refractive index distribution")
plot_mesh(mesh,IOR_dict=IOR_dict)

A,B = construct_AB(mesh,IOR_dict,k)

solve(A,B,mesh,k,IOR_dict,plot=True)
