""" use finite element method with quadratic triangular elements to solve for modes in the SCALAR approximation. 
    if this works, perhaps vector forthcoming. also, maybe fe-bpm?
"""
import numpy as np
import matplotlib.pyplot as plt
import pygmsh
from scipy.linalg import eigh

def circ_points(radius,res):
    thetas = np.linspace(0,2*np.pi,res,endpoint=False)
    points = []
    for t in thetas:
        points.append((radius*np.cos(t),radius*np.sin(t)))
    return points

def IOR_fiber(r,n,n0):
    def inner(x,y):
        if x*x+y*y <= r*r:
            return n
        return n0
    return inner

def construct_mesh(w,r,res):
    ''' construct triangular mesh for FE analysis. a mesh is a tuple of points and connections.
        points is a 2D numpy array of all nodes in the mesh (whose positions are length 2 numpy arrays)
        each element of connections is a list containing the 6 indices that locate the nodes in each tri

        w: width of comp zone (assumed square)
        r: core radius (need to make this more extensible beyond circular step-index later)
    '''
    with pygmsh.occ.Geometry() as geom:
        rect = geom.add_rectangle((-w/2,-w/2,0),w,w) 
        circ = geom.add_disk((0,0),r,mesh_size=w/res)
        geom.boolean_difference(rect,circ,delete_other = False)
        mesh = geom.generate_mesh(dim=2,order=2)
        points = mesh.points
        tris = mesh.cells[1].data
        return (points,tris)

def construct_mesh_circ(w,r,res):
    ''' construct triangular mesh for FE analysis. a mesh is a tuple of points and connections. outer bound 
        is a circle. points is a 2D numpy array of all nodes in the mesh (whose positions are length 2 numpy arrays)
        each element of connections is a list containing the 6 indices that locate the nodes in each tri

        w: width of comp zone (assumed square)
        r: core radius (need to make this more extensible beyond circular step-index later)
    '''
    with pygmsh.occ.Geometry() as geom:
        circ0 = geom.add_polygon(circ_points(w/2,int(res/2)))
        circ = geom.add_disk((0,0),r,mesh_size=w/res)
        geom.boolean_difference(circ0,circ,delete_other = False)
        mesh = geom.generate_mesh(dim=2,order=2)
        points = mesh.points
        tris = mesh.cells[1].data
        return (points,tris)

def plot_mesh(mesh,IOR_arr = None,show=True):
    points,tris = mesh

    if show:
        fig,ax = plt.subplots(figsize=(5,5))

    for point in points:
        plt.plot(point[0],point[1],'ko',ms=2)

    cm = plt.get_cmap("viridis")
    for i,_e in enumerate(tris):
        e = _e[:3]
        
        if IOR_arr is not None:
            n,n0 = np.max(IOR_arr),np.min(IOR_arr)
            cval = IOR_arr[i]/(n-n0) - n0/(n-n0)
            color = cm(cval)
            t=plt.Polygon(points[e][:,:2], ec='k', fc=color,lw=0.7)
        else:
            t=plt.Polygon(points[e][:,:2], ec='k', fc='None',lw=0.7)
        plt.gca().add_patch(t)

    plt.xlim(np.min(points[:,0]),np.max(points[:,0]) )
    plt.ylim(np.min(points[:,1]),np.max(points[:,1]) )
    if show:
        plt.show()

def discretize_IOR(IOR_func,mesh):
    ''' discretize an input function IOR(x,y) on the faces of a mesh'''
    out = []
    points,tris = mesh
    for tri in tris:
        c = np.mean(points[tri[:3]],axis=0)
        out.append(IOR_func(c[0],c[1]))
    return np.array(out)

def construct_AB(mesh,IOR_arr,k,IOR_func = None):
    from shape_funcs import compute_dNdN, compute_NN
    points,qtris = mesh

    N = len(points)

    A = np.zeros((N,N))
    B = np.zeros((N,N))

    for tri,IOR in zip(qtris,IOR_arr):
        tri_points = points[tri]
        NN = compute_NN(tri_points)
        dNdN = compute_dNdN(tri_points)
        #NN,dNdN = compute_matrices_from_file(tri_points)
        
        ix = np.ix_(tri,tri)
        
        if IOR_func is not None:
            IORs_alt = np.array([IOR_func(*points[i][:2]) for i in tri])
            A[ix] += k**2*NN * np.power(IORs_alt[None,:],2) - dNdN
        else:
            A[ix] += k**2*IOR**2 * NN - dNdN
        
        B[ix] += NN
    
    return A,B

def solve(A,B,mesh,k,ncore,nclad,plot=False):
    w,v = eigh(A,B)
    points,tris = mesh
    if plot:
        for _w,_v in zip(w[::-1],v.T[::-1]):
            ne = np.sqrt(_w/k**2)
            if not (nclad <= ne <= ncore):
                print("warning: spurious mode! ")
            
            print("effective index: ",np.sqrt(_w/k**2))
            plot_eigenvector(mesh,_v)

def plot_eigenvector(mesh,v):
    plt.figure(figsize=(8,8))
    plt.axis('equal')
    plt.tricontourf(points[:,0],points[:,1],v,levels=40)
    plt.colorbar()
    plot_mesh(mesh,show=False)
    plt.show()

ncore = 1.444
nclad = 1.444-5.5e-3
k = 2*np.pi/1.55
res = 40

r = 10
w = 4*r
IOR_func = IOR_fiber(r,ncore,nclad)

mesh = construct_mesh_circ(w,r,res)

points,tris = mesh

IOR_arr = discretize_IOR(IOR_func,mesh)

print("mesh and refractive index distribution")
plot_mesh(mesh,IOR_arr=IOR_arr)

A,B = construct_AB(mesh,IOR_arr,k)

solve(A,B,mesh,k,ncore,nclad,plot=True)
