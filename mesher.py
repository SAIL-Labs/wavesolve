import pygmsh
import numpy as np
import matplotlib.pyplot as plt

def circ_points(radius,res):
    thetas = np.linspace(0,2*np.pi,res,endpoint=False)
    points = []
    for t in thetas:
        points.append((radius*np.cos(t),radius*np.sin(t)))
    return points

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

def construct_mesh(w,r,res,ret="tuple"):
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
        if ret == "tuple":
            return (points,tris)
        else:
            return mesh

def construct_mesh2(w,r,res,ret="tuple"):
    ''' construct triangular mesh for FE analysis. a mesh is a tuple of points and connections.
        points is a 2D numpy array of all nodes in the mesh (whose positions are length 2 numpy arrays)
        each element of connections is a list containing the 6 indices that locate the nodes in each tri

        w: width of comp zone (assumed square)
        r: core radius (need to make this more extensible beyond circular step-index later)
    '''
    with pygmsh.occ.Geometry() as geom:
        circ0 = geom.add_polygon(circ_points(w/2,int(res/2)))
        circ = geom.add_disk((0,0),r,mesh_size=w/res)

        geom.boolean_difference(circ0,circ,delete_other = False)
        mesh = geom.generate_mesh(dim=2,order=2,algorithm=6)
        points = mesh.points
        tris = mesh.cells[1].data
        if ret == "tuple":
            return (points,tris)
        else:
            return mesh

m = construct_mesh2(2,0.5,40)
plot_mesh(m)