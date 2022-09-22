import pygmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

def circ_points(radius,res,center=(0,0)):
    thetas = np.linspace(0,2*np.pi,res,endpoint=False)
    points = []
    for t in thetas:
        points.append((radius*np.cos(t)+center[0],radius*np.sin(t)+center[1]))
    return points

def plot_mesh(mesh,IOR_dict=None,show=True,ax=None):
    points = mesh.points
    tris = mesh.cells[1].data
    materials = mesh.cell_sets.keys()

    if IOR_dict is not None:
        IORs = [ior[1] for ior in IOR_dict.items()]
        n,n0 = max(IORs) , min(IORs)

    if show and ax is None:
        fig,ax = plt.subplots(figsize=(5,5))

    for material in materials:
        if IOR_dict is not None:        
            cval = IOR_dict[material]/(n-n0) - n0/(n-n0)
            cm = plt.get_cmap("viridis")
            color = cm(cval)
        else:
            color = "None"
        _tris = tris[tuple(mesh.cell_sets[material])][0,:,0,:]
        patches = []
        for i,_tri in enumerate(_tris):
            t=plt.Polygon(points[_tri[:3]][:,:2], ec='k', facecolor=color,lw=0.7)
            patches.append(t)
            ax.add_patch(t)

    for point in points:
        plt.plot(point[0],point[1],'ko',ms=2)

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
        circ = geom.add_polygon(circ_points(r,res))

        #cladding = geom.boolean_difference(circ0,circ,)

        union = geom.boolean_union([circ0,circ])
        geom.boolean_fragments(union,circ)
        mesh = geom.generate_mesh(dim=2,order=2,algorithm=6)
        points = mesh.points
        tris = mesh.cells[1].data
        if ret == "tuple":
            return (points,tris)
        else:
            return mesh

def lantern_mesh(r_jack,r_clad,pos_core,r_core,res):
    """ construct a mesh that conforms to a 'lantern' structure: circular jacker,
    smaller circular cladding, and even smaller circular inclusions (cores) at
    arbitrary locations """

    with pygmsh.occ.Geometry() as geom:
        jacket_base = geom.add_polygon(circ_points(r_jack,int(res/2)))
        cladding_base = geom.add_polygon(circ_points(r_clad,res))

        jacket = geom.boolean_difference(jacket_base,cladding_base,delete_other = False)
        geom.add_physical(jacket,"jacket")

        if type(r_core) != list and type(pos_core) == list:
            rs = [r_core] * len(pos_core)
            ps = pos_core
        elif type(r_core) != list and type(pos_core) != list:
            rs = [r_core]
            ps = [pos_core]

        cores = []
        for r,p in zip(rs,ps):
            core = geom.add_disk(p,r,mesh_size = r)
            cores.append(core)

        geom.add_physical(cores,"core")
        cladding = geom.boolean_difference(cladding_base,cores,delete_other=False)
        geom.add_physical(cladding,"cladding")
        mesh = geom.generate_mesh(dim=2,order=2,algorithm=6)
        mesh.cell_data["radius"] = r_clad
        return mesh

if __name__ == "__main__":
    cores = [(0,0)] + circ_points(0.25,5)
    m = lantern_mesh(1,0.5,cores,0.05,30)
    #m = construct_mesh2(2,0.5,30)

    IOR_dict = {"jacket":2,"cladding":1.5,"core":1}
    
    plot_mesh(m,IOR_dict)