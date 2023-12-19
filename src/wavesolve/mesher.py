import pygmsh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree  

def circ_points(radius,res,center=(0,0)):
    """generates a list of points defining a circle
    Args:
    radius: circle radius
    res: number of points to generate
    center: (x,y) coords of circle center

    Returns:
    points: (res, 2) list of circle points
    """
     
    thetas = np.linspace(0,2*np.pi,res,endpoint=False)
    points = []
    for t in thetas:
        points.append((radius*np.cos(t)+center[0],radius*np.sin(t)+center[1]))
    return points

def plot_mesh(mesh,IOR_dict=None,show=True,ax=None,verts=3,alpha=0.2):
    """ plot a mesh and associated refractive index distribution
    Args:
    mesh: the mesh to be plotted
    IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
    show: set True to plot directly
    ax: optionally, provide matplotlib axis object for plotting
    """

    points = mesh.points
    els = mesh.cells[1].data
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
        _els = els[tuple(mesh.cell_sets[material])][0,:,0,:]
        for i,_el in enumerate(_els):
            t=plt.Polygon(points[_el[:verts]][:,:2], facecolor=color)
            t_edge=plt.Polygon(points[_el[:verts]][:,:2], lw=0.5,color='white',alpha=alpha,fill=False)
            ax.add_patch(t)
            ax.add_patch(t_edge)

    for point in points:
        ax.plot(point[0],point[1],'wo',ms=1,alpha=alpha)

    ax.set_xlim(np.min(points[:,0]),np.max(points[:,0]) )
    ax.set_ylim(np.min(points[:,1]),np.max(points[:,1]) )
    if show:
        plt.show()

def plot_mesh_expl_IOR(mesh,IORs,show=True,ax=None,verts=3,alpha=0.2):
    points = mesh.points
    els = mesh.cells[1].data

    n,n0 = max(IORs) , min(IORs)

    if show and ax is None:
        fig,ax = plt.subplots(figsize=(5,5))

    for i,el in enumerate(els):
  
        cval = IORs[i]/(n-n0) - n0/(n-n0)
        cm = plt.get_cmap("viridis")
        color = cm(cval)

        t=plt.Polygon(points[el[:verts]][:,:2], facecolor=color)
        t_edge=plt.Polygon(points[el[:verts]][:,:2], lw=0.5,color='white',alpha=alpha,fill=False)
        ax.add_patch(t)
        ax.add_patch(t_edge)

    for point in points:
        plt.plot(point[0],point[1],'wo',ms=1,alpha=alpha)

    plt.xlim(np.min(points[:,0]),np.max(points[:,0]) )
    plt.ylim(np.min(points[:,1]),np.max(points[:,1]) )
    if show:
        plt.show()

def fiber_mesh(r_clad,r_core,res,mode="tri"):
    with pygmsh.occ.Geometry() as geom:
        cladding_base = geom.add_polygon(circ_points(r_clad,int(res/2)))
        core = geom.add_polygon(circ_points(r_core,int(res)))

        cladding = geom.boolean_difference(cladding_base,core,delete_other = False)
        geom.add_physical(cladding,"cladding")
        geom.add_physical(core,"core")

        algo = 6
        if mode=="quad":
            algo = 11
        mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
        mesh.cell_data["radius"] = r_core
        return mesh

def remove_face_points(mesh):
    """ converts quad9 to quad8 """
    new_vertex_indices = np.array(sorted(list(set(mesh.cells[1].data[:,:-1].flatten()))))
    new_points = mesh.points[new_vertex_indices]
    deleted_indices = mesh.cells[1].data[:,-1]
    mesh.cells[1].data = mesh.cells[1].data[:,:-1]

    for quad in mesh.cells[1].data:
        for i,idx in enumerate(quad):
            _shift = np.sum(idx>=deleted_indices)
            quad[i] -= _shift
    
    mesh.points = new_points
    return mesh

def lantern_mesh_displaced_circles(r_jack,r_clad,pos_core,r_core,res,ds=0.1):
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

        cores = [[],[],[],[],[],[],[]]
        cores_flat = []
        for r,p in zip(rs,ps):
            #core = geom.add_disk(p,r)
            cpoints = circ_points(r_core,res=16,center=p)
            core = geom.add_polygon(cpoints)
            #cores.append(core)

            # add x and y displacements
            px = (p[0]+ds,p[1])
            cxpoints = circ_points(r_core,res=16,center=px)
            corex = geom.add_polygon(cxpoints)
            #cores.append(corex)
            py = (p[0],p[1]+ds)
            cypoints = circ_points(r_core,res=16,center=py)
            corey = geom.add_polygon(cypoints)

            core_pieces = geom.boolean_fragments([core],[corex,corey])
            
            for i,c in enumerate(core_pieces):
                cores[i].append(c)
            
            cores_flat += core_pieces

            #cores += [core_sub,intx,corex_sub]

        for i,c in enumerate(cores):
            geom.add_physical(c,"core"+str(i))

        cladding = geom.boolean_difference(cladding_base,cores_flat,delete_other=False)
        geom.add_physical(cladding,"cladding")
        algo = 6
        mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
        mesh.cell_data["radius"] = r_clad
        return mesh


def lantern_mesh_3PL(r,res):
    """ generates a mesh for a 3-port lantern with a non-circular cladding. 
        similar to the 3-port PL on SCExAO. """
    with pygmsh.occ.Geometry() as geom:
        jacket_base = geom.add_polygon(circ_points(r*4,int(2*res))) # from microscope image
        center_offset = r*np.sqrt(3)/3
        centers = [[center_offset,0],[-center_offset/2,center_offset*np.sqrt(3)/2],[-center_offset/2,-center_offset*np.sqrt(3)/2]]
        clad0 = geom.add_polygon(circ_points(r,res,center=centers[0]))
        clad1 = geom.add_polygon(circ_points(r,res,center=centers[1]))
        clad2 = geom.add_polygon(circ_points(r,res,center=centers[2]))
        full_clad = geom.boolean_union([clad0,clad1,clad2])

        jacket = geom.boolean_difference(jacket_base,full_clad,delete_other = False)
        geom.add_physical(jacket,"jacket")
        geom.add_physical(full_clad,"cladding")
        algo = 6
        mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
        return mesh

def lantern_mesh_6PL(r,res):
    """ generates a mesh for a 6-port lantern with a circular cladding. 
        tapered-down single-mode cores are also included. """
    with pygmsh.occ.Geometry() as geom:
        jacket_base = geom.add_polygon(circ_points(r*4,int(2*res))) 
        rcore = r*9.2/255
        center_offset = r*140/255
        clad_base = geom.add_polygon(circ_points(r,res))
        cores = []
        cores.append(geom.add_polygon(circ_points(rcore,6)))
        for i in range(5):
            c = geom.add_polygon(circ_points(rcore,6,center=(center_offset*np.cos(2*np.pi/5*i),center_offset*np.sin(2*np.pi/5*i))))
            cores.append(c)
        jacket = geom.boolean_difference(jacket_base,clad_base,delete_other=False)[0]

        core_circs = []
        core_circs.append(geom.add_polygon(circ_points(rcore*4,12)))
        for i in range(5):
            c = geom.add_polygon(circ_points(rcore*6,12,center=(center_offset*np.cos(2*np.pi/5*i),center_offset*np.sin(2*np.pi/5*i))))
            core_circs.append(c)
        
        for c in core_circs:
            clad_base = geom.boolean_difference(clad_base,c,delete_other=False)[0]

        for i in range(6):
            core_circs[i] = geom.boolean_difference(core_circs[i],cores[i],delete_other=False)[0]
        claddings = [clad_base]+core_circs
        geom.add_physical(jacket,"jacket")
        geom.add_physical(claddings,"cladding")
        geom.add_physical(cores,"core")
        algo = 6
        mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
        return mesh        

def construct_meshtree(mesh):
    """ compute a KDtree from mesh triangle centroids """

    tris = mesh.cells[1].data
    cntrs = np.mean(mesh.points[tris][:,:3,:2],axis=1)
    return KDTree(cntrs)