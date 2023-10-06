import pygmsh
import numpy as np
import matplotlib.pyplot as plt

def circ_points(radius,res,center=(0,0),petals=0,petal_amp=0.1):
    thetas = np.linspace(0,2*np.pi,res,endpoint=False)
    points = []
    for t in thetas:
        offset = 0 if petals == 0 else np.abs(np.cos(petals/2*t))*radius*petal_amp - 2/np.pi*radius*petal_amp #compensation so avg radius is same
        points.append(((radius+offset)*np.cos(t)+center[0],(radius+offset)*np.sin(t)+center[1]))
    return points

def plot_mesh(mesh,IOR_dict=None,show=True,ax=None,verts=3,alpha=0.2):
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
        plt.plot(point[0],point[1],'wo',ms=1,alpha=alpha)

    plt.xlim(np.min(points[:,0]),np.max(points[:,0]) )
    plt.ylim(np.min(points[:,1]),np.max(points[:,1]) )
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

def lantern_mesh(r_jack,r_clad,pos_core,r_core,res,petals=0,petal_amp = 0.1,mode="tri"):
    """ construct a mesh that conforms to a 'lantern' structure: circular jacker,
    smaller circular cladding, and even smaller circular inclusions (cores) at
    arbitrary locations """

    with pygmsh.occ.Geometry() as geom:
        jacket_base = geom.add_polygon(circ_points(r_jack,int(res/2)))
        cladding_base = geom.add_polygon(circ_points(r_clad,res,petals=petals,petal_amp=petal_amp))

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
        algo = 6
        if mode=="quad":
            algo = 11
        mesh = geom.generate_mesh(dim=2,order=2,algorithm=algo)
        mesh.cell_data["radius"] = r_clad
        return mesh

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

    with pygmsh.occ.Geometry() as geom:
        jacket_base = geom.add_polygon(circ_points(r*8,int(2*res))) # from microscope image

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

if __name__ == "__main__":
    #cores = [(0,0)] + circ_points(0.25,5)
    #m = lantern_mesh_3PL(1,16) #lantern_mesh_displaced_circles(1,0.5,cores,0.5/8,40,ds=0.05)
    #m = remove_face_points(m)
    #m = construct_mesh2(2,0.5,30)

    #IOR_dict = {"jacket":2.5,"cladding":2,"core0":1,"core1":1,"core2":1,"core3":1,"core4":1.4,"core5":1.5,"core6":1.6}
    IOR_dict = {"jacket":1.444-4e-3,"cladding":1.444,"core":1.4504 }
    m = lantern_mesh_6PL(6,16)
    plot_mesh(m,IOR_dict,verts=3)