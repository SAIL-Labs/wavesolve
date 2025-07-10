import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import meshio
import math

#region miscellaneous functions   

def get_unique_edges(mesh,mutate=True):
    """ get set of unique edges in mesh. when mutate = True, this function adds a connectivity table for edges to the mesh;
        this table is not included in the base pygmsh/gmsh meshes, but is needed for vectorial FEM. """
    tris = mesh.cells[1].data
    
    unique_edges = list()
    edge_indices = np.zeros((tris.shape[0],3),dtype=np.uint)
    edge_flips = np.ones((tris.shape[0],3))

    i = 0
    for j,tri in enumerate(tris):
        e0 = sorted([tri[0],tri[1]])
        e1 = sorted([tri[1],tri[2]])
        e2 = sorted([tri[2],tri[0]])
        es = [e0,e1,e2]
        es_unsort = [[tri[0],tri[1]],[tri[1],tri[2]],[tri[2],tri[0]]]
        for k,e in enumerate(es):
            if e not in unique_edges:
                unique_edges.append(e)
                edge_indices[j,k] = i
                i += 1
            else:
                edge_indices[j,k] = unique_edges.index(e)
            if e != es_unsort[k]:
                edge_flips[j,k] = -1

    out = np.array(unique_edges)
    if mutate:
        mesh.cells[0].data = out
        mesh.edge_indices = edge_indices
        mesh.edge_flips = edge_flips
        mesh.num_edges = len(out)
        mesh.num_points = mesh.points.shape[0]
    return out,edge_indices

def plot_mesh(mesh,IOR_dict=None,alpha=0.3,ax=None,plot_points=True):
    """ plot a mesh and associated refractive index distribution
    Args:
    mesh: the mesh to be plotted. if None, we auto-compute a mesh using default values
    IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
    """
    show=False
    verts=3
    if ax is None:
        fig,ax = plt.subplots(figsize=(5,5))
        show=True

    ax.set_aspect('equal')

    points = mesh.points
    els = mesh.cells[1].data
    materials = mesh.cell_sets.keys()

    if IOR_dict is not None:
        IORs = [ior[1] for ior in IOR_dict.items()]
        n,n0 = max(IORs) , min(IORs)

    for material in materials:   
        if IOR_dict is not None:    
            cval = IOR_dict[material]/(n-n0) - n0/(n-n0)
            cm = plt.get_cmap("inferno")
            color = cm(cval)
        else:
            color="None"

        _els = els[tuple(mesh.cell_sets[material])][0,:,0,:]
        for i,_el in enumerate(_els):
            t=plt.Polygon(points[_el[:verts]][:,:2], facecolor=color)
            t_edge=plt.Polygon(points[_el[:verts]][:,:2], lw=0.5,color='0.5',alpha=alpha,fill=False)
            ax.add_patch(t)
            ax.add_patch(t_edge)

    ax.set_xlim(np.min(points[:,0]),np.max(points[:,0]) )
    ax.set_ylim(np.min(points[:,1]),np.max(points[:,1]) )
    if plot_points:
        for point in points:
            ax.plot(point[0],point[1],color='0.5',marker='o',ms=1.5,alpha=alpha)

    if show:
        plt.show()

def load_meshio_mesh(meshname):
    mesh = meshio.read(meshname+".msh")
    keys = list(mesh.cell_sets.keys())[:-1] # ignore bounding entities for now
    _dict = {}
    for key in keys:
        _dict[key] = []

    cells1data = []
    for i,c in enumerate(mesh.cells): # these should all be triangle6
        for key in keys:
            if len(mesh.cell_sets[key][i]) != 0:
                triangle_indices = mesh.cell_sets[key][i]
                Ntris = len(triangle_indices)
                totaltris = len(cells1data)
                cons = mesh.cells[i].data[triangle_indices]
                if len(cells1data) != 0:
                    cells1data = np.concatenate([cells1data,cons])
                else:
                    cells1data = cons
                if len(_dict[key]) != 0:
                    _dict[key] = np.concatenate([_dict[key],np.arange(totaltris,totaltris+Ntris)])
                else:
                    _dict[key] = np.arange(totaltris,totaltris+Ntris)
                continue
    
    # this is to match the format made by pygmsh
    for key,val in _dict.items():
        _dict[key] = [None,val,None]

    mesh.cell_sets=_dict
    mesh.cells[1].data = cells1data
    for i in range(len(mesh.cells)):
        if i == 1:
            continue
        mesh.cells[i]=None
    return mesh

def boolean_fragment(geom:pygmsh.occ.Geometry,_object,tool):
    """ fragment the tool and the object, and return the fragments in the following order:
        intersection, object_fragment, tool_fragment.
        in some cases one of the later two may be empty
    """
    object_copy = geom.copy(_object)
    tool_copy = geom.copy(tool)
    try:
        intersection = geom.boolean_intersection([object_copy,tool_copy])
    except:
        # no intersection - make first element None to signal
        return [None,_object,tool]

    _object = geom.boolean_difference(_object,intersection,delete_first=True,delete_other=False)
    tool = geom.boolean_difference(tool,intersection,delete_first=True,delete_other=False)
    return intersection+_object+tool

def boolean_difference(geom,_object,_tool):
    if type(_object) == list:
        for o in _object:
            if type(_tool) == list:
                for t in _tool:
                    o = geom.boolean_difference(o,t,delete_other=False,delete_first=True)
            else:
                o = geom.boolean_difference(o,_tool,delete_other=False,delete_first=True)
    else:
        if type(_tool) == list:
            for t in _tool:
                _object = geom.boolean_difference(_object,t,delete_other=False,delete_first=True)
        else:
            _object = geom.boolean_difference(_object,_tool,delete_other=False,delete_first=True)
    
    return _object

def dist(p1,p2):
    return np.sqrt(np.sum(np.power(p1-p2,2)))

def ellipse_dist(semi_major, semi_minor, c, p, iters=3):
    """ compute signed distance to axis-aligned ellipse boundary """  

    _p = [p[0]-c[0],p[1]-c[1]]

    px = abs(_p[0])
    py = abs(_p[1])

    tx = 0.707
    ty = 0.707

    a = semi_major
    b = semi_minor

    inside = _p[0]**2/semi_major**2 + _p[1]**2/semi_minor**2 <= 1

    for x in range(0,iters):
        x = a * tx
        y = b * ty

        ex = (a*a - b*b) * tx**3 / a
        ey = (b*b - a*a) * ty**3 / b

        rx = x - ex
        ry = y - ey

        qx = px - ex
        qy = py - ey

        r = math.hypot(ry, rx)
        q = math.hypot(qy, qx)
        
        if q == 0:
            tx = 1
            ty = 1
        else:
            tx = min(1, max(0, (qx * r / q + ex) / a))
            ty = min(1, max(0, (qy * r / q + ey) / b))
        
        t = math.hypot(ty, tx)
        tx /= t 
        ty /= t 

    isect = [math.copysign(a * tx,_p[0]), math.copysign(b * ty,_p[1])]
    sgn = 1 if not inside else -1
    dist = math.sqrt((_p[0]-isect[0])**2 + (_p[1]-isect[1])**2)*sgn
    return dist

#endregion    

#region Prim2D
class Prim2D:
    """ a Prim2D (2D primitive) is an an array of N (x,y) points, shape (N,2), that denote a closed curve (so, a polygon). 
        inside the closed curve, the primitive has refractive index n. 
    """
    def __init__(self,n,label,points=[]):
        self.points = points
        self.label = label
        self.n = n
        self.res = len(points)
        self.mesh_size = None # set to a numeric value to force a triangle size within the closed region
        self.skip_refinement = False
    
    def make_poly(self,geom):
        # check depth of self.points
        if hasattr(self.points[0][0],'__len__'):
            ps = [geom.add_polygon(p) for p in self.points]
            poly = geom.boolean_union(ps)[0]
        else:
            poly = geom.add_polygon(self.points)
        return poly

    def update(self,points):
        """ update the primitive according to some args and return an Nx2 array of points.
            the default behavior is to manually pass in a points array. more specific primitives
            inheriting from Prim2D should implement their own update().
        """
        self.points = points
        self.res = len(self.points)
        return points

    def make_points(self):
        """ make an Nx2 array of points for marking the primitive boundary,
            according to some args.
        """
        return self.points

    def plot_mesh(self):
        """ a quick check to see what this object looks like. generates a mesh with default parameters. """
        with pygmsh.occ.Geometry() as geom:
            poly = self.make_poly(geom)
            geom.add_physical(poly,"poly")
            m = geom.generate_mesh(2,2,6)
        plot_mesh(m)            

    def boundary_dist(self,x,y):
        """ this function computes the distance between the point (x,y) and the boundary of the primitive. negative distances -> inside the boundary, while positive -> outside
            note that this doesn't need to be exact. the "distance" just needs to be positive outside the boundary, negative inside the boundary, and go to 0 as you approach the boundary.
        """
        pass

class Circle(Prim2D):
    """ a Circle primitive, defined by radius, center, and number of sides """

    def make_points(self,radius,res,center=(0,0)):
        thetas = np.linspace(0,2*np.pi,res,endpoint=False)
        points = []
        for t in thetas:
            points.append((radius*np.cos(t)+center[0],radius*np.sin(t)+center[1]))        
        points = np.array(points)

        self.radius = radius # save params for later comp
        self.center = center # 
        self.points = points
        return points
    
    def boundary_dist(self, x, y):
        return np.sqrt(np.power(x-self.center[0],2)+np.power(y-self.center[1],2)) - self.radius

class Rectangle(Prim2D):
    """ rectangle primitive, defined by corner pounts. """

    def make_points(self,xmin,xmax,ymin,ymax):
        points = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        self.bounds = [xmin,xmax,ymin,ymax]
        self.points = points
        self.center = [(xmin+xmax)/2,(ymin+ymax)/2]
        return points

    def boundary_dist(self, x, y):
        bounds = self.bounds
        xdist = min(abs(bounds[0]-x),abs(bounds[1]-x))
        ydist = min(abs(bounds[2]-y),abs(bounds[3]-y))
        dist = min(xdist,ydist)
        if bounds[0]<=x<=bounds[1] and bounds[2]<=y<=bounds[3]:
            return -dist
        return dist
    
class Ellipse(Prim2D):
    """axis-aligned ellipse. a is the semi-axis along x, b is the semi-axis along y."""
    def make_points(self,a,b,res,center=(0,0)):
        thetas = np.linspace(0,2*np.pi,res,endpoint=False)
        points = []
        for t in thetas:
            points.append((a*np.cos(t)+center[0],b*np.sin(t)+center[1]))        
        points = np.array(points)

        self.a = a
        self.b = b
        self.center = center
        self.points = points
        return points

    def boundary_dist(self, x, y):
        return ellipse_dist(self.a,self.b,self.center,[x,y])

class Prim2DUnion(Prim2D):
    """ a union of Prim2Ds """
    def __init__(self,ps:list[Prim2D],label):
        ns = [p.n for p in ps]
        assert np.all(np.array(ns)==ns[0]),"primitives must have the same refractive index"
        centers = np.array([p.center for p in ps])
        self.center = np.mean(centers,axis=0)
        points = [p.points for p in ps]
        super().__init__(ps[0].n,label,points)
        self.ps = ps
        d = 0
        eps=1e-10
        boundary_pts = []
        for pts in points:
            for pt in pts:
                for p in ps:
                    d = p.boundary_dist(pt[0],pt[1])
                    if d+eps < 0:
                        break
                if d+eps >=0:
                    boundary_pts.append(pt)
        self.boundary_pts = np.array(boundary_pts)

    def make_points(self,args):
        out = []
        for i,p in enumerate(self.ps):
            points = p.make_points(args[i])
            out.append(points)
        return out
    
    def inside(self,x,y):
        for p in self.ps:
            if p.boundary_dist(x,y)<=0:
                return -1
        return 1

    def boundary_dist(self,x,y): 
        # yes, there is probably some more general way to compute distance to boundary of an arbitrary union of polygons...
        return np.min(np.sqrt((np.power(self.boundary_pts[:,0]-x,2) + np.power(self.boundary_pts[:,1]-y,2)))) * self.inside(x,y)

    def make_poly(self,geom):
        if hasattr(self.points[0][0],'__len__'):
            ps = [geom.add_polygon(p) for p in self.points]
            polys = geom.boolean_union(ps)
            poly = polys
        else:
            poly = geom.add_polygon(self.points)
        return poly

class Prim2DArray(Prim2D):
    """an array of identical non-intersecting Prim2Ds copied at different locations """
    def __init__(self,ps:list[Prim2D],label):
        ns = [p.n for p in ps]
        assert np.all(np.array(ns)==ns[0]),"primitives must have the same refractive index"
        for p in ps:
            assert hasattr(p,'center'), "all primitives must have a defined center point"

        super().__init__(ps[0].n,label,[p.points for p in ps])
        self.ps = ps
        self.centers = np.array([p.center for p in ps])
        self.label = label
    
    def boundary_dist(self, x, y):
        dist_to_centers = np.sqrt( np.power(x- self.centers[:,0],2)+np.power(y- self.centers[:,1],2))
        idx = np.argmin(dist_to_centers)
        return self.ps[idx].boundary_dist(x,y)

    def make_poly(self,geom):
        polys = [geom.add_polygon(p) for p in self.points]
        return polys

#endregion    

#region Waveguide
        
class Waveguide:
    """ a Waveguide is a collection of prim2Dgroups, organized into layers. the refractive index 
    of earler layers is overwritten by later layers.
    """

    isect_skip_layers = [0]

    # mesh params
    mesh_dist_scale = 1.0   # mesh boundary refinement linear distance scaling   
    mesh_dist_power = 1.0   # mesh boundary refinement power scaling
    min_mesh_size = 0.1     # minimum allowed mesh size
    max_mesh_size = 10.     # maximum allowed mesh size

    recon_midpts = True
    vectorized_transform = False

    def __init__(self,prim2Dgroups):
        self.prim2Dgroups = prim2Dgroups # an arrangement of Prim2D objects, stored as a (potentially nested) list. each element is overwritten by the next.
        self.IOR_dict = {}
        
        primsflat = [] # flat array of primitives
        for i,p in enumerate(self.prim2Dgroups):
            if type(p) == list:    
                for _p in p:
                    primsflat.append(_p)
            else:
                primsflat.append(p)  
        
        self.primsflat = primsflat

    def make_mesh(self,algo=6,order=2,adaptive=False):
        """ construct a mesh with boundary refinement at material interfaces."""

        _scale = self.mesh_dist_scale
        _power = self.mesh_dist_power
        min_mesh_size = self.min_mesh_size
        max_mesh_size = self.max_mesh_size

        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber('General.Terminal', 0)
            # make the polygons
            nested_polygons = []
            for el in self.prim2Dgroups:
                if type(el) != list:
                    #polygons.append(geom.add_polygon(el.prim2D.points))
                    poly = el.make_poly(geom)
                    nested_polygons.append(poly)
                else:
                    els = []
                    nested_els = []
                    for _el in el:
                        poly = _el.make_poly(geom)
                        nested_els.append(poly)
                        if type(poly) == list:
                            els += poly
                        else:
                            els.append(poly)
                    nested_polygons.append(nested_els)

            # diff the polygons
            for i in range(0,len(self.prim2Dgroups)-1):
                polys = nested_polygons[i]
                for j in range(i+1,len(self.prim2Dgroups)):
                    _polys = nested_polygons[j]
                    polys = boolean_difference(geom,polys,_polys)

            # add physical groups
            for i,prim in enumerate(self.prim2Dgroups):
                if type(prim) == list:
                    for j,pprim in enumerate(prim):
                        geom.add_physical(nested_polygons[i][j],pprim.label)
                else:
                    geom.add_physical(nested_polygons[i],prim.label)

            if adaptive:
                # mesh refinement callback
                def callback(dim,tag,x,y,z,lc):
                    return self.compute_mesh_size(x,y,_scale=_scale,_power=_power,min_size=min_mesh_size,max_size=max_mesh_size)
                geom.set_mesh_size_callback(callback)

            geom.env.removeAllDuplicates()
            mesh = geom.generate_mesh(dim=2,order=order,algorithm=algo)
            get_unique_edges(mesh)

            return mesh
        
    def compute_mesh_size(self,x,y,_scale=1.,_power=1.,min_size=None,max_size=None):
        """ compute a target mesh size (triangle side length) at the point (x,y). 
        ARGS:
            x: x point to compute mesh size at
            y: y point to compute mesh size at
            _scale: a factor that determines how quickly mesh size should increase away from primitive boundaries. higher = more quickly.
            _power: another factor that determines how mesh size increases away from boundaries. default = 1 (linear increase). higher = more quickly.
            min_size: the minimum mesh size that the algorithm can choose
            max_size: the maximum mesh size that the algorithm can chooose
        """
        
        prims = self.primsflat
        dists = np.zeros(len(prims)) # compute a distance to each primitive boundary
        for i,p in enumerate(prims): 
            if p.skip_refinement and p.mesh_size is not None:
                dists[i] = 0. # if there is a set mesh size and we dont care about boundary refinement, set dist=0 -> fixed mesh size inside primitive later
            else:
                dists[i] = p.boundary_dist(x,y)
        # compute target mesh sizes
        mesh_sizes = np.zeros(len(prims))
        for i,d in enumerate(dists): 
            p = prims[i]
            if hasattr(p.points[0][0],'__len__'):
                boundary_mesh_size = np.sqrt((p.points[0][0][0]-p.points[0][1][0])**2 + (p.points[0][0][1]-p.points[0][1][1])**2) 
            else:
                boundary_mesh_size = np.sqrt((p.points[0][0]-p.points[1][0])**2 + (p.points[0][1]-p.points[1][1])**2) 
            scaled_size = np.power(1+np.abs(d)/boundary_mesh_size *_scale ,_power) * boundary_mesh_size # this goes to boundary_mesh_size as d->0, and increases as d->inf for _power>0
            if d<=0 and p.mesh_size is not None:
                mesh_sizes[i] = min(scaled_size,p.mesh_size)
            else:
                mesh_sizes[i] = scaled_size
        target_size = np.min(mesh_sizes)
        if min_size:
            target_size = max(min_size,target_size)
        if max_size:
            scaled_size = min(max_size,target_size)    
        return target_size

    def assign_IOR(self):
        """ build a dictionary which maps all material labels in the Waveguide mesh
            to the corresponding refractive index value """
        for p in self.prim2Dgroups:
            if type(p) == list:
                for _p in p:
                    if _p.label in self.IOR_dict:
                        continue
                    self.IOR_dict[_p.label] = _p.n
            else:
                if p.label in self.IOR_dict:
                    continue
                self.IOR_dict[p.label] = p.n  
        return self.IOR_dict

    def plot_mesh(self,mesh=None,IOR_dict=None,alpha=0.3,ax=None,plot_points=True):
        """ plot a mesh and associated refractive index distribution
        Args:
        mesh: the mesh to be plotted. if None, we auto-compute a mesh using default values
        IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
        """
        if mesh is None:
            mesh = self.make_mesh()
        if IOR_dict is None:
            IOR_dict = self.assign_IOR()

        plot_mesh(mesh,IOR_dict,alpha,ax,plot_points)
    
    def plot_boundaries(self):
        """ plot the boundaries of all prim2Dgroups. For unioned primitives, all boundaries of 
            the original parts of the union are plotted in a lighter color. """
        for group in self.prim2Dgroups:
            if not type(group) == list:
                group = [group]
            for prim in group:
                p = prim.points
                if hasattr(p[0][0],'__len__'):
                    for _p in p:
                        p2 = np.zeros((_p.shape[0]+1,_p.shape[1]))
                        p2[:-1] = _p[:]
                        p2[-1] = _p[0]
                        plt.plot(p2.T[0],p2.T[1],color='0.5')
                else:
                    p2 = np.zeros((p.shape[0]+1,p.shape[1]))
                    p2[:-1] = p[:]
                    p2[-1] = p[0]
                    plt.plot(p2.T[0],p2.T[1],color='k')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.axis('equal')
        plt.show()
    
class CircularFiber(Waveguide):
    """ circular step-index fiber """
    def __init__(self,rcore,rclad,ncore,nclad,core_res,clad_res=None,core_mesh_size=None,clad_mesh_size=None):
        """
        ARGS
            rcore: radius of core
            rclad: radius of cladding
            ncore: core index
            nclad: cladding index
            core_res: number of line segments to divide the core boundary into
            clad_res: number of line segments to divide the cladding boundary into, default core_res/2
        """
        if clad_res == None:
            clad_res = int(core_res/2)
        core = Circle(ncore,"core")
        core.make_points(rcore,core_res)
        core.mesh_size = core_mesh_size

        cladding = Circle(nclad,"cladding")
        cladding.make_points(rclad,clad_res)
        cladding.mesh_size = clad_mesh_size
        super().__init__([cladding,core])

class EllipticalFiber(Waveguide):
    """ axis-aligned elliptical core step-index fiber """
    def __init__(self,acore,bcore,rclad,ncore,nclad,core_res,clad_res=None,core_mesh_size=None,clad_mesh_size=None):
        """ 
        ARGS
            acore: extent of elliptical core along x (the "x" radius)
            bcore: extent of elliptical core along y (the "y" radius)
            rclad: radius of cladding, assumed circular
            ncore: core index
            nclad: cladding index
            core_res: number of line segments to divide the core boundary into
            clad_res: number of line segments to divide the cladding boundary into, default core_res/2
        """
        if clad_res == None:
            clad_res = int(core_res/2)
        core = Ellipse(ncore,"core")
        core.make_points(acore,bcore,core_res)
        core.mesh_size = core_mesh_size
        cladding = Circle(nclad,"cladding")
        cladding.make_points(rclad,clad_res)
        cladding.mesh_size = clad_mesh_size
        super().__init__([cladding,core])

class PhotonicCrystalFiber(Waveguide):
    """ an optical fiber filled with a hexagonal pattern of air holes, except at the center. must be solved with vector solver. """
    def __init__(self,rhole,rclad,nclad,spacing,hole_res,clad_res,hole_mesh_size=None,clad_mesh_size=None,nhole=1.,rcore=0):
        """
        ARGS:
            rhole: the radius of each air hole perforating the fiber
            rclad: the outer cladding radius of the fiber
            nclad: the index of the cladding material
            spacing: the physical spacing between holes
            hole_res: the # of line segments used to resolve each hole boundary
            clad_res: the # of line segments used to resolve the outer cladding boundary
            hole_mesh_size: (opt.) target mesh size inside holes
            clad_mesh_size: (opt.) target mesh size inside cladding, but outside holes
            nhole: (opt.) index of the holes, default 1.
            rcore: (opt.) the "core radius" of the fiber. holes whose centers are within this radius from the origin are not generated. default 0 (no central hole).
        """    
        
        # get air hole positions
        layers = int(rclad/spacing)
        xa = ya = np.linspace(-layers*spacing,layers*spacing,2*layers+1,endpoint=True)
        xg , yg = np.meshgrid(xa,ya)

        yg *= np.sqrt(3)/2
        if layers%2==1:
            xg[::2, :] += 0.5 * spacing
        else:
            xg[1::2, :] += 0.5 * spacing

        rg = np.sqrt(xg*xg + yg*yg)
        xhole , yhole = xg[rg < rclad-rhole].flatten() , yg[rg < rclad-rhole].flatten()

        # make holes
        holes = []
        for xh,yh in zip(xhole,yhole):
            if xh*xh+yh*yh <= rcore*rcore:
                continue
            hole = Circle(nhole,None)
            hole.make_points(rhole,hole_res,(xh,yh))
            hole.mesh_size = hole_mesh_size
            holes.append(hole)

        # make cladding
        cladding = Circle(nclad,"cladding")
        cladding.make_points(rclad,clad_res)
        cladding.mesh_size = clad_mesh_size

        super().__init__([cladding,Prim2DArray(holes,"holes")])

class PhotonicBandgapFiber(Waveguide):
    """ an optical fiber filled with a hexagonal pattern of air holes, except at the center. must be solved with vector solver. """
    def __init__(self,rvoid,rhole,rclad,nclad,spacing,hole_res,clad_res,hole_mesh_size=None,clad_mesh_size=None,nhole=1.):
        """
        ARGS:
            rcore: the radius of the central air hole
            rhole: the radius of the cladding air holes perforating the fiber
            rclad: the outer cladding radius of the fiber
            nclad: the index of the cladding material
            spacing: the physical spacing between holes
            hole_res: the # of line segments used to resolve each hole boundary
            clad_res: the # of line segments used to resolve the outer cladding boundary
            hole_mesh_size: (opt.) target mesh size inside holes
            clad_mesh_size: (opt.) target mesh size inside cladding, but outside holes
            nhole: (opt.) index of the holes, default 1.
        """    
        
        # get air hole positions
        layers = int(rclad/spacing)+1
        xa = ya = np.linspace(-layers*spacing,layers*spacing,2*layers+1,endpoint=True)
        xg , yg = np.meshgrid(xa,ya)

        yg *= np.sqrt(3)/2
        if layers%2==1:
            xg[::2, :] += 0.5 * spacing
        else:
            xg[1::2, :] += 0.5 * spacing

        rg = np.sqrt(xg*xg + yg*yg)
        xhole , yhole = xg[rg < rclad-rhole].flatten() , yg[rg < rclad-rhole].flatten()

        # make holes
        holes = []
        overlapped_holes = []
        for xh,yh in zip(xhole,yhole):
            hole = Circle(nhole,None)
            hole.make_points(rhole,hole_res,(xh,yh))
            hole.mesh_size = hole_mesh_size
            if np.sqrt(xh*xh+yh*yh)-rhole <= rvoid:
                overlapped_holes.append(hole)
            else:
                holes.append(hole)

        # make cladding
        cladding = Circle(nclad,"cladding")
        cladding.make_points(rclad,clad_res)
        cladding.mesh_size = clad_mesh_size
        
        # make center void
        center = Circle(nhole,None)
        center.make_points(rvoid,int(rvoid/rhole)*hole_res)
        overlapped_holes.append(center)

        void = Prim2DUnion(overlapped_holes,"void")
        void.mesh_size = hole_mesh_size

        hole_array = Prim2DArray(holes,"holes")
        hole_array.mesh_size = hole_mesh_size

        super().__init__([cladding,[hole_array,void]])


class FiberBundleLantern(Waveguide):
    """Photonic lantern with hexagonal arrangement of individual fibers - WaveSolve compatible"""

    def __init__(self, r_jack, r_fiber_clad, r_core, n_rings, n_core, n_clad,
                 core_res=16, clad_res=32, jack_res=None,
                 spacing_factor=2.0, include_center=True,
                 taper_ratio=1.0, r_target_mmcore_size=None,
                 core_mesh_size=None, clad_mesh_size=None,
                 n_jack=None, center_clad_factor=1.5,
                 ring_clad_factors=None):
        """
        Initialize hexagonal fiber bundle photonic lantern.

        Args:
            r_jack: jacket radius
            r_fiber_clad: individual fiber cladding radius
            r_core: core radius (final size after taper)
            n_rings: number of hexagonal rings
            n_core: core refractive index
            n_clad: cladding refractive index
            core_res: resolution for each core circle
            clad_res: resolution for each fiber cladding circle
            jack_res: resolution for jacket circle (default clad_res/2)
            spacing_factor: multiplier for fiber spacing (center-to-center)
            include_center: whether to include center fiber
            taper_ratio: scaling factor (initial_size/final_size)
            r_target_mmcore_size: desired MM core size. Will override taper ratio.
            core_mesh_size: target mesh size in cores
            clad_mesh_size: target mesh size in cladding
            n_jack: jacket refractive index (default same as cladding)
            center_clad_factor: factor to enlarge center fiber cladding (simulates fusing)
            ring_clad_factors: list/dict of cladding scaling factors for each ring
                              Can be:
                              - list: [ring0_factor, ring1_factor, ring2_factor, ...]
                              - dict: {0: ring0_factor, 1: ring1_factor, ...}
                              - None: use center_clad_factor for ring 0, 1.0 for others
        """
        if jack_res is None:
            jack_res = int(clad_res / 2)
        if n_jack is None:
            n_jack = n_clad

        # Process ring cladding factors
        self.ring_clad_factors = self._process_ring_clad_factors(
            ring_clad_factors, n_rings, center_clad_factor, include_center
        )

        # Calculate taper ratio based on target bundle size
        if r_target_mmcore_size is not None:
            # Calculate the radius of the outermost fiber bundle without taper
            original_bundle_radius = self._calculate_bundle_radius(
                n_rings, r_fiber_clad, spacing_factor, include_center
            )
            taper_ratio = r_target_mmcore_size / original_bundle_radius

        # Apply taper ratio to all dimensions
        r_fiber_clad_tapered = r_fiber_clad * taper_ratio
        r_core_tapered = r_core * taper_ratio
        r_jack_tapered = r_jack * taper_ratio

        # Calculate fiber spacing (center-to-center distance)
        spacing = spacing_factor * r_fiber_clad_tapered
        fiber_positions, fiber_rings = self._hex_grid_positions_with_rings(
            n_rings, spacing, include_center
        )

        # Create jacket
        jacket = Circle(n_jack, "jacket")
        jacket.make_points(r_jack_tapered, jack_res)
        jacket.mesh_size = clad_mesh_size

        # Create individual fiber claddings and cores
        fiber_claddings = []
        cores = []

        for i, (pos, ring_idx) in enumerate(zip(fiber_positions, fiber_rings)):
            # Get cladding scaling factor for this ring
            clad_factor = self.ring_clad_factors.get(ring_idx, 1.0)
            clad_radius = r_fiber_clad_tapered * clad_factor

            # Create fiber cladding
            fiber_clad = Circle(n_clad, "cladding")
            fiber_clad.make_points(clad_radius, clad_res, center=pos)
            fiber_clad.mesh_size = clad_mesh_size
            fiber_claddings.append(fiber_clad)

            # Create core at the same position
            core = Circle(n_core, "core")
            core.make_points(r_core_tapered, core_res, center=pos)
            core.mesh_size = core_mesh_size
            cores.append(core)

        # Create arrays for claddings and cores
        fiber_clad_array_Union = Prim2DUnion(fiber_claddings, "cladding")
        fiber_clad_array_Union.mesh_size = clad_mesh_size

        core_array = Prim2DArray(cores, "core")

        # Store metadata
        self.n_fibers = len(fiber_positions)
        self.fiber_positions = fiber_positions
        self.fiber_rings = fiber_rings
        self.taper_ratio = taper_ratio
        self.spacing = spacing
        self.r_fiber_clad = r_fiber_clad_tapered
        self.center_clad_factor = center_clad_factor
        self.bundle_radius = self._calculate_actual_bundle_radius(fiber_positions, r_fiber_clad_tapered)

        # Initialize waveguide with layers (jacket, fiber claddings, cores)
        super().__init__([jacket, fiber_clad_array_Union, core_array])

    def _process_ring_clad_factors(self, ring_clad_factors, n_rings, center_clad_factor, include_center):
        """Process and validate ring cladding factors"""
        factors = {}

        if ring_clad_factors is None:
            # Default behavior: center gets center_clad_factor, others get 1.0
            if include_center:
                factors[0] = center_clad_factor
            for ring in range(1, n_rings + 1):
                factors[ring] = 1.0

        elif isinstance(ring_clad_factors, (list, tuple)):
            # List format: [ring0, ring1, ring2, ...]
            start_ring = 0 if include_center else 1
            for i, factor in enumerate(ring_clad_factors):
                ring_idx = start_ring + i
                if ring_idx <= n_rings:
                    factors[ring_idx] = factor

            # Fill in missing rings with 1.0
            for ring in range(start_ring, n_rings + 1):
                if ring not in factors:
                    factors[ring] = 1.0

        elif isinstance(ring_clad_factors, dict):
            # Dictionary format: {ring_idx: factor}
            factors = ring_clad_factors.copy()

            # Fill in missing rings with 1.0
            start_ring = 0 if include_center else 1
            for ring in range(start_ring, n_rings + 1):
                if ring not in factors:
                    factors[ring] = 1.0

        else:
            raise ValueError("ring_clad_factors must be None, list, tuple, or dict")

        return factors

    def _hex_grid_positions_with_rings(self, n_rings, spacing, include_center=True):
        """Generate hexagonal grid positions for fiber centers with ring information"""
        positions = []
        rings = []

        if include_center:
            positions.append((0, 0))
            rings.append(0)

        for ring in range(1, n_rings + 1):
            for i in range(6 * ring):
                angle = 2 * np.pi * i / (6 * ring)
                edge = int(i / ring)
                pos_on_edge = i % ring
                edge_angle = edge * np.pi / 3
                edge_dir = (edge + 2) * np.pi / 3

                x = ring * spacing * np.cos(edge_angle) + pos_on_edge * spacing * np.cos(edge_dir)
                y = ring * spacing * np.sin(edge_angle) + pos_on_edge * spacing * np.sin(edge_dir)

                positions.append((x, y))
                rings.append(ring)

        return positions, rings

    def _calculate_bundle_radius(self, n_rings, r_fiber_clad, spacing_factor, include_center=True):
        """Calculate the radius of the enscribing circle for the fiber bundle"""
        if n_rings == 0:
            return r_fiber_clad

        # Calculate fiber spacing
        spacing = spacing_factor * r_fiber_clad

        # Distance from center to outermost fiber centers
        outermost_distance = n_rings * spacing

        # Add the fiber cladding radius to get the enscribing circle
        bundle_radius = outermost_distance + r_fiber_clad

        return bundle_radius

    def _calculate_actual_bundle_radius(self, fiber_positions, r_fiber_clad_tapered):
        """Calculate the actual bundle radius from fiber positions"""
        if not fiber_positions:
            return r_fiber_clad_tapered

        # Find the maximum distance from center to any fiber edge
        max_distance = 0
        for pos in fiber_positions:
            fiber_center_distance = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
            fiber_edge_distance = fiber_center_distance + r_fiber_clad_tapered
            max_distance = max(max_distance, fiber_edge_distance)

        return max_distance

    def _hex_grid_positions(self, n_rings, spacing, include_center=True):
        """Generate hexagonal grid positions for fiber centers (legacy method)"""
        positions, _ = self._hex_grid_positions_with_rings(n_rings, spacing, include_center)
        return positions

    def make_mesh(self, algo=6, order=2, adaptive=True):
        """Generate mesh with enhanced control for fiber bundle lanterns"""
        mesh = super().make_mesh(algo, order, adaptive)

        # Add fiber bundle-specific metadata
        mesh.field_data.update({
            "n_fibers": self.n_fibers,
            "taper_ratio": self.taper_ratio,
            "spacing": self.spacing,
            "center_clad_factor": self.center_clad_factor,
            "ring_clad_factors": self.ring_clad_factors,
            "bundle_radius": self.bundle_radius
        })

        return mesh

    def get_fiber_info(self):
        """Return information about individual fibers"""
        info = {
            'n_fibers': self.n_fibers,
            'fiber_positions': self.fiber_positions,
            'fiber_rings': self.fiber_rings,
            'fiber_cladding_radius': self.r_fiber_clad,
            'center_enlarged': self.center_clad_factor > 1.0,
            'center_clad_factor': self.center_clad_factor,
            'ring_clad_factors': self.ring_clad_factors,
            'bundle_radius': self.bundle_radius
        }
        return info

    def get_ring_info(self):
        """Return information about rings and their cladding factors"""
        ring_info = {}
        for ring_idx, factor in self.ring_clad_factors.items():
            fiber_count = 1 if ring_idx == 0 else 6 * ring_idx
            ring_info[ring_idx] = {
                'cladding_factor': factor,
                'fiber_count': fiber_count,
                'effective_clad_radius': self.r_fiber_clad * factor
            }
        return ring_info

class MCFPhotonicLantern(Waveguide):
    """Photonic lantern with hexagonal grid of core, i.e. MCF based lantern."""

    def __init__(self, r_jack, r_clad, r_core, n_rings, n_core, n_clad,
                 core_res=16, clad_res=32, jack_res=None,
                 spacing_factor=2.2, include_center=True,
                 taper_ratio=1.0, r_target_cladding_size=None,
                 core_mesh_size=None, clad_mesh_size=None,
                 n_jack=None):
        """
        Initialize MCF photonic lantern.

        Args:
            r_jack: jacket radius
            r_clad: cladding radius of MCF
            r_core: core radius (final size after taper)
            n_rings: number of hexagonal rings
            n_core: core refractive index
            n_clad: cladding refractive index
            core_res: resolution for each core circle
            clad_res: resolution for cladding circle
            jack_res: resolution for jacket circle (default clad_res/2)
            spacing_factor: multiplier for core spacing
            include_center: whether to include center core
            taper_ratio: scaling factor (initial_size/final_size)
            r_target_cladding_size: desired MM core size. Will overide taper ratio.
            core_mesh_size: target mesh size in cores
            clad_mesh_size: target mesh size in cladding
            n_jack: jacket refractive index (default same as cladding)
        """
        if jack_res is None:
            jack_res = int(clad_res/2)
        if n_jack is None:
            n_jack = n_clad

        # Calculate core positions based on taper
        if r_target_cladding_size is not None:
            taper_ratio = r_target_cladding_size/r_clad

        spacing_base = r_core if taper_ratio == 1.0 else r_core * taper_ratio
        r_clad=r_clad*taper_ratio
        r_core=r_core*taper_ratio
        r_jack=r_jack*taper_ratio

        spacing = spacing_factor * spacing_base
        core_positions = self._hex_grid_positions(n_rings, spacing, include_center)

        # Create jacket
        jacket = Circle(n_jack, "jacket")
        jacket.make_points(r_jack, jack_res)
        jacket.mesh_size = clad_mesh_size

        # Create cladding
        cladding = Circle(n_clad, "cladding")
        cladding.make_points(r_clad, clad_res)
        cladding.mesh_size = clad_mesh_size

        # Create cores
        cores = []
        for i, pos in enumerate(core_positions):
            core = Circle(n_core, None)  # Individual label added later
            core.make_points(r_core, core_res, center=pos)
            core.mesh_size = core_mesh_size
            cores.append(core)

        # Create core array
        core_array = Prim2DArray(cores, "cores")

        # Store metadata
        self.n_cores = len(cores)
        self.core_positions = core_positions
        self.taper_ratio = taper_ratio
        self.spacing = spacing

        # Initialize waveguide with layers
        super().__init__([jacket, cladding, core_array])

        # Override mesh parameters for better lantern meshing
        self.mesh_dist_scale = 0.5
        self.mesh_dist_power = 1.2
        self.min_mesh_size = min(r_core/4, 0.1)
        self.max_mesh_size = r_clad/5

    def _hex_grid_positions(self, n_rings, spacing, include_center=True):
        """Generate hexagonal grid positions"""
        positions = []

        if include_center:
            positions.append((0, 0))

        for ring in range(1, n_rings + 1):
            for i in range(6 * ring):
                angle = 2 * np.pi * i / (6 * ring)
                edge = int(i / ring)
                pos_on_edge = i % ring
                edge_angle = edge * np.pi / 3
                edge_dir = (edge + 2) * np.pi / 3

                x = ring * spacing * np.cos(edge_angle) + pos_on_edge * spacing * np.cos(edge_dir)
                y = ring * spacing * np.sin(edge_angle) + pos_on_edge * spacing * np.sin(edge_dir)

                positions.append((x, y))

        return positions

    def make_mesh(self, algo=6, order=2, adaptive=True):
        """Generate mesh with enhanced control for photonic lanterns"""
        mesh = super().make_mesh(algo, order, adaptive)

        # Add lantern-specific metadata
        mesh.field_data.update({
            "n_cores": self.n_cores,
            "taper_ratio": self.taper_ratio,
            "spacing": self.spacing
        })

        return mesh
#endregion