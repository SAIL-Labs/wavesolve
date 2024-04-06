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
    edge_indices = np.zeros_like(tris)

    i = 0
    for j,tri in enumerate(tris):
        e0 = sorted([tri[0],tri[1]])
        e1 = sorted([tri[1],tri[2]])
        e2 = sorted([tri[2],tri[0]])
        es= [e0,e1,e2]
        for k,e in enumerate(es):
            if e not in unique_edges:
                unique_edges.append(e)
                edge_indices[j,k] = i
                i += 1
            else:
                edge_indices[j,k] = unique_edges.index(e)
    out = np.array(unique_edges)
    if mutate:
        mesh.cells[0].data = out
        mesh.edge_indices = edge_indices
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

        super().__init__([cladding,[hole_array,void]])


#endregion