""" use finite element method with quadratic triangular elements to solve for modes in the SCALAR approximation. 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

def construct_AB(mesh,IOR_dict,k,poke_index = None):
    """ construct the A and B matrices corresponding to the given waveguide geometry.
    Args:
    mesh: the waveguide mesh, produced by wavesolve.mesher or pygmsh
    IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
    k: free-space wavenumber of propagation

    Returns:
    A,B: matrices
    """
    from wavesolve.shape_funcs import compute_dNdN, compute_NN
    
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
            A[ix] += (k**2*IOR_dict[material]**2) * NN - dNdN
            B[ix] += NN

    # now poke if necessary
    if poke_index is not None:
        for j,tri in enumerate(mesh.cells[1].data):
            if j == poke_index:
                tri_points = mesh.points[tri]
                NN = compute_NN(tri_points)
                ix = np.ix_(tri,tri)
                A[ix] += 0.1 * NN

    return A,B

def construct_AB_expl_IOR(mesh,IOR_arr,k):
    from wavesolve.shape_funcs import compute_dNdN, compute_NN
    
    points = mesh.points
    materials = mesh.cell_sets.keys()

    N = len(points)

    A = np.zeros((N,N))
    B = np.zeros((N,N))

    for tri,IOR in zip(mesh.cells[1].data,IOR_arr):
        tri_points = points[tri]
        NN = compute_NN(tri_points)
        dNdN = compute_dNdN(tri_points)
        ix = np.ix_(tri,tri)
        A[ix] += (k**2*IOR**2) * NN - dNdN
        B[ix] += NN

    return A,B

# turns out... this isn't actually faster. lol. completely dominated by solving system, not generating matrix
def construct_AB_fast(mesh,IOR_dict,k):
    from wavesolve.shape_funcs import compute_dNdN_precomp, compute_J_and_mat, compute_NN_precomp
    
    points = mesh.points
    verts = mesh.cells[1].data
    materials = mesh.cell_sets.keys()

    N = len(points)

    A = np.zeros((N,N))
    B = np.zeros((N,N))

    IORs = np.zeros(N)
    
    for material in materials:
        IORs[tuple(mesh.cell_sets[material])] = IOR_dict[material]

    _Js,mat = compute_J_and_mat(points,verts)

    for _J,row,IOR,idxs in zip(_Js,mat,IORs,verts):
        NN = compute_NN_precomp(_J)
        dNdN = compute_dNdN_precomp(row)
        ix = np.ix_(idxs,idxs)
        A[ix] += k**2*IOR**2 * NN - dNdN
        B[ix] += NN

    return A,B

def solve(A,B,mesh,k,IOR_dict,plot=False):
    """ Given the A,B matrices, solve the general eigenvalue problem A v = w B v
        where v are the eigenvectors and w the eigenvalues.
        Args:
        A: A matrix of eigenvalue problem
        B: B matrix of eigenvalue
        mesh: the waveguide mesh, produced by wavesolve.mesher or pygmsh
        k: free-space wavenumber of propagation
        IOR_dict: dictionary that assigns each named region in the mesh to a refractive index value
        plot: set True to plot eigenvectors in descending order of eigenvalue

        Returns: 
        w: list of eigenvalues in descending order
        v: list of eigenvectors
        N: number of non-spurious eigenvectors (guided modes) found.
    """
    w,v = eigh(A,B)

    IORs = [ior[1] for ior in IOR_dict.items()]
    nmin,nmax = min(IORs) , max(IORs)
    mode_count = 0
    
    for _w,_v in zip(w[::-1],v.T[::-1]):
        if _w<0:
            continue
        ne = np.sqrt(_w/k**2)
        if plot:
            if not (nmin <= ne <= nmax):
                print("warning: spurious mode! ")
            
            print("effective index: ",np.sqrt(_w/k**2))
            plot_eigenvector(mesh,_v)
        if (nmin <= ne <= nmax):
            mode_count+=1
        else:
            break
    
    # normalization
    # v /= np.sqrt(np.sum(np.power(v,2),axis=0))[None,:]

    return w[::-1],v.T[::-1],mode_count

def solve_sparse(A,B,mesh,k,IOR_dict,plot=False,num_modes=6):
    """An extension of solve() to A and B matrices in CSR format."""
    w,v = eigsh(A,M=B,k=num_modes,which="LA")
    IORs = [ior[1] for ior in IOR_dict.items()]
    nmin,nmax = min(IORs) , max(IORs)
    mode_count = 0
    
    for _w,_v in zip(w[::-1],v.T[::-1]):
        if _w<0:
            continue
        ne = np.sqrt(_w/k**2)
        if plot:
            if not (nmin <= ne <= nmax):
                print("warning: spurious mode! ")
            
            print("effective index: ",np.sqrt(_w/k**2))
            plot_eigenvector(mesh,_v)
        if (nmin <= ne <= nmax):
            mode_count+=1
        else:
            break
    
    # normalization
    # v /= np.sqrt(np.sum(np.power(v,2),axis=0))[None,:]

    return w[::-1],v.T[::-1],mode_count

def plot_eigenvector(mesh,v,plot_mesh = False,plot_circle=False):
    points = mesh.points
    fig,ax = plt.subplots(figsize=(5,5))
    plt.axis('equal')
    plt.tricontourf(points[:,0],points[:,1],v,levels=60)
    plt.colorbar()
    if plot_mesh:
        plot_mesh(mesh,show=False,ax=ax)
    if plot_circle:
        circle = plt.Circle((0,0),mesh.cell_data['radius'],ec='white',fc='None',lw=2)
        ax.add_patch(circle)
    plt.show()

def compute_diff(tri_idx,mesh,_pinv):
    from wavesolve.shape_funcs import compute_NN
    point_idxs = mesh.cells[1].data[tri_idx]
    points = mesh.points[point_idxs]
    N = len(mesh.points)
    ix = np.ix_(point_idxs,point_idxs)
    B_tri = compute_NN(points)

    return np.dot(_pinv[ix],-B_tri),ix,point_idxs

def compute_IOR_arr(mesh,IOR_dict):
    IORs = np.zeros(len(mesh.cells[1].data))
    materials = mesh.cell_sets.keys()
    for material in materials:
        IORs[tuple(mesh.cell_sets[material])] = IOR_dict[material]
    
    return IORs

def optimize_for_mode_structure(mesh,IOR_dict,k,target_field,iterations = 1):
    '''find the refractive index profile such that the fundamental mode most closely matches the target field'''

    # first, solve the base mesh
    A,B = construct_AB(mesh,IOR_dict,k)
    w,v = solve(A,B,mesh,k,IOR_dict,plot=False)

    for _it in range(iterations):

        # next, compute the matrix of derivatives
        _pinv = np.linalg.pinv(A - w[0]*B)

        N_tris = len(mesh.cells[1].data)
        N_points = A.shape[0]

        diff_mat = np.zeros((N_tris,N_points))

        for i in range(N_tris):
            _diff,ix,point_idxs = compute_diff(i,mesh,_pinv)
            diff_field = np.dot(_diff,v[0][point_idxs])
            diff_field /= np.sqrt(np.sum(np.power(diff_field,2)))
            diff_mat[i,point_idxs] = diff_field

        # next, compute the difference between v[0] and the target
        diff = v[0] - target_field

        # compute the overlaps between the derivatives and the difference field
        coeffs = np.dot(diff_mat,diff)

        # subtract off the coeffs from refractive index profile
        IOR0 = compute_IOR_arr(mesh,IOR_dict)
        IOR = np.sqrt(np.power(IOR0,2)*k**2-coeffs)/k

        # check the new results
        A,B = construct_AB_expl_IOR(mesh,IOR,k)
        w,v = solve(A,B,mesh,k,IOR_dict)

    # plot the result
    #plot_eigenvector(IOR)
    from mesher import plot_mesh_expl_IOR
    plot_mesh_expl_IOR(mesh,IOR)

    plot_eigenvector(mesh,v[0])

    xcs = []
    ycs = []
    for tri in mesh.cells[1].data:
        tri_points = mesh.points[tri]
        xm = np.mean(tri_points[:,0])
        ym = np.mean(tri_points[:,1])
        xcs.append(xm)
        ycs.append(ym)

    plt.tricontourf(xcs,ycs,IOR,levels=40)
    plt.colorbar()
    plt.show()

def isinside(v, tri, include_vertex = True):
    ''' checks if the given point is inside the triangle '''
    v0 = tri[0]
    v1 = tri[1] - tri[0]
    v2 = tri[2] - tri[0]
    det = lambda u,v : u[0]*v[1] - u[1]*v[0]
    a = (det(v,v2) - det(v0,v2)) / det(v1,v2)
    b = -(det(v,v1) - det(v0,v1)) / det(v1,v2)

    if include_vertex:
        if (a>=0 and b>=0 and a+b<=1): 
            return True
        else: return False    
    else:
        if (a>0 and b>0 and a+b<1): 
            return True
        else: return False    

def find_triangle(gridpoint, mesh):
    ''' 
    Finds which triangle the point lies in

    Args: 
    gridpoint: [x,y] cartesian coordinates
    mesh: mesh
    
    Output:
    triangle_index: the index of the triangle that the [x,y] point lies in.
                    returns -99 if the point doesn't lie in any triangle.
    '''
    points = mesh.points
    tris = mesh.cells[1].data 
    for i in range(len(tris)):
        tri_points = points[tris[i]]
        if isinside(gridpoint, tri_points[:,:2]) is True:
            return i
    return None

def interpolate_field(gridpoint, index, v, mesh):
    '''
    Finds the field at [x,y] by interpolating the solution found on the triangular mesh

    Args:
    gridpoint: [x,y] cartesian coordinates
    index: the index of the triangle that the point lies in (found from find_triangle function)
    v: the field (solution) to interpolate
    mesh: mesh

    Output:
    interpolated: the interpolated field on the [x, y] point
    '''
    from wavesolve.shape_funcs import affine_transform, get_basis_funcs_affine    

    if index*0 != 0: return np.nan
    points = mesh.points
    tris = mesh.cells[1].data
    field_points = v[tris[int(index)]]

    vertices = points[tris[int(index)]][:,:2]
    uvcoord = affine_transform(vertices)(gridpoint)
    interpolated = 0
    for ii in range(6):
        interpolated += get_basis_funcs_affine()[ii](uvcoord[0], uvcoord[1]) * field_points[ii]

    return interpolated    