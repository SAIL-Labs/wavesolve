import numpy as np
import numexpr as ne

# shape functions
N0 = lambda u,v: 2 * (1 - u - v) * (0.5 - u - v)
N1 = lambda u,v: 2 * u * (u - 0.5)
N2 = lambda u,v: 2 * v * (v - 0.5)
N3 = lambda u,v: 4 * u * (1 - u - v)
N4 = lambda u,v: 4 * u * v
N5 = lambda u,v: 4 * v * (1 - u - v)

# shape function derivs
dN0u = lambda u,v: 4*u+4*v-3
dN0v = lambda u,v: 4*u+4*v-3
dN1u = lambda u,v: 4*u-1
dN1v = lambda u,v: 0
dN2u = lambda u,v: 0
dN2v = lambda u,v: 4*v-1
dN3u = lambda u,v: -4*(2*u+v-1)
dN3v = lambda u,v: -4*u
dN4u = lambda u,v: 4*v
dN4v = lambda u,v: 4*u
dN5u = lambda u,v: -4*v
dN5v = lambda u,v: -4*(u+2*v-1)

def evaluate_basis_funcs(u,v):
    return np.array([N0(u,v),N1(u,v),N2(u,v),N3(u,v),N4(u,v),N5(u,v)])

def get_basis_funcs_affine():
    """ returns 6 functions correpsonding to the 6 nodes in a QT element."""
    return [N0,N1,N2,N3,N4,N5]

def basis_derivs_affine():
    return [[dN0u,dN0v],[dN1u,dN1v],[dN2u,dN2v],[dN3u,dN3v],[dN4u,dN4v],[dN5u,dN5v]]

def affine_transform(vertices):
    """ returns a function that maps x-y to u-v such that u,v = (0,0), (1,0), and (0,1) are the new coords for the vertices"""
    
    x21 = vertices[1,0] - vertices[0,0]
    y21 = vertices[1,1] - vertices[0,1]
    x31 = vertices[2,0] - vertices[0,0]
    y31 = vertices[2,1] - vertices[0,1]
    _J = x21*y31-x31*y21
    M = np.array([[y31,-x31],[-y21,x21]])/_J
    
    def inner(xy):
        return np.dot(M,xy-vertices[0])

    return inner

def affine_transform_matrix(vertices):
    """ returns a matrix that maps x-y to u-v such that u,v = (0,0), (1,0), and (0,1) are the new coords for the vertices"""
    
    x21 = vertices[1,0] - vertices[0,0]
    y21 = vertices[1,1] - vertices[0,1]
    x31 = vertices[2,0] - vertices[0,0]
    y31 = vertices[2,1] - vertices[0,1]
    _J = x21*y31-x31*y21
    M = np.array([[y31,-x31],[-y21,x21]])/_J
    return M

def apply_affine_transform(vertices,xy):
    M = affine_transform_matrix(vertices)
    return np.dot(M,xy-vertices[0])

def compute_NN(tri):
    out = np.zeros((6,6),dtype=np.float64)
    
    # nodes assumed to be ordered vertices (clockwise) then edges (clockwise). first vertex is origin of affine transform
    x21 = tri[1,0] - tri[0,0]
    y21 = tri[1,1] - tri[0,1]
    x31 = tri[2,0] - tri[0,0]
    y31 = tri[2,1] - tri[0,1]
    x32 = tri[2,0] - tri[1,0]
    y32 = tri[2,1] - tri[1,1]
    _J = x21*y31 - x31*y21

    ## diagonals
    out[0,0] = out[1,1] = out[2,2] = _J/60
    out[3,3] = out[4,4] = out[5,5] = 4*_J/45
    ## off diagonals
    # vertex only relations
    out[1,0] = out[0,1] = out[1,2] = out[2,1] = out[2,0] = out[0,2] = -_J/360
    # vertex - adjacent edge relations: 0
    # vertex - opposite edge relations
    out[0,4] = out[4,0] = out[1,5] = out[5,1] = out[2,3] = out[3,2] = -_J/90
    # edge edge relations
    out[3,4] = out[4,3] = out[3,5] = out[5,3] = out[4,5] = out[5,4] = 2*_J/45
    return out

def compute_NN_precomp(_J):
    out = np.zeros((6,6),dtype=np.float64)
    out[0,0] = out[1,1] = out[2,2] = _J/60
    out[3,3] = out[4,4] = out[5,5] = 4*_J/45
    out[1,0] = out[0,1] = out[1,2] = out[2,1] = out[2,0] = out[0,2] = -_J/360
    out[0,4] = out[4,0] = out[1,5] = out[5,1] = out[2,3] = out[3,2] = -_J/90
    out[3,4] = out[4,3] = out[3,5] = out[5,3] = out[4,5] = out[5,4] = 2*_J/45
    return out

def compute_J_vec(points,tris):

    x0 = points[tris[:,0]][:,0]
    x1 = points[tris[:,1]][:,0]
    x2 = points[tris[:,2]][:,0]

    y0 = points[tris[:,0]][:,1]
    y1 = points[tris[:,1]][:,1]
    y2 = points[tris[:,2]][:,1]

    _J = ne.evaluate("(x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)",local_dict={"x0":x0,"x1":x1,"x2":x2,"y0":y0,"y1":y1,"y2":y2}) 
    return _J

def compute_J_and_mat(points,tris):
    x1 = points[tris[:,0]][:,0]
    x2 = points[tris[:,1]][:,0]
    x3 = points[tris[:,2]][:,0]

    y1 = points[tris[:,0]][:,1]
    y2 = points[tris[:,1]][:,1]
    y3 = points[tris[:,2]][:,1]

    x21 = x2-x1
    y21 = y2-y1
    x32 = x3-x2
    y32 = y3-y2
    x31 = x3-x1
    y31 = y3-y1

    var_dict = {"x21":x21,"y31":y31,"x31":x31,"y21":y21,"x32":x32,"y32":y32} 
    _J = ne.evaluate("x21*y31-x31*y21",local_dict=var_dict)
    var_dict["_J"] = _J

    mat = np.zeros((len(tris),15))
    mat[:,0] = ne.evaluate("(y32*y32 + x32*x32)/(2*_J)",local_dict=var_dict)
    mat[:,1] = ne.evaluate("(y31*y31 + x31*x31)/(2*_J)")
    mat[:,2] = ne.evaluate("(y21*y21 + x21*x21)/(2*_J)") 
    mat[:,3] = ne.evaluate("4/(3*_J) * (y32**2+y31*y21+x32**2+x31*x21)")
    mat[:,4] = ne.evaluate("4/(3*_J) * (y31**2-y21*y32+x31**2-x21*x32)")
    mat[:,5] = ne.evaluate("4/(3*_J) * (y21**2+y32*y31+x21**2+x32*x31)")
    mat[:,6] = ne.evaluate("(y32*y31+x32*x31)/(6*_J)")
    mat[:,7] = ne.evaluate("(y31*y21+x31*x21)/(6*_J)")
    mat[:,8] = ne.evaluate("(-y21*y32-x21*x32)/(6*_J)")
    mat[:,9] = ne.evaluate("-2*(y32*y31+x32*x31)/(3*_J)")
    mat[:,10] = ne.evaluate("2*(y21*y32+x21*x32)/(3*_J)")
    mat[:,11] = ne.evaluate("-2*(y31*y21+x31*x21)/(3*_J)")
    mat[:,12] = ne.evaluate("4/(3*_J)*(y21*y32+x21*x32)")
    mat[:,13] = ne.evaluate("-4/(3*_J)*(y31*y32+x31*x32)")
    mat[:,14] = ne.evaluate("-4/(3*_J)*(y31*y21+x31*x21)")

    return _J,mat
    
def compute_dNdN(tri,_J=None,row=None):

    if _J is None:
        x21 = tri[1,0] - tri[0,0]
        y21 = tri[1,1] - tri[0,1]
        x31 = tri[2,0] - tri[0,0]
        y31 = tri[2,1] - tri[0,1]
        x32 = tri[2,0] - tri[1,0]
        y32 = tri[2,1] - tri[1,1]
        _J = x21*y31 - x31*y21

    out = np.zeros((6,6),dtype=np.float64)

    # diagonals
    out[0,0] = (y32*y32 + x32*x32)/(2*_J)
    out[1,1] = (y31*y31 + x31*x31)/(2*_J)
    out[2,2] = (y21*y21 + x21*x21)/(2*_J)
    out[3,3] = 4/(3*_J) * (y32**2+y31*y21+x32**2+x31*x21)
    out[4,4] = 4/(3*_J) * (y31**2-y21*y32+x31**2-x21*x32)
    out[5,5] = 4/(3*_J) * (y21**2+y32*y31+x21**2+x32*x31)

    # vertex - vertex (going clockwise)
    out[0,1] = out[1,0] = (y32*y31+x32*x31)/(6*_J)
    out[1,2] = out[2,1] = (y31*y21+x31*x21)/(6*_J)
    out[2,0] = out[0,2] = (-y21*y32-x21*x32)/(6*_J)

    # vertex - adjacent edge
    out[0,3] = out[3,0] = out[1,3] = out[3,1]  = -2*(y32*y31+x32*x31)/(3*_J)
    out[0,5] = out[5,0] = out[2,5] = out[5,2] = 2*(y21*y32+x21*x32)/(3*_J)
    out[2,4] = out[4,2] = out[1,4] = out[4,1] = -2*(y31*y21+x31*x21)/(3*_J)

    # vertex - opposite edge: 0

    # edge-edge
    out[3,4] = out[4,3] =  4/(3*_J)*(y21*y32+x21*x32)
    out[4,5] = out[5,4] = -4/(3*_J)*(y31*y32+x31*x32)
    out[5,3] = out[3,5] = -4/(3*_J)*(y31*y21+x31*x21)

    return out

def compute_dNdN_precomp(row):
    out = np.zeros((6,6),dtype=np.float64)

    # diagonals
    out[0,0] = row[0]
    out[1,1] = row[1]
    out[2,2] = row[2]
    out[3,3] = row[3]
    out[4,4] = row[4]
    out[5,5] = row[5]

    # vertex - vertex (going clockwise)
    out[0,1] = out[1,0] = row[6]
    out[1,2] = out[2,1] = row[7]
    out[2,0] = out[0,2] = row[8]

    # vertex - adjacent edge
    out[0,3] = out[3,0] = out[1,3] = out[3,1]  = row[9]
    out[0,5] = out[5,0] = out[2,5] = out[5,2] = row[10]
    out[2,4] = out[4,2] = out[1,4] = out[4,1] = row[11]

    # vertex - opposite edge: 0

    # edge-edge
    out[3,4] = out[4,3] = row[12]
    out[4,5] = out[5,4] = row[13]
    out[5,3] = out[3,5] = row[14]

    return out

def compute_J(vertices):
    x21 = vertices[1,0] - vertices[0,0]
    y21 = vertices[1,1] - vertices[0,1]
    x31 = vertices[2,0] - vertices[0,0]
    y31 = vertices[2,1] - vertices[0,1]
    _J = x21*y31-x31*y21
    return _J   

### vector shape functions - linear triangle elements ###
### reference: https://ieeexplore.ieee.org/document/5628380

LN0 = lambda u,v: 1 - u - v
LN1 = lambda u,v: u
LN2 = lambda u,v: v 

dLN0du = -1
dLN0dv = -1
dLN1du = 1
dLN1dv = 0
dLN2du = 0
dLN2dv = 1

# shape functions for edges
# edge 0 = point 0 -> point 1, etc.

LNe0 = lambda u,v: (1-v,u) #(LN0(u,v)*dLN1du - LN1(u,v)*dLN0du , LN0(u,v)*dLN1dv - LN1(u,v)*dLN0dv)
LNe1 = lambda u,v: (-np.sqrt(2)*v,np.sqrt(2)*u) #((LN1(u,v)*dLN2du - LN2(u,v)*dLN1du)*np.sqrt(2) , (LN1(u,v)*dLN2dv - LN2(u,v)*dLN1dv)*np.sqrt(2)) # sqrt(2) * (-v,u)
LNe2 = lambda u,v: (-v,-1+u)# (LN2(u,v)*dLN0du - LN0(u,v)*dLN2du , LN2(u,v)*dLN0dv - LN0(u,v)*dLN2dv)

curlLNe0 = 2
curlLNe1 = 2*np.sqrt(2)
curlLNe2 = 2

def computeL_Ne_Ne(tri): # nenenene
    """ integral of LNe_i LNe_j over triangle tri """
    out = np.zeros((3,3),dtype=np.float64)
    x10 = tri[1,0] - tri[0,0]
    y10 = tri[1,1] - tri[0,1]
    x20 = tri[2,0] - tri[0,0]
    y20 = tri[2,1] - tri[0,1]
    _J = x10*y20 - x20*y10
    
    # edge order is 01 , 12, 20 
    out[0,0] = out[1,1] = out[2,2] = 1/3
    out[0,1] = out[1,0] = out[1,2] = out[2,1] = 0
    out[0,2] = out[2,0] = -1/6
    out *= _J
    return out

def computeL_Ne_dN(tri):
    """ integral of LNe_i and LN_j """
    out = np.zeros((3,3),dtype=np.float64)
    x10 = tri[1,0] - tri[0,0]
    y10 = tri[1,1] - tri[0,1]
    x20 = tri[2,0] - tri[0,0]
    y20 = tri[2,1] - tri[0,1]
    _J = x10*y20 - x20*y10
    out[0,0] = -1/2
    out[1,1] = -np.sqrt(2)/6
    out[2,2] = -1/3
    out[0,1] = 1/3
    out[1,0] = 0
    out[0,2] = 1/6
    out[2,0] = 1/2
    out[1,2] = np.sqrt(2)/6
    out[2,1] = -1/6
    out *= _J
    return out

def computeL_NN(tri):
    out = np.zeros((3,3),dtype=np.float64)
    x10 = tri[1,0] - tri[0,0]
    y10 = tri[1,1] - tri[0,1]
    x20 = tri[2,0] - tri[0,0]
    y20 = tri[2,1] - tri[0,1]
    _J = x10*y20 - x20*y10
    out[:] = 1/24
    out[0,0] = out[1,1] = out[2,2] = 1/12
    out *= _J
    return out

def computeL_dNdN(tri):
    out = np.zeros((3,3),dtype=np.float64)
    x10 = tri[1,0] - tri[0,0]
    y10 = tri[1,1] - tri[0,1]
    x20 = tri[2,0] - tri[0,0]
    y20 = tri[2,1] - tri[0,1]
    _J = x10*y20 - x20*y10
    out[0,0] = 1
    out[1,1] = out[2,2] = 1/2
    out[0,1] = out[1,0] = out[0,2] = out[2,0] = -1/2
    out[1,2] = out[2,1] = 0
    out *= _J
    return out
