import numpy as np

def get_basis_funcs_affine():
    """ returns 6 functions correpsonding to the 6 nodes in a QT element."""

    N0 = lambda u,v: 2 * (1 - u - v) * (0.5 - u - v)
    N1 = lambda u,v: 2 * u * (u - 0.5)
    N2 = lambda u,v: 2 * v * (v - 0.5)
    N3 = lambda u,v: 4 * u * (1 - u - v)
    N4 = lambda u,v: 4 * u * v
    N5 = lambda u,v: 4 * v * (1 - u - v)
    return [N0,N1,N2,N3,N4,N5]

def basis_derivs_affine():
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

def compute_dNdN(tri):

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

def compute_J(vertices):
    x21 = vertices[1,0] - vertices[0,0]
    y21 = vertices[1,1] - vertices[0,1]
    x31 = vertices[2,0] - vertices[0,0]
    y31 = vertices[2,1] - vertices[0,1]
    _J = x21*y31-x31*y21
    return _J   



