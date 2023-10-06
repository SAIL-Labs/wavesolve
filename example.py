
"""
ncore = 1.444 + 0.01036
nclad = 1.444
njack = 1.444 - 5.5e-3

k = 2*np.pi/1.55
res = 40

r_clad = 10
w = 4*r_clad
r_core = 2.2/4

### core shift analysis

cores = [(0,0)] + circ_points(2*r_clad/3,5)
IOR_dict = {"core0":ncore,"core1":ncore,"core2":ncore,"core3":ncore,"core4":nclad,"core5":nclad,"core6":nclad,"cladding":nclad,"jacket":njack}

IOR_dict2 = IOR_dict = {"core0":ncore,"core1":ncore,"core2":nclad,"core3":nclad,"core4":ncore,"core5":ncore,"core6":nclad,"cladding":nclad,"jacket":njack}

mesh = lantern_mesh_displaced_circles(w/2,r_clad,cores,r_core,30,ds=0.2)
#mesh = lantern_mesh(w/2,r_clad,cores,r_core,res,petals=5,petal_amp=0.2)

print("mesh and refractive index distribution")
plot_mesh(mesh,IOR_dict=IOR_dict)

A,B = construct_AB(mesh,IOR_dict,k)
w,v = solve(A,B,mesh,k,IOR_dict,plot=False)


_A,_B = construct_AB(mesh,IOR_dict2,k)
_w,_v = solve(_A,_B,mesh,k,IOR_dict2,plot=False)

for vvec, _vvec in zip(v,_v):
    plot_eigenvector(mesh,_vvec-vvec)
"""

# target mode, v[5]

#optimize_for_mode_structure(mesh,IOR_dict,k,v[1],iterations=1)

def get_num_modes(wl,r):
    k = 2*np.pi/wl
    ncore = 1.444
    nclad = 1.444-5.5e-3
    m = lantern_mesh_3PL(r,16)
    IOR_dict = {"jacket":nclad,"cladding":ncore}
    A,B = construct_AB(m,IOR_dict,k)
    _A = csr_matrix(A)
    _B = csr_matrix(B)

    w,v,n = solve_sparse(_A,_B,m,k,IOR_dict,plot=False)
    return n

def get_num_modes_6pl(wl,r):

    k = 2*np.pi/wl
    ncore = 1.4504
    nclad = 1.444
    njack = 1.444-5.8e-3
    m = lantern_mesh_6PL(r,16)
    IOR_dict = {"jacket":njack,"cladding":nclad,"core":ncore}
    #plot_mesh(m,IOR_dict)
    A,B = construct_AB(m,IOR_dict,k)
    w,v,n = solve(A,B,m,k,IOR_dict,plot=False)
    return n

# 6pl
wl_arr = np.linspace(0.9,1.8,20)
r_arr = np.linspace(5.5,7.5,20)
out = np.zeros((20,20))

ncore = 1.4504
nclad = 1.444
njack = 1.444-5.8e-3
IOR_dict = {"jacket":njack,"cladding":nclad,"core":ncore}
itot=0
"""
for j in range(20):
    m = lantern_mesh_6PL(r_arr[j],16)
    for i in range(20):
        print("iteration "+str(itot))
        wl = wl_arr[i]
        k = 2*np.pi/wl
        A,B = construct_AB(m,IOR_dict,k)
        w,v,n = solve(A,B,m,k,IOR_dict,plot=False)
        out[i,j] = n
        itot+=1
"""
#np.save("countmap_6pl",out)
out = np.load("countmap_6pl.npy")
plt.imshow(out.T,origin='lower',extent=(0.9,1.8,5.5,7.5),aspect=0.8/2)
plt.xlabel("wavelength")
plt.ylabel("core radius")
plt.colorbar()
plt.show()

# 3 PL
"""
wl_arr = np.linspace(1,1.8,20)
r_arr = np.linspace(3.3-0.5,5.3-0.5,20)

out = np.zeros((20,20))
ncore = 1.444
nclad = 1.444-5.5e-3
IOR_dict = {"jacket":nclad,"cladding":ncore}
itot = 0
for j in range(20):
    m = lantern_mesh_3PL(r_arr[j],16)
    for i in range(20):
        print("iteration "+str(itot))
        wl = wl_arr[i]
        k = 2*np.pi/wl
        A,B = construct_AB(m,IOR_dict,k)
        #_A = csr_matrix(A)
        #_B = csr_matrix(B)
        w,v,n = solve(A,B,m,k,IOR_dict,plot=False)
        out[i,j] = n
        itot+=1

np.save("countmap",out)
plt.imshow(out.T,origin='lower',extent=(1,1.8,2.8*2/np.sqrt(3),4.8*2/np.sqrt(3)),aspect=4/np.sqrt(3)/0.8)
plt.xlabel("wavelength")
plt.ylabel("core radius")
plt.show()
"""