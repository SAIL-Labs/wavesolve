# wavesolve
`wavesolve` is a lightweight code to solve for the eigenmodes of waveguides. 
It uses the finite-element method, and generates waveguide meshes through the `pygmsh` package. More details on the math behind `wavesolve` are included <a href="finite_element_method_notes.pdf">here</a>.

## installation
Use pip: `pip install git+https://github.com/jw-lin/wavesolve.git`

Python dependencies: `numpy`,`scipy`,`matplotlib`,`numexpr`,`pygmsh`,`jupyter` \
Other dependencies: <a href="https://gmsh.info/">`Gmsh`</a> (required for `pygmsh`).

## documentation
Admittedly a little sparse right now (but this will change in the future!). See <a href="getting-started.ipynb">`getting-started.ipynb`</a> for an overview and some working examples.



