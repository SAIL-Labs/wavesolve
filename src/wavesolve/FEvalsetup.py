from juliacall import Pkg as jlPkg
import wavesolve,os
path = os.path.dirname(wavesolve.__file__)

jlPkg.activate(path+"/FEval")
jlPkg.add("PythonCall")
jlPkg.add("StaticArrays")
