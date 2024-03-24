from juliacall import Pkg as jlPkg
import wavesolve,os
path = os.path.dirname(wavesolve.__file__)


def FEvalsetup():
    path = os.path.dirname(wavesolve.__file__)
    #jlPkg.develop(jlPkg.PackageSpec(path = path+"\FEval") )
    jlPkg.activate(path+"/FEval")
    jlPkg.add("StaticArrays")
    jlPkg.add("PythonCall")
    jlPkg.precompile()