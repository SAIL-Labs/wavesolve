import numpy as np
import pytest

from wavesolve import fe_solver
from wavesolve.fe_solver import solve_waveguide_vec
from wavesolve.waveguide import CircularFiber

WL = 1.55


@pytest.fixture(scope="module")
def small_mesh():
    pytest.importorskip("pygmsh")
    fiber = CircularFiber(2.2, 12, 1.4521, 1.44692, 16, clad_res=16)
    mesh = fiber.make_mesh(order=1)
    return mesh, fiber.assign_IOR()


def test_invalid_mode_raises(small_mesh):
    mesh, IOR = small_mesh
    with pytest.raises(ValueError):
        solve_waveguide_vec(mesh, WL, IOR, Nmax=4, sparse_solve_mode="bogus", verbose=False)


@pytest.mark.skipif(fe_solver.pypardiso is not None, reason="pypardiso is installed")
def test_pardiso_blocked_when_unavailable(small_mesh):
    mesh, IOR = small_mesh
    with pytest.raises(ImportError, match="pardiso"):
        solve_waveguide_vec(mesh, WL, IOR, Nmax=4, sparse_solve_mode="pardiso", verbose=False)


@pytest.mark.skipif(fe_solver._umfpack is None, reason="scikit-umfpack not installed")
def test_umfpack_matches_transform(small_mesh):
    mesh, IOR = small_mesh
    w_t, _, n_t = solve_waveguide_vec(mesh, WL, IOR, Nmax=4, sparse_solve_mode="transform", verbose=False)
    w_u, _, n_u = solve_waveguide_vec(mesh, WL, IOR, Nmax=4, sparse_solve_mode="umfpack", verbose=False)
    assert n_u == n_t
    assert np.allclose(np.sort(w_u.real), np.sort(w_t.real), rtol=1e-8)
