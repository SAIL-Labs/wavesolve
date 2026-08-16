import numpy as np
import pytest

from wavesolve.packing import CIRCLE_PACKINGS, ENCLOSING_RADIUS, circle_packing_positions
from wavesolve.waveguide import FiberBundleLantern

TOL = 1e-9


def test_packings_have_no_overlap_and_fit_enclosing_circle():
    for n, pts in CIRCLE_PACKINGS.items():
        pts = np.array(pts)
        assert pts.shape == (n, 2)
        # all circles (radius 1) inside the enclosing circle
        assert np.max(np.hypot(pts[:, 0], pts[:, 1])) + 1 <= ENCLOSING_RADIUS[n] * (1 + TOL)
        if n > 1:
            d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
            mind = np.min(d[np.triu_indices(n, 1)])
            assert mind >= 2 * (1 - TOL), f"overlap in packing n={n}"


def test_positions_scaling():
    spacing = 125.0
    pos = circle_packing_positions(7, spacing)
    d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])
    mind = np.min(d[np.triu_indices(7, 1)])
    assert mind == pytest.approx(spacing)


def test_unsupported_n_raises():
    for n in (0, 38, -1):
        with pytest.raises(ValueError):
            circle_packing_positions(n, 1.0)


def _lantern(**kw):
    args = dict(r_jack=700, r_fiber_clad=125 / 2, r_core=10.2 / 2,
                n_core=1.4521, n_clad=1.44692, n_jack=1.44692 - 5.e-3,
                core_res=3, clad_res=16, jack_res=16)
    args.update(kw)
    return FiberBundleLantern(**args)


def test_circle_mode_requires_n_fibers():
    with pytest.raises(ValueError):
        _lantern(packing="circle")


def test_invalid_packing_raises():
    with pytest.raises(ValueError):
        _lantern(packing="square", n_fibers=7)


def test_hex_mode_unchanged():
    pl = _lantern(n_rings=1, ring_clad_factors={0: 2.5, 1: 1.0})
    assert pl.n_fibers == 7
    assert sorted(pl.fiber_rings) == [0] + [1] * 6
    assert pl.fiber_positions[0] == (0, 0)
    assert pl.ring_clad_factors[0] == 2.5


def test_circle_mode_19_shells():
    pl = _lantern(packing="circle", n_fibers=19,
                  ring_clad_factors={0: 2.5, 1: 1.0, 2: 1.0})
    assert pl.n_fibers == 19
    counts = {r: pl.fiber_rings.count(r) for r in set(pl.fiber_rings)}
    assert counts == {0: 1, 1: 6, 2: 12}
    # shell 0 is a true center fiber
    center = pl.fiber_positions[pl.fiber_rings.index(0)]
    assert np.hypot(*center) < 1e-6 * pl.spacing
    assert pl.ring_clad_factors[0] == 2.5


def test_circle_mode_5_no_center():
    pl = _lantern(packing="circle", n_fibers=5)
    assert pl.n_fibers == 5
    # optimal 5-packing has no center fiber: a single shell of 5
    assert pl.fiber_rings == [0] * 5
    dists = [np.hypot(x, y) for x, y in pl.fiber_positions]
    assert min(dists) > 0.1 * pl.spacing


def test_circle_mode_taper_target():
    target = 35.0
    pl = _lantern(packing="circle", n_fibers=19, r_target_mmcore_size=target)
    assert pl.bundle_radius == pytest.approx(target, rel=1e-9)


def test_fusion_boundary():
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon as ShPolygon, Point
    from shapely.ops import unary_union
    from wavesolve.waveguide import Polygon2D

    pl = _lantern(packing="circle", n_fibers=19, r_target_mmcore_size=35.0, clad_res=64,
                  ring_clad_factors={0: 2.5, 1: 1.0, 2: 1.0}, fusion_radius=5.0)
    clad = pl.prim2Dgroups[1]
    assert isinstance(clad, Polygon2D)

    fused_poly = ShPolygon(clad.points)
    raw_union = unary_union([Point(p).buffer(pl.r_fiber_clad * pl.ring_clad_factors[r], quad_segs=16)
                             for p, r in zip(pl.fiber_positions, pl.fiber_rings)])
    # closing only adds area (fills notches), never removes it
    assert fused_poly.area > raw_union.area
    assert fused_poly.contains(raw_union.buffer(-0.1))
    # every fiber center is inside the fused region
    for p in pl.fiber_positions:
        assert fused_poly.contains(Point(p))
    # boundary_dist sign convention: negative inside, positive outside
    assert clad.boundary_dist(0, 0) < 0
    assert clad.boundary_dist(1000, 0) > 0


def test_fusion_mesh_smoke():
    pytest.importorskip("shapely")
    pytest.importorskip("pygmsh")
    pl = _lantern(packing="circle", n_fibers=5, r_target_mmcore_size=20.0,
                  fusion_radius=3.0)
    m = pl.make_mesh(order=1, adaptive=True)
    assert m.field_data["fusion_radius"] == 3.0
    assert len(m.points) > 0


def test_circle_mode_mesh_smoke():
    pytest.importorskip("pygmsh")
    pl = _lantern(packing="circle", n_fibers=5, r_target_mmcore_size=20.0)
    m = pl.make_mesh(order=1, adaptive=True)
    assert m.field_data["n_fibers"] == 5
    assert m.field_data["packing"] == "circle"
    assert len(m.points) > 0
