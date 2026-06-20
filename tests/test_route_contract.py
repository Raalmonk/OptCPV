from optcpv.models import BBox, Point
from optcpv.route_contract import orthogonalize_route, route_crosses_keepout
from optcpv.segments import is_axis_aligned


def test_axis_aligned_route_detours_keepout() -> None:
    keepout = BBox(1.0, 0.0, 2.0, 2.0)

    route = orthogonalize_route([Point(0.0, 1.0), Point(4.0, 1.0)], [keepout])

    assert route[0] == Point(0.0, 1.0)
    assert route[-1] == Point(4.0, 1.0)
    assert len(route) > 2
    assert all(is_axis_aligned(start, end) for start, end in zip(route, route[1:]))
    assert not route_crosses_keepout(route, keepout)
