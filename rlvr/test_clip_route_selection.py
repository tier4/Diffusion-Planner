"""Route selection for the clip renderer.

Start indices collide between the recorded drives an eval spans, so a pick from
``find_disagreeing_starts`` is only meaningful together with its route. These tests pin the
behaviour that a multi-route root must be disambiguated explicitly rather than silently
rendering whichever route sorts first.
"""

import pytest

from rlvr.autoresearch.tools.render_recovery_clip import select_route

TWO = {"drive-B_00000000": ["b.npz"], "drive-A_00000000": ["a.npz"]}
ONE = {"drive-A_00000000": ["a.npz"]}


def test_single_route_needs_no_flag():
    assert select_route(ONE, None) == "drive-A_00000000"


def test_two_routes_without_route_flag_is_refused():
    """The regression: this used to return the lexicographically first route, so a pick
    belonging to drive B rendered drive A at the same start index."""
    with pytest.raises(SystemExit) as e:
        select_route(TWO, None)
    assert "--route" in str(e.value)


def test_two_routes_honour_the_requested_route():
    assert select_route(TWO, "drive-B_00000000") == "drive-B_00000000"
    assert select_route(TWO, "drive-A_00000000") == "drive-A_00000000"


def test_unknown_route_is_refused_and_lists_what_exists():
    with pytest.raises(SystemExit) as e:
        select_route(TWO, "drive-C_00000000")
    msg = str(e.value)
    assert "drive-C_00000000" in msg and "drive-A_00000000" in msg
