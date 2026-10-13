import numpy as np

from scenario_generation.canvas import Figure, finite_runs, nice_ticks, scene_panel
from scenario_generation.colormaps import colormap, map_values


def test_finite_runs_split_a_polyline_at_non_finite_points():
    points = np.array([[0, 0], [1, 0], [np.nan, 0], [2, 0], [3, np.inf], [4, 0], [5, 0]])
    runs = finite_runs(points)
    assert [run.tolist() for run in runs] == [[[0, 0], [1, 0]], [[2, 0]], [[4, 0], [5, 0]]]


def test_nice_ticks_land_on_round_steps_inside_the_range():
    np.testing.assert_allclose(nice_ticks(-50.0, 50.0), [-40, -20, 0, 20, 40])
    ticks = nice_ticks(0.0, 1.0)
    assert ticks[0] >= 0.0 and ticks[-1] <= 1.0 and len(ticks) >= 4
    np.testing.assert_allclose(np.diff(ticks), np.diff(ticks)[0])


def test_non_finite_points_draw_without_error(tmp_path):
    panel = scene_panel((-10.0, 10.0), (-10.0, 10.0))
    line = np.array([[0, 0], [5, 0], [np.nan, np.nan], [6, 6], [np.inf, 0]])
    panel.lines([line], "#ff0000", dash="dash")
    panel.markers(line, "*", 8, "#0000ff")
    panel.text((np.nan, 0.0), "skipped")
    path = tmp_path / "scene.png"
    panel.save(path)
    assert path.stat().st_size > 0


def test_autoscaled_chart_composes_at_the_figure_size():
    figure = Figure((4.0, 3.0), 50.0)
    ((left, right),) = figure.grid(1, 2)
    left.hist(np.random.default_rng(0).normal(size=100), bins=10)
    right.markers(np.array([[0, 0], [1, 1]]), values=np.array([0.0, 1.0]), cmap="RdYlGn")
    right.colorbar("RdYlGn", 0.0, 1.0, "value")
    assert figure.array().shape == (150, 200, 3)


def test_colormaps_scale_values_to_their_ends():
    colors = map_values(np.array([0.0, 10.0]), "viridis")
    np.testing.assert_array_equal(colors[0], colormap("viridis")[0])
    np.testing.assert_array_equal(colors[1], colormap("viridis")[-1])
    np.testing.assert_array_equal(colormap("RdYlGn_r")[0], colormap("RdYlGn")[-1])
