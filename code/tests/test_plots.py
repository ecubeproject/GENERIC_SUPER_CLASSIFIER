import matplotlib.pyplot as plt
import pytest

import plots


def _mark_count(ax):
    return len(ax.lines) + len(ax.collections)


@pytest.mark.parametrize("label", list(plots.PLOTS))
def test_plot_renders_multiclass(iris_result, label):
    fig, ax = plt.subplots()
    try:
        plots.draw(label, ax, iris_result)
        assert _mark_count(ax) >= 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("label", list(plots.PLOTS))
def test_plot_renders_binary(diabetes_result, label):
    fig, ax = plt.subplots()
    try:
        plots.draw(label, ax, diabetes_result)
        assert _mark_count(ax) >= 1
    finally:
        plt.close(fig)


def test_class_color_stable_and_distinct():
    assert plots.class_color(0, 3) == plots.class_color(0, 3)
    assert plots.class_color(0, 3) != plots.class_color(1, 3)
    # many classes -> still returns a colour tuple
    c = plots.class_color(42, 100)
    assert len(c) == 4


def test_cls_label_out_of_range_falls_back():
    assert plots.cls_label(0, ["a", "b"]) == "a"
    assert plots.cls_label(9, ["a", "b"]) == "9"


def test_ovr_indices_binary_vs_multiclass():
    assert plots._ovr_indices([0, 1]) == [1]
    assert plots._ovr_indices([0, 1, 2]) == [0, 1, 2]
