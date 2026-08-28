import pytest

import pipeline


def test_empty_leaderboard():
    lb = pipeline.Leaderboard()
    assert len(lb) == 0
    assert lb.best() is None
    assert "no models" in lb.table()


def test_add_trainresult_and_cvresult(iris_df):
    lb = pipeline.Leaderboard()
    tr = pipeline.train_and_evaluate(
        iris_df, "species", "Random Forest", pipeline.DEFAULT_PARAMS["Random Forest"])
    cv = pipeline.cross_validate_model(
        iris_df, "species", "KNN", pipeline.DEFAULT_PARAMS["KNN"], k=4)
    lb.add(tr)
    lb.add(cv, label="KNN (4-fold)")

    assert len(lb) == 2
    assert lb.rows[0].source == "holdout"
    assert lb.rows[1].source == "cv-4"
    assert lb.rows[1].label == "KNN (4-fold)"
    for r in lb.rows:
        assert 0.0 <= r.metrics["accuracy"] <= 1.0
        assert 0.0 <= r.metrics["f1_macro"] <= 1.0


def test_table_sorted_by_metric(iris_df):
    lb = pipeline.Leaderboard()
    lb.add(pipeline.train_and_evaluate(
        iris_df, "species", "Random Forest", pipeline.DEFAULT_PARAMS["Random Forest"]))
    lb.add(pipeline.train_and_evaluate(
        iris_df, "species", "Gaussian Naive Bayes",
        pipeline.DEFAULT_PARAMS["Gaussian Naive Bayes"]))

    best = lb.best("accuracy")
    assert best is not None
    accs = [r.metrics["accuracy"] for r in lb.rows]
    assert best.metrics["accuracy"] == max(accs)

    table = lb.table(sort_by="accuracy")
    first_data_line = table.splitlines()[2]
    assert best.label[:24] in first_data_line


def test_add_rejects_bad_type():
    lb = pipeline.Leaderboard()
    with pytest.raises(TypeError):
        lb.add({"accuracy": 0.9})


def test_row_index_is_stable_insertion_order(iris_df):
    lb = pipeline.Leaderboard()
    for name in ("KNN", "Decision Tree", "Random Forest"):
        lb.add(pipeline.train_and_evaluate(
            iris_df, "species", name, pipeline.DEFAULT_PARAMS[name]))
    assert [r.index for r in lb.rows] == [1, 2, 3]
    # sorting the table does not renumber the rows
    lb.table(sort_by="f1_macro")
    assert [r.index for r in lb.rows] == [1, 2, 3]
