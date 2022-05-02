import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from nndet.evaluator.case import CaseEvaluator, _CaseEvaluator


@pytest.fixture
def evaluator() -> _CaseEvaluator:
    return _CaseEvaluator(
        classes=["class0", "class1"],
        target_class=0,
    )


@pytest.fixture
def evaluator_dummy_metrics() -> _CaseEvaluator:
    return _CaseEvaluator(
        classes=["class0", "class1"],
        target_class=0,
        score_metrics_scalar={"metric0": 0},
        class_metrics_scalar={"metric1": 1},
        score_metrics_curve={"metric2": 2},
        class_metrics_curve={"metric3": 3},
    )


def get_example():
    example_gts_1 = [
        np.array([0, 0, 0, 0, 0]),
        np.array([0, 1, 1, 0, 1]),
        np.array([1]),
        np.array([1, 1, 1]),
    ]
    example_gts_2 = [
        np.array([0, 0, 1]),
        np.array([]),
    ]
    example_pred_score_1 = [
        np.array([0.9, 0.8, 0.2]),
        np.array([0.1, 1.0]),
        np.array([1.0]),
        np.array([]),
    ]
    example_pred_score_2 = [
        np.array([0.9, 0.8, 0.2]),
        np.array([0.2, 0.8]),
    ]
    example_pred_label_1 = [
        np.array([0, 1, 0]),
        np.array([0, 1]),
        np.array([0]),
        np.array([]),
    ]
    example_pred_label_2 = [
        np.array([1, 0, 0]),
        np.array([1, 0]),
    ]
    return (
        [example_gts_1, example_gts_2],
        [example_pred_score_1, example_pred_score_2],
        [example_pred_label_1, example_pred_label_2],
    )


@pytest.fixture
def evaluator_filled() -> _CaseEvaluator:
    _evaluator = _CaseEvaluator(
        classes=["class0", "class1"],
        target_class=0,
    )

    (
        [example_gts_1, example_gts_2],
        [example_pred_score_1, example_pred_score_2],
        [example_pred_label_1, example_pred_label_2],
    ) = get_example()

    _evaluator.run_online_evaluation(
        pred_classes=example_pred_label_1,
        pred_scores=example_pred_score_1,
        gt_classes=example_gts_1,
    )
    _evaluator.run_online_evaluation(
        pred_classes=example_pred_label_2,
        pred_scores=example_pred_score_2,
        gt_classes=example_gts_2,
    )
    return _evaluator


class TestCaseEvaluator:
    def test_init(self, evaluator_dummy_metrics):
        assert evaluator_dummy_metrics.num_classes == 2
        assert evaluator_dummy_metrics.classes == ["class0", "class1"]
        assert evaluator_dummy_metrics.target_class == 0
        assert len(evaluator_dummy_metrics.results_list) == 0
        assert evaluator_dummy_metrics.score_metrics_scalar["metric0"] == 0
        assert evaluator_dummy_metrics.class_metrics_scalar["metric1"] == 1
        assert evaluator_dummy_metrics.score_metrics_curve["metric2"] == 2
        assert evaluator_dummy_metrics.class_metrics_curve["metric3"] == 3

    def test_target_class_error_str(self):
        with pytest.raises(ValueError):
            _CaseEvaluator(
                classes=["class0", "class1"],
                target_class="0",
            )

    def test_target_class_error_none(self):
        with pytest.raises(ValueError):
            _CaseEvaluator(
                classes=["class0", "class1"],
                target_class=None,
            )

    def test_reset(self, evaluator):
        assert len(evaluator.results_list) == 0
        evaluator.results_list["preds"].append(0)

        assert len(evaluator.results_list) == 1
        assert len(evaluator.results_list["preds"]) == 1

        evaluator.reset()

        assert len(evaluator.results_list) == 0

    @pytest.mark.parametrize(
        "pred_classes,pred_scores,gt_classes,expected_case_classes,expected_case_scores",
        [
            (
                np.array([0, 1, 1]),
                np.array([0.1, 0.4, 0.9]),
                np.array([0, 1, 1]),
                np.array([0, 1]),
                np.array([0.1, 0.9]),
            ),
            (
                np.array([1, 1]),
                np.array([0.4, 0.9]),
                np.array([0]),
                np.array([0]),
                np.array([0.0, 0.9]),
            ),
            (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([0.0, 0.0]),
            ),
            (
                np.array([]),
                np.array([]),
                np.array([1]),
                np.array([1]),
                np.array([0.0, 0.0]),
            ),
            (
                np.array([]),
                np.array([]),
                np.array([0]),
                np.array([0]),
                np.array([0.0, 0.0]),
            ),
            (
                np.array([0]),
                np.array([0.9]),
                np.array([]),
                np.array([]),
                np.array([0.9, 0.0]),
            ),
        ],
    )
    def test_run_online_evaluation(
        self,
        evaluator,
        pred_classes,
        pred_scores,
        gt_classes,
        expected_case_classes,
        expected_case_scores,
    ):
        assert len(evaluator.results_list) == 0

        evaluator.run_online_evaluation(
            pred_classes=[pred_classes],
            pred_scores=[pred_scores],
            gt_classes=[gt_classes],
        )

        assert len(evaluator.results_list["case_classes"]) == 1
        evaluator.results_list["case_classes"]
        assert np.allclose(
            evaluator.results_list["case_classes"][0],
            expected_case_classes,
        )

        assert len(evaluator.results_list["case_scores"]) == 1
        assert np.allclose(
            evaluator.results_list["case_scores"][0],
            expected_case_scores,
        )

    @pytest.mark.parametrize(
        "pred_classes,pred_scores,gt_classes,pred_class_multiplier,pred_score_multiplier,gt_multiplier",
        [
            (np.array([0, 1, 1]), np.array([0.1, 0.4, 0.9]), np.array([0]), 2, 1, 1),
            (np.array([0, 1, 1]), np.array([0.1, 0.4, 0.9]), np.array([0]), 1, 2, 1),
            (np.array([0, 1, 1]), np.array([0.1, 0.4, 0.9]), np.array([0]), 1, 1, 2),
            (
                np.array([0, 1, 1]),
                np.array([0.0, 0.1, 0.4, 0.9]),
                np.array([0]),
                1,
                1,
                1,
            ),
            (np.array([0, 1, 1, 1]), np.array([0.1, 0.4, 0.9]), np.array([0]), 1, 1, 1),
        ],
    )
    def test_run_online_evaluation_inconsistency(
        self,
        evaluator,
        pred_classes,
        pred_scores,
        gt_classes,
        pred_class_multiplier,
        pred_score_multiplier,
        gt_multiplier,
    ):
        with pytest.raises(ValueError):
            evaluator.run_online_evaluation(
                pred_classes=[pred_classes] * pred_class_multiplier,
                pred_scores=[pred_scores] * pred_score_multiplier,
                gt_classes=[gt_classes] * gt_multiplier,
            )

    def test_class_count(self, evaluator_filled):
        # target class = 0
        count = evaluator_filled.class_count()
        assert count == {-1: 1, 0: 3, 1: 4}

    def test_aggregate_classes(self, evaluator_filled):
        aggregated_classes = evaluator_filled.aggregate_classes()
        assert np.allclose(aggregated_classes, [1, 1, 0, 0, 1, 0])

    def test_aggregate_prdictions(self, evaluator_filled):
        agg_pred_score, agg_pred_class = evaluator_filled.aggregate_prdictions(
            threshold=0.5
        )
        assert np.allclose(agg_pred_score, [0.9, 0.1, 1.0, 0.0, 0.8, 0.8])
        assert np.allclose(agg_pred_class, [1, 0, 1, 0, 1, 1])

    def test_integration(self):
        _evaluator = CaseEvaluator.create(
            classes=["class0", "class1"],
            target_class=0,
        )

        (
            [example_gts_1, example_gts_2],
            [example_pred_score_1, example_pred_score_2],
            [example_pred_label_1, example_pred_label_2],
        ) = get_example()

        _evaluator.run_online_evaluation(
            pred_classes=example_pred_label_1,
            pred_scores=example_pred_score_1,
            gt_classes=example_gts_1,
        )
        _evaluator.run_online_evaluation(
            pred_classes=example_pred_label_2,
            pred_scores=example_pred_score_2,
            gt_classes=example_gts_2,
        )

        results_scalar, results_curve = _evaluator.finish_online_evaluation()
        # check all results present
        assert "auc_case" in results_scalar
        assert "ap_case" in results_scalar

        assert "roc_curve" in results_curve

        for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            for key in ("f1_case", "prec_case", "rec_case", "acc_case"):
                k = f"{key}_th={th:0.3f}"
                assert k in results_scalar

            for key in ("cfm_case",):
                k = f"{key}_th={th:0.3f}"
                assert k in results_curve

        # check scalar results
        exp_gt = [1, 1, 0, 0, 1, 0]
        exp_pred = [0.9, 0.1, 1.0, 0.0, 0.8, 0.8]
        assert np.isclose(results_scalar["auc_case"], roc_auc_score(exp_gt, exp_pred))
        assert np.isclose(
            results_scalar["ap_case"], average_precision_score(exp_gt, exp_pred)
        )

        # at th=0.500
        # gt [1, 1, 0, 0, 1, 0]
        # p  [1, 0, 1, 0, 1, 1]
        assert np.isclose(results_scalar["f1_case_th=0.500"], (2 / (2 + 0.5 * (2 + 1))))
        assert np.isclose(results_scalar["prec_case_th=0.500"], 2 / 4)
        assert np.isclose(results_scalar["rec_case_th=0.500"], 2 / 3)
        assert np.isclose(results_scalar["acc_case_th=0.500"], 3 / 6)

        # check confusion matrix
        assert results_curve["cfm_case_th=0.500"][0, 0] == 1
        assert results_curve["cfm_case_th=0.500"][0, 1] == 2
        assert results_curve["cfm_case_th=0.500"][1, 0] == 1
        assert results_curve["cfm_case_th=0.500"][1, 1] == 2
