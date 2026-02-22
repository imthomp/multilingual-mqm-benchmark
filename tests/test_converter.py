"""Unit tests for span-level MQM annotation → sentence-level score conversion."""

import pandas as pd
import pytest

from mqmbench.data.converter import annotations_to_sentence_scores, SEVERITY_WEIGHTS


def _make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "source": "Hello world.",
        "hypothesis": "Hola mundo.",
        "reference": "Hola mundo.",
        "lang": "es",
        "error_start": None,
        "error_end": None,
    }
    records = []
    for i, row in enumerate(rows):
        r = {**defaults, "segment_id": i, **row}
        records.append(r)
    return pd.DataFrame(records)


class TestAnnotationsToSentenceScores:

    def test_no_errors_gives_zero_penalty(self):
        df = _make_df([{"error_type": "no_error", "severity": "neutral"}])
        result = annotations_to_sentence_scores(df)
        assert result.iloc[0]["error_penalty"] == 0.0
        assert result.iloc[0]["quality_score"] == pytest.approx(1.0)

    def test_single_major_error(self):
        df = _make_df([{"error_type": "mistranslation", "severity": "major"}])
        result = annotations_to_sentence_scores(df)
        assert result.iloc[0]["error_penalty"] == pytest.approx(SEVERITY_WEIGHTS["major"])
        assert result.iloc[0]["major_errors"] == 1
        assert result.iloc[0]["minor_errors"] == 0

    def test_single_minor_error(self):
        df = _make_df([{"error_type": "grammar", "severity": "minor"}])
        result = annotations_to_sentence_scores(df)
        assert result.iloc[0]["error_penalty"] == pytest.approx(SEVERITY_WEIGHTS["minor"])
        assert result.iloc[0]["minor_errors"] == 1

    def test_multiple_errors_same_segment(self):
        df = _make_df([
            {"segment_id": 0, "error_type": "mistranslation", "severity": "major"},
            {"segment_id": 0, "error_type": "grammar", "severity": "minor"},
        ])
        result = annotations_to_sentence_scores(df)
        assert len(result) == 1
        expected_penalty = SEVERITY_WEIGHTS["major"] + SEVERITY_WEIGHTS["minor"]
        assert result.iloc[0]["error_penalty"] == pytest.approx(expected_penalty)
        assert result.iloc[0]["num_errors"] == 2

    def test_two_segments_scored_independently(self):
        df = _make_df([
            {"segment_id": 0, "error_type": "no_error", "severity": "neutral"},
            {"segment_id": 1, "error_type": "mistranslation", "severity": "major"},
        ])
        result = annotations_to_sentence_scores(df).sort_values("segment_id").reset_index(drop=True)
        assert len(result) == 2
        assert result.iloc[0]["error_penalty"] == 0.0
        assert result.iloc[1]["error_penalty"] == pytest.approx(SEVERITY_WEIGHTS["major"])

    def test_normalized_score_is_non_positive(self):
        df = _make_df([{"error_type": "mistranslation", "severity": "major"}])
        result = annotations_to_sentence_scores(df)
        assert result.iloc[0]["normalized_score"] <= 0.0

    def test_quality_score_between_zero_and_one(self):
        df = _make_df([
            {"segment_id": 0, "error_type": "no_error", "severity": "neutral"},
            {"segment_id": 1, "error_type": "mistranslation", "severity": "major"},
            {"segment_id": 2, "error_type": "mistranslation", "severity": "major",
             "hypothesis": "Bad bad bad bad bad bad bad bad bad."},
        ])
        result = annotations_to_sentence_scores(df)
        for _, row in result.iterrows():
            assert 0.0 <= row["quality_score"] <= 1.0

    def test_empty_input_returns_empty_dataframe(self):
        df = pd.DataFrame(columns=[
            "segment_id", "source", "hypothesis", "reference", "lang",
            "error_type", "severity", "error_start", "error_end"
        ])
        result = annotations_to_sentence_scores(df)
        assert len(result) == 0

    def test_unknown_severity_raises(self):
        df = _make_df([{"error_type": "grammar", "severity": "critical"}])
        with pytest.raises(ValueError, match="Unknown severity"):
            annotations_to_sentence_scores(df)

    def test_perfect_segment_has_quality_one(self):
        df = _make_df([{"error_type": "no_error", "severity": "neutral"}])
        result = annotations_to_sentence_scores(df)
        assert result.iloc[0]["quality_score"] == pytest.approx(1.0)

    def test_worse_segment_has_lower_quality(self):
        df = _make_df([
            {"segment_id": 0, "error_type": "grammar", "severity": "minor"},
            {"segment_id": 1, "error_type": "mistranslation", "severity": "major"},
        ])
        result = annotations_to_sentence_scores(df).sort_values("segment_id").reset_index(drop=True)
        assert result.iloc[0]["quality_score"] > result.iloc[1]["quality_score"]
