import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.elo import EloModel, expected_score


def test_expected_score_equal_ratings():
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-6


def test_expected_score_higher_rating_wins_more():
    assert expected_score(1700, 1500) > 0.5


def test_update_increases_winner_rating():
    model = EloModel()
    before = model.get_rating("TeamA")
    model.update("TeamA", "TeamB", home_score=2, away_score=0)
    after = model.get_rating("TeamA")
    assert after > before


def test_predict_probabilities_sum_to_one():
    model = EloModel()
    result = model.predict("TeamA", "TeamB")
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-2
