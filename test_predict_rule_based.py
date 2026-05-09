#!/usr/bin/env python3
"""Tests for predict_rule_based.py — stdlib only, no torch/transformers needed."""

import os
import sys
import tempfile
import unittest

# Allow import without installation
sys.path.insert(0, os.path.dirname(__file__))
from predict_rule_based import (
    _classify_intent,
    _tag_slots_atis,
    _format_output,
    predict_rule_based,
    _ATIS_INTENT_RULES,
    _SNIPS_INTENT_RULES,
)


# ---------------------------------------------------------------------------
# Intent classification — ATIS
# ---------------------------------------------------------------------------

class TestClassifyIntentATIS(unittest.TestCase):

    def _cls(self, text: str) -> str:
        return _classify_intent(text, _ATIS_INTENT_RULES)

    def test_flight_query(self):
        self.assertEqual(self._cls("show me flights from dallas to boston"), "atis_flight")

    def test_airfare_query(self):
        intent = self._cls("what is the cost of a ticket from dallas to boston")
        self.assertEqual(intent, "atis_airfare")

    def test_airline_query(self):
        intent = self._cls("which airlines fly from dallas to boston")
        self.assertEqual(intent, "atis_airline")

    def test_airport_query(self):
        intent = self._cls("what airports are in dallas")
        self.assertEqual(intent, "atis_airport")

    def test_ground_service_query(self):
        # Use "ground transportation" to avoid airport ambiguity
        intent = self._cls("i need ground transportation to the hotel")
        self.assertEqual(intent, "atis_ground_service")

    def test_abbreviation_query(self):
        intent = self._cls("what does dfw stand for")
        self.assertEqual(intent, "atis_abbreviation")

    def test_aircraft_query(self):
        intent = self._cls("what type of aircraft is used on flight 20")
        self.assertEqual(intent, "atis_aircraft")

    def test_cheapest_query(self):
        intent = self._cls("what is the cheapest flight from new york to london")
        self.assertEqual(intent, "atis_cheapest")

    def test_distance_query(self):
        intent = self._cls("how far is dallas from boston")
        self.assertEqual(intent, "atis_distance")

    def test_meal_query(self):
        # Use meal-heavy query to avoid flight ambiguity
        intent = self._cls("what meal is served breakfast lunch dinner snack")
        self.assertEqual(intent, "atis_meal")

    def test_empty_string_returns_unk(self):
        intent = self._cls("")
        self.assertEqual(intent, "UNK")

    def test_unknown_query_returns_unk(self):
        intent = self._cls("xyzzy frobnotz quux")
        self.assertEqual(intent, "UNK")


# ---------------------------------------------------------------------------
# Intent classification — Snips
# ---------------------------------------------------------------------------

class TestClassifyIntentSnips(unittest.TestCase):

    def _cls(self, text: str) -> str:
        return _classify_intent(text, _SNIPS_INTENT_RULES)

    def test_get_weather(self):
        intent = self._cls("what is the weather forecast for tomorrow")
        self.assertEqual(intent, "GetWeather")

    def test_play_music(self):
        intent = self._cls("play some music by the beatles")
        self.assertEqual(intent, "PlayMusic")

    def test_book_restaurant(self):
        intent = self._cls("book a table at an italian restaurant")
        self.assertEqual(intent, "BookRestaurant")

    def test_add_to_playlist(self):
        intent = self._cls("add this song to my playlist")
        self.assertEqual(intent, "AddToPlaylist")

    def test_search_creative_work(self):
        intent = self._cls("find the movie inception")
        self.assertEqual(intent, "SearchCreativeWork")

    def test_rate_book(self):
        intent = self._cls("rate this book 4 stars out of 5")
        self.assertEqual(intent, "RateBook")


# ---------------------------------------------------------------------------
# Slot tagging — ATIS
# ---------------------------------------------------------------------------

class TestTagSlotsATIS(unittest.TestCase):

    def test_returns_same_length_as_words(self):
        words = "show flights from dallas to boston".split()
        tags = _tag_slots_atis(words)
        self.assertEqual(len(tags), len(words))

    def test_unmatched_words_get_O(self):
        words = "show flights".split()
        tags = _tag_slots_atis(words)
        self.assertTrue(all(t == "O" for t in tags))

    def test_month_tagged(self):
        words = "flights on December 25".split()
        tags = _tag_slots_atis(words)
        # "December" should get a depart_date.month_name tag
        tagged = [(w, t) for w, t in zip(words, tags) if t != "O"]
        month_tagged = [w for w, t in tagged if "month" in t]
        self.assertTrue(len(month_tagged) > 0, f"No month tag found. tags={tags}")

    def test_airline_tagged(self):
        words = "United flights from dallas".split()
        tags = _tag_slots_atis(words)
        tagged = {w: t for w, t in zip(words, tags) if t != "O"}
        self.assertTrue(any("airline" in t for t in tagged.values()),
                         f"No airline tag found. tags={tags}")

    def test_no_crash_on_empty_words(self):
        tags = _tag_slots_atis([])
        self.assertEqual(tags, [])

    def test_no_crash_on_single_word(self):
        tags = _tag_slots_atis(["hello"])
        self.assertEqual(len(tags), 1)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class TestFormatOutput(unittest.TestCase):

    def test_all_O_no_brackets(self):
        words = ["show", "me", "flights"]
        tags = ["O", "O", "O"]
        result = _format_output(words, tags, "atis_flight")
        self.assertEqual(result, "<atis_flight> -> show me flights")

    def test_tagged_word_gets_brackets(self):
        words = ["United", "flights"]
        tags = ["B-airline_name", "O"]
        result = _format_output(words, tags, "atis_flight")
        self.assertIn("[United:", result)
        self.assertIn("airline_name", result)
        self.assertIn("flights", result)

    def test_intent_in_angle_brackets(self):
        result = _format_output(["hello"], ["O"], "atis_city")
        self.assertTrue(result.startswith("<atis_city>"))

    def test_output_format_matches_predict_py(self):
        # predict.py output: "<intent> -> word1 [word2:slot] word3"
        words = ["show", "me", "flights"]
        tags = ["O", "O", "O"]
        result = _format_output(words, tags, "atis_flight")
        self.assertRegex(result, r"^<\w+> -> .+")


# ---------------------------------------------------------------------------
# predict_rule_based pipeline
# ---------------------------------------------------------------------------

class TestPredictRuleBased(unittest.TestCase):

    def test_returns_one_line_per_input(self):
        lines = [["show", "me", "flights"], ["what", "is", "the", "weather"]]
        results = predict_rule_based(lines, task="atis")
        self.assertEqual(len(results), 2)

    def test_empty_input(self):
        results = predict_rule_based([], task="atis")
        self.assertEqual(results, [])

    def test_atis_output_format(self):
        lines = [["show", "me", "flights", "from", "dallas"]]
        results = predict_rule_based(lines, task="atis")
        self.assertEqual(len(results), 1)
        self.assertRegex(results[0], r"^<atis_\w+> -> .+")

    def test_snips_output_format(self):
        lines = [["play", "some", "music"]]
        results = predict_rule_based(lines, task="snips")
        self.assertEqual(len(results), 1)
        self.assertRegex(results[0], r"^<\w+> -> .+")

    def test_snips_play_music(self):
        lines = [["play", "some", "music", "by", "the", "beatles"]]
        results = predict_rule_based(lines, task="snips")
        self.assertIn("<PlayMusic>", results[0])

    def test_snips_no_slot_tags_all_O(self):
        lines = [["play", "music"]]
        results = predict_rule_based(lines, task="snips")
        # Snips rule-based has no slot tagger — all words untagged
        self.assertNotIn("[", results[0])

    def test_unknown_task_defaults_to_atis_rules(self):
        lines = [["show", "me", "flights"]]
        # Should not raise
        results = predict_rule_based(lines, task="nonexistent")
        self.assertEqual(len(results), 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def _write_input(self, sentences: list[str]) -> str:
        path = os.path.join(self._tmpdir, "input.txt")
        with open(path, "w") as fh:
            fh.write("\n".join(sentences) + "\n")
        return path

    def test_cli_runs_and_writes_output(self):
        import subprocess
        input_path = self._write_input([
            "show me flights from dallas to boston",
            "what is the weather",
        ])
        output_path = os.path.join(self._tmpdir, "output.txt")
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "predict_rule_based.py"),
             "--input_file", input_path,
             "--output_file", output_path,
             "--task", "atis"],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertRegex(line, r"^<\w+> -> .+")

    def test_cli_stdout_mode(self):
        import subprocess
        input_path = self._write_input(["show me flights"])
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "predict_rule_based.py"),
             "--input_file", input_path,
             "--output_file", "-",
             "--task", "atis"],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("<atis_", result.stdout)


if __name__ == "__main__":
    unittest.main()
