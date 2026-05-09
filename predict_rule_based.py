#!/usr/bin/env python3
"""
predict_rule_based.py — Rule-based fallback predictor for JointBERT.

Provides a lightweight inference path when a trained BERT model is not
available (no GPU, no downloaded checkpoint, CI environments, quick demos).

Covers the ATIS intent taxonomy using weighted token matching — same 21 intent
classes as the ATIS dataset, same output format as predict.py. Slot filling
uses regex-based BIO tagging for the most common ATIS slot types.

Why this exists:
  - predict.py requires torch + transformers + a trained model directory
  - This file requires only stdlib (Python 3.6+)
  - Useful for: smoke tests, prototyping, environments without GPU

Output format: identical to predict.py
    <intent> -> word1 [word2:slot_type] word3 ...

Usage:
    python3 predict_rule_based.py --input_file sample_pred_in.txt \
                                   --output_file sample_pred_out.txt \
                                   --task atis

    python3 predict_rule_based.py --input_file sample_pred_in.txt \
                                   --output_file - \
                                   --task snips
"""

import argparse
import re
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# ATIS intent taxonomy (21 classes from data/atis/intent_label.txt)
# Maps each intent to weighted keyword signals.
# ---------------------------------------------------------------------------

_ATIS_INTENT_RULES: dict[str, list[tuple[str, float]]] = {
    "atis_flight": [
        ("flight", 3.0), ("fly", 2.0), ("flights", 3.0), ("flying", 2.0),
        ("travel", 1.5), ("trip", 1.5), ("route", 1.5),
        ("from", 0.5), ("to", 0.5), ("nonstop", 1.0), ("direct", 1.0),
    ],
    "atis_airfare": [
        ("fare", 3.0), ("fares", 3.0), ("cost", 2.0), ("price", 2.0),
        ("prices", 2.0), ("costs", 2.0), ("ticket", 2.0), ("tickets", 2.0),
        ("how much", 2.5), ("cheap", 1.5), ("expensive", 1.5),
        ("cheapest", 2.0), ("lowest", 1.5),
    ],
    "atis_ground_service": [
        ("ground", 2.5), ("transportation", 2.5), ("taxi", 2.5), ("cab", 2.0),
        ("bus", 2.0), ("shuttle", 2.0), ("limousine", 2.0), ("limo", 2.0),
        ("car", 1.5), ("rental", 2.0), ("rent", 1.5),
        ("from airport", 1.5), ("to airport", 1.5),
    ],
    "atis_airline": [
        ("airline", 3.0), ("airlines", 3.0), ("carrier", 2.0), ("carriers", 2.0),
        ("which airline", 2.5), ("what airline", 2.5),
    ],
    "atis_abbreviation": [
        ("abbreviation", 3.0), ("abbreviate", 3.0), ("stand for", 2.5),
        ("stands for", 2.5), ("mean", 2.0), ("means", 2.0), ("code", 2.0),
        ("what does", 1.0), ("what is", 1.0),
    ],
    "atis_aircraft": [
        ("aircraft", 3.0), ("plane", 2.5), ("planes", 2.5), ("jet", 2.0),
        ("equipment", 2.0), ("boeing", 2.0), ("airbus", 2.0), ("type", 1.0),
        ("model", 1.5),
    ],
    "atis_flight_time": [
        ("time", 2.0), ("times", 2.0), ("schedule", 2.5), ("schedules", 2.5),
        ("depart", 2.0), ("departs", 2.0), ("arrive", 2.0), ("arrives", 2.0),
        ("departure", 2.5), ("arrival", 2.5), ("duration", 2.5),
        ("how long", 2.5), ("when", 1.5),
    ],
    "atis_city": [
        ("city", 3.0), ("cities", 3.0), ("serve", 2.0), ("serves", 2.0),
        ("what cities", 2.5), ("which cities", 2.5), ("list cities", 2.0),
        ("destination", 1.5), ("destinations", 1.5),
    ],
    "atis_airport": [
        ("airport", 3.0), ("airports", 3.0), ("terminal", 2.0), ("terminals", 2.0),
        ("gate", 2.0), ("which airport", 2.5), ("what airport", 2.5),
    ],
    "atis_ground_fare": [
        ("ground", 1.5), ("fare", 1.5), ("cost", 1.5), ("price", 1.5),
        ("taxi fare", 3.0), ("cab fare", 3.0), ("shuttle cost", 3.0),
        ("how much", 1.5), ("transportation cost", 3.0),
    ],
    "atis_quantity": [
        ("how many", 3.0), ("number of", 2.5), ("count", 2.0), ("total", 2.0),
        ("quantity", 3.0), ("many flights", 2.5),
    ],
    "atis_flight_no": [
        ("flight number", 3.0), ("flight no", 3.0), ("flight #", 3.0),
        ("number", 1.5), ("what number", 2.0), ("which number", 2.0),
    ],
    "atis_distance": [
        ("distance", 3.0), ("far", 2.0), ("how far", 3.0), ("miles", 2.5),
        ("kilometers", 2.5), ("km", 2.0),
    ],
    "atis_capacity": [
        ("capacity", 3.0), ("seats", 2.5), ("seat", 2.0), ("passengers", 2.5),
        ("how many seats", 3.0), ("size", 1.5),
    ],
    "atis_flight#atis_airfare": [
        ("flight", 1.5), ("fare", 1.5), ("flights and fares", 3.0),
        ("flights with fares", 3.0),
    ],
    "atis_airline#atis_flight_no": [
        ("airline", 1.5), ("flight number", 1.5), ("airline flight number", 3.0),
    ],
    "atis_aircraft#atis_flight#atis_flight_no": [
        ("aircraft", 1.0), ("flight", 1.0), ("flight number", 1.0),
        ("aircraft type", 2.0), ("plane flight", 2.0),
    ],
    "atis_ground_service#atis_ground_fare": [
        ("ground transportation", 2.0), ("cost", 1.0), ("ground service cost", 3.0),
    ],
    "atis_cheapest": [
        ("cheapest", 3.0), ("cheapest flight", 3.0), ("lowest fare", 3.0),
        ("least expensive", 3.0), ("best price", 2.5), ("most affordable", 2.5),
    ],
    "atis_meal": [
        ("meal", 3.0), ("meals", 3.0), ("food", 2.5), ("snack", 2.0),
        ("dinner", 2.0), ("lunch", 2.0), ("breakfast", 2.0), ("eat", 1.5),
    ],
    "atis_restriction": [
        ("restriction", 3.0), ("restrictions", 3.0), ("limit", 2.0), ("limits", 2.0),
        ("require", 2.0), ("requirement", 2.5), ("condition", 2.0),
        ("baggage", 2.0), ("luggage", 2.0),
    ],
}

# Snips intents (7 classes from data/snips/intent_label.txt)
_SNIPS_INTENT_RULES: dict[str, list[tuple[str, float]]] = {
    "AddToPlaylist": [
        ("add", 2.0), ("put", 1.5), ("include", 1.5), ("playlist", 3.0),
        ("queue", 2.5), ("list", 1.5),
    ],
    "BookRestaurant": [
        ("book", 2.5), ("reserve", 2.5), ("reservation", 3.0), ("restaurant", 3.0),
        ("table", 2.0), ("dinner", 1.5), ("lunch", 1.5),
    ],
    "GetWeather": [
        ("weather", 3.0), ("forecast", 3.0), ("temperature", 2.5), ("rain", 2.0),
        ("sunny", 2.0), ("cloudy", 2.0), ("hot", 1.5), ("cold", 1.5),
        ("will it", 1.5), ("what's the weather", 3.0),
    ],
    "PlayMusic": [
        ("play", 2.5), ("music", 2.5), ("song", 2.5), ("songs", 2.5),
        ("listen", 2.0), ("album", 2.0), ("artist", 2.0), ("stream", 2.0),
    ],
    "RateBook": [
        ("rate", 3.0), ("rating", 3.0), ("review", 2.5), ("book", 2.0),
        ("novel", 2.0), ("stars", 2.5), ("out of", 1.5), ("give", 1.5),
    ],
    "SearchCreativeWork": [
        ("find", 2.0), ("search", 2.0), ("look", 1.5), ("movie", 2.5),
        ("film", 2.5), ("show", 2.0), ("series", 2.0), ("episode", 2.0),
        ("book", 1.5), ("game", 2.0),
    ],
    "SearchScreeningEvent": [
        ("screening", 3.0), ("showing", 3.0), ("cinema", 3.0), ("theater", 2.5),
        ("theatre", 2.5), ("playing", 2.0), ("when is", 1.5), ("movie time", 2.5),
    ],
}

_TASK_INTENTS = {
    "atis": _ATIS_INTENT_RULES,
    "snips": _SNIPS_INTENT_RULES,
}

# ---------------------------------------------------------------------------
# ATIS slot patterns — BIO tagging without a model
# Maps slot label → regex pattern
# ---------------------------------------------------------------------------

_ATIS_SLOT_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Airport codes: 3 uppercase letters
    ("B-fromloc.airport_code", re.compile(r"\b([A-Z]{3})\b(?=\s+to\b|\s+and\b)")),
    ("B-toloc.airport_code", re.compile(r"\bto\s+([A-Z]{3})\b")),
    # City names after "from" / "to"
    ("B-fromloc.city_name", re.compile(
        r"\b(?:from|leaving|departing)\s+([A-Za-z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
    )),
    ("B-toloc.city_name", re.compile(
        r"\b(?:to|arriving|arriving\s+in)\s+([A-Za-z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
    )),
    # Dates
    ("B-depart_date.month_name", re.compile(
        r"\b(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\b", re.IGNORECASE
    )),
    ("B-depart_date.day_number", re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\b")),
    # Times
    ("B-depart_time.time", re.compile(r"\b(\d{1,2}:\d{2}(?:\s*[ap]m)?)\b", re.IGNORECASE)),
    ("B-depart_time.period_of_day", re.compile(
        r"\b(morning|afternoon|evening|night|noon|midnight)\b", re.IGNORECASE
    )),
    # Airline names
    ("B-airline_name", re.compile(
        r"\b(United|Delta|American|Southwest|JetBlue|Alaska|Spirit|Frontier|"
        r"Continental|US Airways|TWA|Pan Am)\b", re.IGNORECASE
    )),
    # Aircraft types
    ("B-aircraft_code", re.compile(r"\b(B\d{3}|A\d{3}|737|747|757|767|777|787|"
                                    r"320|330|340|350|380)\b")),
    # Fare classes
    ("B-fare_basis_code", re.compile(r"\b(first class|business class|economy|coach)\b",
                                      re.IGNORECASE)),
    # Meal types
    ("B-meal_description", re.compile(
        r"\b(breakfast|lunch|dinner|snack|meal|food)\b", re.IGNORECASE
    )),
]

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def _classify_intent(
    text: str,
    intent_rules: dict[str, list[tuple[str, float]]],
) -> str:
    """Score each intent and return the highest-scoring one.

    Falls back to UNK when no intent scores above zero.
    """
    text_lower = text.lower()
    tokens = set(_tokenize(text))

    best_intent = "UNK"
    best_score = 0.0

    for intent, rules in intent_rules.items():
        score = 0.0
        for keyword, weight in rules:
            if " " in keyword:
                if keyword.lower() in text_lower:
                    score += weight
            else:
                if keyword.lower() in tokens:
                    score += weight
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent


# ---------------------------------------------------------------------------
# Slot tagging (ATIS)
# ---------------------------------------------------------------------------

def _tag_slots_atis(words: list[str]) -> list[str]:
    """Return a BIO slot tag for each word in the sentence.

    Uses regex matching over the joined sentence, then maps character spans
    back to word indices. Unmatched words get 'O'.
    """
    tags = ["O"] * len(words)
    text = " ".join(words)

    # Build word start positions
    word_starts = []
    pos = 0
    for word in words:
        word_starts.append(pos)
        pos += len(word) + 1  # +1 for space

    def _word_idx(char_pos: int) -> Optional[int]:
        for i, start in enumerate(word_starts):
            end = start + len(words[i])
            if start <= char_pos < end:
                return i
        return None

    for slot_label, pattern in _ATIS_SLOT_PATTERNS:
        for m in pattern.finditer(text):
            start_idx = _word_idx(m.start(1) if m.lastindex else m.start())
            if start_idx is not None and tags[start_idx] == "O":
                tags[start_idx] = slot_label
                # Mark continuation words as I- tags
                end_text = m.group(1) if m.lastindex else m.group()
                for extra in range(1, len(end_text.split())):
                    if start_idx + extra < len(tags) and tags[start_idx + extra] == "O":
                        i_label = slot_label.replace("B-", "I-", 1)
                        tags[start_idx + extra] = i_label

    return tags


# ---------------------------------------------------------------------------
# Output formatting (mirrors predict.py)
# ---------------------------------------------------------------------------

def _format_output(words: list[str], slot_tags: list[str], intent: str) -> str:
    line = ""
    for word, tag in zip(words, slot_tags):
        if tag == "O":
            line += word + " "
        else:
            # Strip B-/I- prefix to get slot type name
            slot_type = re.sub(r"^[BI]-", "", tag)
            line += f"[{word}:{slot_type}] "
    return f"<{intent}> -> {line.strip()}"


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

def predict_rule_based(
    lines: list[list[str]],
    task: str = "atis",
) -> list[str]:
    """Run rule-based prediction on tokenized input lines.

    Args:
        lines: list of word lists (same format as predict.py read_input_file)
        task: "atis" or "snips"

    Returns:
        list of formatted output strings, one per input line
    """
    intent_rules = _TASK_INTENTS.get(task, _ATIS_INTENT_RULES)
    results = []

    for words in lines:
        text = " ".join(words)
        intent = _classify_intent(text, intent_rules)

        if task == "atis":
            slot_tags = _tag_slots_atis(words)
        else:
            # Snips: no slot tagging in rule-based mode (all O)
            slot_tags = ["O"] * len(words)

        results.append(_format_output(words, slot_tags, intent))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rule-based JointBERT fallback predictor. "
            "No torch or transformers required. "
            "Output format identical to predict.py."
        )
    )
    parser.add_argument(
        "--input_file",
        default="sample_pred_in.txt",
        help="Input file for prediction (one sentence per line)",
    )
    parser.add_argument(
        "--output_file",
        default="sample_pred_out.txt",
        help="Output file for prediction (use '-' for stdout)",
    )
    parser.add_argument(
        "--task",
        default="atis",
        choices=["atis", "snips"],
        help="Dataset/task to use intent labels from (default: atis)",
    )
    args = parser.parse_args()

    # Read input
    lines = []
    with open(args.input_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                lines.append(line.split())

    # Predict
    results = predict_rule_based(lines, task=args.task)

    # Write output
    if args.output_file == "-":
        for r in results:
            print(r)
    else:
        with open(args.output_file, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(r + "\n")

    print(f"Rule-based prediction done. {len(results)} sentences processed.", file=sys.stderr)


if __name__ == "__main__":
    main()
