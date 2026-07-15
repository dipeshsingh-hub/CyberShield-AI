"""
event_store.py
----------------
Lightweight persistence for every event the orchestrator processes —
whether it triggers a full playbook or just gets logged ("store event
only"). Append-only JSONL — simple, human-inspectable, diffable, and
sufficient for this platform's scale; swap for a real database in
production without changing the calling code's shape.
"""

import os

from utils import get_logger, safe_json_dumps, utc_now_iso

logger = get_logger("event_store")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVENT_LOG_PATH = os.path.join(BASE_DIR, "data", "event_log.jsonl")


def store_event(event_record: dict) -> None:
    """Appends one event record (with or without a playbook) to the event log."""
    os.makedirs(os.path.dirname(EVENT_LOG_PATH), exist_ok=True)
    record = dict(event_record)
    record.setdefault("stored_at", utc_now_iso())
    with open(EVENT_LOG_PATH, "a") as f:
        f.write(safe_json_dumps(record) + "\n")
    has_playbook = isinstance(record.get("playbook"), dict)
    logger.info(f"Stored event (risk_level={record.get('risk_level', 'unknown')}, "
                f"playbook={'yes' if has_playbook else 'no'}) -> {EVENT_LOG_PATH}")


def load_all_events() -> list:
    if not os.path.exists(EVENT_LOG_PATH):
        return []
    import json
    events = []
    with open(EVENT_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


if __name__ == "__main__":
    store_event({"anomaly_score": 12.3, "phishing_probability": 0.02,
                 "final_risk_probability": 0.05, "risk_level": "Low", "playbook": None})
    events = load_all_events()
    logger.info(f"event_store.py self-test: {len(events)} event(s) in log, last entry: {events[-1]}")
