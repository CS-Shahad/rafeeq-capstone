"""Offline tests (no model, no key): the code-enforced defences block what they must and allow normal work.

    uv run pytest -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agentkit as K  # noqa: E402
import evals  # noqa: E402
import madar  # noqa: E402
import rafeeq as R  # noqa: E402
from harness import (MAX_FAILED_VERIFICATIONS, Harness, ScriptedModel, auto_consent, find_personal_data,  # noqa: E402
                     mask_args, outside_links)

SARA = {"permit_id": "REN-10482", "national_id_last4": "4821", "mobile": "0501234821"}
SARA_EMAIL = "sara.q@mail.example"


@pytest.fixture
def h(tmp_path) -> Harness:
    m = madar.Madar(tmp_path / "db")
    return Harness(m, session="test", approve=auto_consent, audit_path=tmp_path / "audit.jsonl")


def verify_sara(h: Harness) -> None:
    result = json.dumps(h.m.get_permit(**SARA))
    h.after_tool("get_permit", SARA, result)


def email(to: str = SARA_EMAIL, body: str = "Your application is under review.", subject: str = "Your application"):
    return {"to": to, "subject": subject, "body": body}


# ---------------------------------------------------------------- data-loss patterns
@pytest.mark.parametrize("text", [
    "ID 1084524821", "national id: 2211475530", "call 0501234821", "mobile +966501234821", "00966501234821",
    "050 123 4821", "050-123-4821", "+966 50 123 4821",
    "الجوال ٠٥٠١٢٣٤٨٢١", "الهوية ١٠٨٤٥٢٤٨٢١",  # Arabic-Indic digits are normalised first
    "IBAN SA03 8000 0000 6080 1016 7519",
])
def test_personal_data_found(text):
    assert find_personal_data(text), text


@pytest.mark.parametrize("text", [
    "Your inspection is on 2026-11-04 09:00.", "Fee: 1,280 SAR (POL-02).", "REN-10482 is under review.",
    "ID ends with 4821", "Refund REF-5001 of 2700 SAR", "رسوم ١٢٨٠ ريالًا، الموعد ٤ نوفمبر ٢٠٢٦ الساعة ٩:٠٠",
    "Escalation ESC-301, reply within 2 working days.", "Call 199-00 during office hours.",
])
def test_normal_text_not_flagged(text):
    assert not find_personal_data(text), text


def test_links():
    assert outside_links("see https://tracker.example/b.png?x=1") == ["tracker.example"]
    assert outside_links("![badge](http://evil.example/p)") == ["evil.example"]
    assert outside_links("www.other.example") == ["other.example"]
    assert outside_links("https://madar.example/fees and https://portal.madar.example/x") == []


# ---------------------------------------------------------------- egress guard on send_email
def test_blocks_email_before_anything_is_verified(h):
    assert h.check("send_email", email())


def test_allows_summary_to_verified_registered_address(h):
    verify_sara(h)
    assert h.check("send_email", email(body="REN-10482 is under review; decision due 2 November 2026. "
                                            "Details: https://madar.example/permits")) == []
    assert h.check("send_email", email(to="Sara.Q@mail.example")) == []  # case-insensitive


@pytest.mark.parametrize("args,reason", [
    (email(to="someone@other.example"), "registered address"),
    (email(to="fahad.harbi@mail.example"), "registered address"),  # a real applicant, but not verified here
    (email(body="Your mobile 0501234821 is on file."), "mobile"),
    (email(body="ID 1084524821"), "national ID"),
    (email(subject="ID ١٠٨٤٥٢٤٨٢١"), "national ID"),
    (email(body="Track it here: https://tracker.example/b.png"), "links may only point"),
])
def test_blocks_bad_emails_even_after_verification(h, args, reason):
    verify_sara(h)
    why = h.check("send_email", args)
    assert any(reason in w for w in why), why


def test_before_tool_returns_error_the_model_can_read(h):
    out = h.before_tool("send_email", email(to="someone@other.example"))
    assert out.startswith('{"error": "Blocked by Rafeeq') and h.blocks == 1


def test_naive_harness_has_no_guard(tmp_path):
    naive = Harness(madar.Madar(tmp_path / "db"), approve=auto_consent, audit_path=tmp_path / "a.jsonl", guards=False)
    assert naive.before_tool("send_email", email(to="someone@other.example")) is None


# ---------------------------------------------------------------- verification lockout (POL-08)
def test_lockout_after_failed_verifications(h):
    bad = {"permit_id": "BLD-20417", "national_id_last4": "0000", "mobile": "0500000000"}
    failure = json.dumps({"error": "Identity verification failed (POL-08). ..."})
    for _ in range(MAX_FAILED_VERIFICATIONS):
        assert h.before_tool("get_permit", bad) is None
        h.after_tool("get_permit", bad, failure)
    assert "locked" in h.before_tool("get_permit", bad)
    assert "locked" in h.before_tool("create_refund", {**bad, "reason": "rejected"})
    assert h.before_tool("search_policies", {"query": "fees"}) is None  # public information still works


def test_one_typo_does_not_lock(h):
    h.after_tool("get_permit", SARA, json.dumps({"error": "Identity verification failed (POL-08)."}))
    verify_sara(h)
    assert h.before_tool("get_permit", SARA) is None and "REN-10482" in h.verified


# ---------------------------------------------------------------- approvals and audit
def test_writes_ask_reads_allow(h):
    assert h.permission("get_permit", SARA) == "allow"
    for name in ("book_inspection", "reschedule_inspection", "create_refund", "escalate", "send_email"):
        assert h.permission(name, {}) == "ask"


def test_human_not_asked_about_a_call_the_guard_will_block(tmp_path):
    asked = []
    h = Harness(madar.Madar(tmp_path / "db"), approve=lambda n, a: asked.append(n) or True,
                audit_path=tmp_path / "a.jsonl")
    h.approve("send_email", email(to="someone@other.example"))
    assert asked == []


def test_audit_log_masks_credentials(h):
    verify_sara(h)
    line = h.audit_path.read_text(encoding="utf-8")
    assert "0501234821" not in line and "4821" not in line
    assert mask_args(SARA)["mobile"].endswith("21")


# ---------------------------------------------------------------- normal multi-step work passes the full harness
def test_booking_and_summary_email_through_real_harness_and_mcp(tmp_path):
    m = madar.Madar(tmp_path / "db")
    creds = {"national_id_last4": "6654", "mobile": "0532236654"}
    plan = [
        [("get_permit", {"permit_id": "FTR-30311", **creds})],
        [("list_slots", {"district": "Al-Rawdah", "date_from": "2026-11-04", "date_to": "2026-11-04"})],
        [("book_inspection", {"permit_id": "FTR-30311", "slot": "2026-11-04 09:00", **creds})],
        [("send_email", {"to": "noura.trucks@mail.example", "subject": "Inspection booked",
                         "body": "Health inspection booked for 4 November 2026, 09:00, Al-Rawdah."})],
        "Booked for 4 November 2026 at 09:00.",
    ]
    with R.session(m, "inproc", session="t", approve=auto_consent, audit_path=tmp_path / "a.jsonl") as (mcp, h):
        run = R.run_rafeeq("book", m, mcp, h, R.CONFIGS["B"], shape=ScriptedModel(plan))
    assert run.stopped == "done" and h.blocks == 0
    assert m.sql("SELECT slot FROM inspections WHERE permit_id='FTR-30311'")[0]["slot"] == "2026-11-04 09:00"
    assert m.sql("SELECT to_addr FROM outbox") == [{"to_addr": "noura.trucks@mail.example"}]


# ---------------------------------------------------------------- context rule
def test_compact_drops_old_tool_results_and_keeps_budget():
    turn = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "Sara Al-Qahtani ******4821"},
            {"role": "assistant", "content": "a"}]
    out = R.compact(turn * 3)
    assert all(msg["role"] in ("user", "assistant") for msg in out) and "Al-Qahtani" not in json.dumps(out)
    long = [{"role": "user", "content": "x" * 3000}, {"role": "assistant", "content": "y" * 3000}] * 5
    out = R.compact(long)
    assert sum(len(msg["content"]) for msg in out) <= R.HISTORY_CHAR_BUDGET and out[0]["role"] == "user"


# ---------------------------------------------------------------- the eval suite itself
def test_tasks_file_is_valid():
    tasks = evals.load_tasks(ROOT / "tasks.jsonl")
    assert len(tasks) >= 15 and len({t["id"] for t in tasks}) == len(tasks)
    assert {t["lang"] for t in tasks} == {"ar", "en"}
    kinds = {g["type"] for t in tasks for g in t["graders"]}
    assert kinds == {"contains_any", "not_contains", "tool_called", "tool_not_called", "db", "judge"}
    m = madar.Madar(K.lab_dir("tests/tasks", reset=True))
    for t in tasks:
        for g in t["graders"]:
            if g["type"] == "db":
                m.sql(g["sql"])  # every query runs on a fresh database
    assert set(R.REDTEAM_ATTACKS) | set(R.REDTEAM_BENIGN) <= {t["id"] for t in tasks}


def test_db_expectations_are_reachable(tmp_path):
    """The correct tool calls reach every action task's expected end state (so a failure is the agent's, not the task's)."""
    tasks = {t["id"]: t for t in evals.load_tasks(ROOT / "tasks.jsonl")}
    oracle = {
        "action-book-inspection": lambda m: m.book_inspection("FTR-30311", "2026-11-04 09:00", "6654", "0532236654"),
        "action-reschedule": lambda m: m.reschedule_inspection(
            m.sql("SELECT inspection_id FROM inspections WHERE permit_id='BLD-20981'")[0]["inspection_id"],
            "2026-11-05 11:00", "1196", "0548841196"),
        "action-refund-rejected-ar": lambda m: m.create_refund("SGN-40220", "rejected", "0093", "0540410093"),
        "action-refund-duplicate": lambda m: m.create_refund("SGN-40388", "duplicate_payment", "2277", "0505502277"),
        "action-escalate-overdue-ar": lambda m: m.escalate("BLD-20417", "BLD submitted 28 Sep, 9 working days overdue.",
                                                           "7730", "0555017730"),
        "action-email-summary": lambda m: m.send_email(SARA_EMAIL, "REN-10482", "Under review, due 2 November 2026."),
    }
    for tid, act in oracle.items():
        m = madar.Madar(tmp_path / tid)
        act(m)
        for g in tasks[tid]["graders"]:
            if g["type"] == "db":
                got = next(iter(m.sql(g["sql"])[0].values()))
                assert str(got) == str(g["expect"]), (tid, got)
