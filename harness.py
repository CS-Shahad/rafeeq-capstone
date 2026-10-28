"""Rafeeq's harness: everything around the model that does not depend on the model obeying.

One `Harness` per conversation (session). It plugs into `agentkit.run_agent` through four hooks:

    permission(name, args)          read → "allow"; write → "ask" (a human approves)
    approve(name, args)             asks the human (CLI prompt, auto-consent in evals, an officer queue in the API)
    before_tool(name, args)         code-enforced guards: egress guard on send_email, verification lockout
    after_tool(name, args, result)  remembers verified permits, caps result size, writes the audit log

Rules live here, in code. The model's judgement lives in the prompt (AGENTS.md).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

import agentkit as K
import madar

WRITE_TOOLS = {"book_inspection", "reschedule_inspection", "create_refund", "escalate", "send_email"}
CREDENTIAL_TOOLS = {"get_permit", *WRITE_TOOLS - {"send_email"}}  # tools that take national_id_last4 + mobile

MAX_RESULT_CHARS = 3000  # context budget per tool result
MAX_FAILED_VERIFICATIONS = 3  # POL-08 brute-force protection, per session
ALLOWED_LINK_DOMAIN = "madar.example"

# ---------------------------------------------------------------- data-loss patterns (egress guard)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ID_RE = re.compile(r"(?<!\d)[12]\d{9}(?!\d)")  # Saudi national ID / iqama: 10 digits starting with 1 or 2
MOBILE_RE = re.compile(r"(?<!\d)(?:\+?966|00966|0)?[\s-]?5\d(?:[\s-]?\d){7}(?!\d)")  # 05x xxx xxxx, +9665…, spaced
IBAN_RE = re.compile(r"\bSA\d{2}(?:\s?\w{4}){5}\b", re.I)
LINK_RE = re.compile(r"(?:https?://|www\.)([^/\s)\"'>]+)", re.I)


def find_personal_data(text: str) -> list[str]:
    """What the data-loss check finds in outbound text (Arabic-Indic digits are normalised first)."""
    t = (text or "").translate(ARABIC_DIGITS)
    found = []
    if ID_RE.search(t):
        found.append("a national ID")
    if MOBILE_RE.search(t):
        found.append("a mobile number")
    if IBAN_RE.search(t):
        found.append("an IBAN")
    return found


def outside_links(text: str) -> list[str]:
    hosts = [h.lower().rstrip(".,;") for h in LINK_RE.findall(text or "")]
    return [h for h in hosts if not (h == ALLOWED_LINK_DOMAIN or h.endswith("." + ALLOWED_LINK_DOMAIN))]


def mask(value: str) -> str:
    """Audit-log masking: keep the shape, hide the secret (data minimisation, PDPL)."""
    s = str(value)
    return "****" if len(s) <= 4 else "*" * (len(s) - 2) + s[-2:]


def mask_args(args: dict) -> dict:
    return {k: (mask(v) if k in ("national_id_last4", "mobile") else v) for k, v in (args or {}).items()}


# ---------------------------------------------------------------- the harness
class Harness:
    """Session state + hooks for run_agent.

    guards=False gives the *naive* harness used as the "before" in attack.py: same tools, same approvals,
    but no egress guard and no verification lockout.
    """

    def __init__(self, m: madar.Madar, session: str = "cli", approve: Callable[[str, dict], bool] | None = None,
                 audit_path: Path | None = None, guards: bool = True):
        self.m, self.session, self.guards = m, session, guards
        self.human = approve or approve_cli
        self.audit_path = audit_path or (K.ARTIFACTS / "rafeeq" / "audit.jsonl")
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.verified: set[str] = set()  # permit IDs this session verified successfully (POL-08)
        self.failed_verifications = 0
        self.blocks = 0

    # -- audit log (JSONL): who, what, decision; secrets masked
    def audit(self, tool: str, args: dict, decision: str, **extra) -> None:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": self.session, "tool": tool,
               "args": mask_args(args), "decision": decision, **extra}
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- hook 1: permission policy
    def permission(self, name: str, args: dict) -> str:
        return "ask" if name in WRITE_TOOLS else "allow"

    # -- hook 2: human approval (the guard runs first, so nobody is asked about a call that will be blocked)
    def approve(self, name: str, args: dict) -> bool:
        if self.guards and self.check(name, args):
            self.audit(name, args, "approval_skipped_guard_will_block")
            return True  # let it through to before_tool, which blocks it with the reason the model can read
        ok = bool(self.human(name, args))
        self.audit(name, args, "approved" if ok else "denied_by_human")
        return ok

    # -- hook 3: code-enforced guards
    def check(self, name: str, args: dict) -> list[str]:
        """Reasons to block this call (empty = allowed). Pure function of the session state."""
        why = []
        if name in CREDENTIAL_TOOLS and self.failed_verifications >= MAX_FAILED_VERIFICATIONS:
            why.append(f"{self.failed_verifications} failed identity checks in this session: lookups are locked; "
                       "hand over to an officer (POL-08)")
        if name == "send_email":
            to = str(args.get("to", "")).strip().lower()
            allowed = self.registered_emails()
            if to not in allowed:
                why.append("emails may only go to the registered address of an application verified in this "
                           "conversation (POL-08, POL-10)" + ("" if allowed else "; nothing is verified yet"))
            text = f"{args.get('subject', '')}\n{args.get('body', '')}"
            if found := find_personal_data(text):
                why.append("the email contains " + " and ".join(found) + " (never send full IDs or mobiles)")
            if hosts := outside_links(text):
                why.append(f"links may only point to {ALLOWED_LINK_DOMAIN} (found {', '.join(sorted(set(hosts)))})")
        return why

    def before_tool(self, name: str, args: dict) -> str | None:
        if not self.guards:
            return None
        why = self.check(name, args)
        if not why:
            return None
        self.blocks += 1
        self.audit(name, args, "blocked_by_guard", reasons=why)
        return json.dumps({"error": "Blocked by Rafeeq's security policy: " + "; ".join(why) +
                                    ". Do not retry; tell the person what you can do instead."}, ensure_ascii=False)

    # -- hook 4: observe results (verification state, size cap, audit)
    def after_tool(self, name: str, args: dict, result: str) -> str | None:
        is_error = result.startswith('{"error"')
        if "Identity verification failed" in result:
            self.failed_verifications += 1
        elif name in CREDENTIAL_TOOLS and not is_error:
            pid = str(args.get("permit_id", "")).strip().upper()
            if not pid and name == "reschedule_inspection":
                pid = self._permit_of_inspection(str(args.get("inspection_id", "")))
            if pid:
                self.verified.add(pid)
        if not result.startswith('{"error": "Blocked by Rafeeq'):
            self.audit(name, args, "error" if is_error else "ok", chars=len(result))
        if len(result) > MAX_RESULT_CHARS:
            return result[:MAX_RESULT_CHARS] + f"\n[… {len(result) - MAX_RESULT_CHARS} characters cut; ask for less]"
        return None

    # -- helpers
    def registered_emails(self) -> set[str]:
        if not self.verified:
            return set()
        marks = ",".join("?" * len(self.verified))
        rows = self.m.sql(f"SELECT a.email FROM permits p JOIN applicants a USING (applicant_id) "
                          f"WHERE p.permit_id IN ({marks})", tuple(self.verified))
        return {r["email"].lower() for r in rows}

    def _permit_of_inspection(self, inspection_id: str) -> str:
        rows = self.m.sql("SELECT permit_id FROM inspections WHERE inspection_id = ?", (inspection_id.strip().upper(),))
        return rows[0]["permit_id"] if rows else ""

    def hooks(self) -> dict:
        return dict(permission=self.permission, approve=self.approve, before_tool=self.before_tool,
                    after_tool=self.after_tool)


def approve_cli(name: str, args: dict) -> bool:
    """The human in the loop, in a terminal. Shows the exact arguments (secrets masked)."""
    shown = json.dumps(mask_args(args), ensure_ascii=False, indent=2)
    answer = input(f"\n🛡️  Rafeeq wants to run a WRITE action / رفيق يطلب تنفيذ إجراء:\n   {name}{shown}\n"
                   "   Approve? / موافق؟ [y/N] ")
    return answer.strip().lower() in ("y", "yes", "نعم", "ن")


def auto_consent(name: str, args: dict) -> bool:
    """Unattended runs (evals): the task prompt carries the person's consent; db graders check what was written."""
    return True


# ---------------------------------------------------------------- token budget per run
class BudgetedShape(K.ChatShape):
    """Chat Completions with a token budget per run: once spent, the next step returns no tool calls and the run
    ends with `exhausted = True` (agentkit's max_cost is 0 on free models, so tokens are the budget that bites)."""

    def __init__(self, max_tokens: int, **kw):
        super().__init__(**kw)
        self.max_tokens, self.exhausted, self.start_tokens = max_tokens, False, None

    def start(self, system, task, history=None):
        self.start_tokens = K.LEDGER.input_tokens + K.LEDGER.output_tokens
        self.exhausted = False
        return super().start(system, task, history)

    def step(self, state, tools, model, use_cache=None, **kw):
        if K.LEDGER.input_tokens + K.LEDGER.output_tokens - self.start_tokens > self.max_tokens:
            self.exhausted = True
            return K.Turn(f"(stopped: token budget of {self.max_tokens} for this request used up)", [])
        return super().step(state, tools, model, use_cache=use_cache, **kw)


# ---------------------------------------------------------------- a scripted "model" (offline tests, attack.py)
class ScriptedModel:
    """Stands in for a model that is fully hijacked (or just for a test): replays tool calls through the REAL harness.

    `plan` is a list of turns; each is a final text, or a function(observations) -> list of (tool, args) or a text.
    `observations` holds every tool result seen so far (strings), so a step can copy data from earlier results,
    exactly like a hijacked model would.
    """

    name = "scripted"

    def __init__(self, plan: list):
        self.plan, self.i, self.obs = plan, 0, []

    def start(self, system: str, task: str, history: list | None = None) -> list:
        self.i, self.obs = 0, []
        return [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": task}]

    def step(self, state, tools, model, use_cache=None, **kw) -> K.Turn:
        turn = self.plan[self.i] if self.i < len(self.plan) else "(end of script)"
        self.i += 1
        turn = turn(self.obs) if callable(turn) else turn
        if isinstance(turn, str):
            state.append({"role": "assistant", "content": turn})
            return K.Turn(turn, [])
        calls = [K.Call(f"call_{self.i}_{j}", n, json.dumps(a, ensure_ascii=False)) for j, (n, a) in enumerate(turn)]
        state.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in calls]})
        return K.Turn("", calls)

    def add_results(self, state, results) -> None:
        for c, out in results:
            self.obs.append(out)
            state.append({"role": "tool", "tool_call_id": c.id, "content": out})
