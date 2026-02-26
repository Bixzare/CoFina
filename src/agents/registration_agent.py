"""
Registration Agent — Streamlined two-stage user onboarding.

Stage 1  Authentication  (6 fields saved immediately to `users` table)
         user_id · first_name · email · password · secret_question · secret_answer

Stage 2  Finance         (4 fields saved on confirmation)
         monthly_income · risk_profile · short_term_goal · long_term_goal

Derived automatically:
  • annual_income = monthly_income × 12

Removed from original flow (can be collected later via profile update):
  last_name, profession, current_role, employment_start_date, age, gender,
  civil_status, number_of_children, retirement_age_target,
  debt collection, debt_strategy, savings_priority.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from db.queries import email_exists, register_user, user_exists
from tools.user_profile import create_financial_plan, update_user_preferences, update_user_profile


# ---------------------------------------------------------------------------
# Step definitions  (10 user-facing fields + 1 confirm)
# ---------------------------------------------------------------------------

STEPS: List[Dict[str, str]] = [
    # ── Stage 1: Authentication ──────────────────────────────────────────
    {"step": "user_id",         "field": "user_id",         "prompt": "Choose a User ID (min 3 characters):",               "section": "auth"},
    {"step": "first_name",      "field": "first_name",      "prompt": "Your first name:",                                   "section": "auth"},
    {"step": "email",           "field": "email",           "prompt": "Your email address:",                                "section": "auth"},
    {"step": "password",        "field": "password",        "prompt": "Create a password (min 6 characters):",              "section": "auth"},
    {"step": "secret_question", "field": "secret_question", "prompt": "Choose a secret question (for account recovery):",   "section": "auth"},
    {"step": "secret_answer",   "field": "secret_answer",   "prompt": "Your answer to that question:",                      "section": "auth"},

    # ── Stage 2: Financial profile ───────────────────────────────────────
    {"step": "monthly_income",  "field": "monthly_income",  "prompt": "Your monthly take-home income ($):",                 "section": "finance"},
    {"step": "risk_profile",    "field": "risk_profile",    "prompt": "Investment risk tolerance (Low / Moderate / High):", "section": "finance"},
    {"step": "short_term_goal", "field": "short_term",      "prompt": "Short-term goal (1-2 years, e.g. 'Save $5 000'):",  "section": "finance"},
    {"step": "long_term_goal",  "field": "long_term",       "prompt": "Long-term goal (5+ years, e.g. 'Buy a house'):",    "section": "finance"},

    # ── Confirmation ──────────────────────────────────────────────────────
    {"step": "confirm_all", "field": "confirm", "prompt": "All set! Create your account now? (yes / no):", "section": "confirm"},
]

_STEP_INDEX: Dict[str, int] = {s["step"]: i for i, s in enumerate(STEPS)}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RegistrationAgent:
    """
    Stateful, step-driven registration agent.

    Public API (used by orchestrator)
    ──────────────────────────────────
    • is_active()              → bool   True while a registration session is open.
    • process(query, context)  → dict   Feed one user utterance; get one response dict.
    • reset()                           Abort and clear all state.

    Response dict shape
    ───────────────────
    {
        "action":  "start" | "ask" | "retry" | "complete" | "error" | "clarify",
        "message": str,
        "data":    dict
    }
    """

    def __init__(self) -> None:
        self.reset()

    # ── Public ──────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        return self.current_flow is not None

    def reset(self) -> None:
        self.data: Dict[str, Any] = {}
        self.current_flow: Optional[str] = None
        self.step_index: int = 0
        self._stage1_saved: bool = False

    def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        clean = query.strip()

        if not self.current_flow:
            triggers = ["register", "sign up", "signup", "new account", "create account"]
            if any(kw in clean.lower() for kw in triggers):
                self.current_flow = "registration"
                self.step_index = 0
                return self._reply(
                    "start",
                    "Welcome! Let's set up your CoFina account — it only takes a minute.\n\n"
                    + STEPS[0]["prompt"],
                    field=STEPS[0]["field"],
                )
            return self._reply(
                "clarify",
                "I can register you and set up your financial profile. "
                "Say 'register' or 'sign up' to begin.",
            )

        return self._handle_step(clean)

    # ── Step router ──────────────────────────────────────────────────────

    def _handle_step(self, value: str) -> Dict[str, Any]:
        step_def = STEPS[self.step_index]
        step_name = step_def["step"]

        if step_name == "confirm_all":
            return self._handle_confirm(value)

        error = self._validate_and_store(step_def, value)
        if error:
            return self._reply("retry", error, field=step_def["field"])

        self.step_index += 1

        # Save Stage 1 immediately when auth section finishes
        if step_def["section"] == "auth" and STEPS[self.step_index]["section"] != "auth":
            result = self._save_stage1()
            if result is not None:
                return result

        return self._prompt_current()

    # ── Confirm handler ──────────────────────────────────────────────────

    def _handle_confirm(self, value: str) -> Dict[str, Any]:
        if value.lower() in ("yes", "y"):
            return self._save_stage2()
        self.reset()
        return self._reply(
            "restart",
            "No problem! Let's start over.\n\n" + STEPS[0]["prompt"],
            field=STEPS[0]["field"],
        )

    # ── Validation ───────────────────────────────────────────────────────

    def _validate_and_store(self, step_def: Dict, value: str) -> Optional[str]:
        section = step_def["section"]
        if section == "auth":
            return self._validate_auth(step_def, value)
        if section == "finance":
            return self._validate_finance(step_def, value)
        return None

    def _validate_auth(self, step_def: Dict, value: str) -> Optional[str]:
        field = step_def["field"]

        if field == "user_id":
            if len(value) < 3:
                return "User ID must be at least 3 characters. Try again:"
            if user_exists(value):
                return f"'{value}' is already taken. Please choose another User ID:"
            self.data[field] = value
            return None

        if field == "email":
            if not re.match(r"^[\w.\-+]+@[\w.\-]+\.\w{2,}$", value):
                return "That doesn't look like a valid email address. Try again:"
            if email_exists(value):
                return "That email is already registered. Please use another:"
            self.data[field] = value
            return None

        if field == "password":
            if len(value) < 6:
                return "Password must be at least 6 characters. Try again:"
            self.data[field] = value
            return None

        # first_name, secret_question, secret_answer
        if not value:
            return f"{step_def['prompt']} cannot be empty."
        self.data[field] = value
        return None

    def _validate_finance(self, step_def: Dict, value: str) -> Optional[str]:
        field = step_def["field"]

        if field == "monthly_income":
            try:
                amount = float(value.replace("$", "").replace(",", ""))
                if amount < 0:
                    return "Income cannot be negative. Try again:"
                self.data["monthly_income"] = round(amount, 2)
                self.data["annual_income"] = round(amount * 12, 2)   # auto-derived
                return None
            except ValueError:
                return "Please enter a valid dollar amount (e.g. 3500 or 3,500):"

        if field == "risk_profile":
            if value.lower() not in ("low", "moderate", "high"):
                return "Please enter Low, Moderate, or High:"
            self.data[field] = value.title()
            return None

        # short_term / long_term — free text
        if not value:
            return f"{step_def['prompt']} cannot be empty."
        self.data[field] = value
        return None

    # ── Save helpers ─────────────────────────────────────────────────────

    def _save_stage1(self) -> Optional[Dict[str, Any]]:
        """Insert user row immediately after auth steps complete."""
        if self._stage1_saved:
            return None
        try:
            ok = register_user(
                self.data["user_id"],
                self.data["first_name"],
                "",                           # last_name — empty, updatable later
                self.data["email"],
                self.data["password"],
                self.data["secret_question"],
                self.data["secret_answer"],
            )
            if not ok:
                self.reset()
                return self._reply(
                    "error",
                    "Could not create your account. The User ID or email may already "
                    "be in use. Please restart and try different credentials.",
                )
            self._stage1_saved = True
            return None
        except Exception as exc:
            print(f"[RegistrationAgent] Stage-1 save error: {exc}")
            self.reset()
            return self._reply("error", "A database error occurred. Please try again later.")

    def _save_stage2(self) -> Dict[str, Any]:
        """Persist financial profile and initial plan on confirmation."""
        try:
            user_id = self.data["user_id"]

            # Profile — income fields only
            profile_data = {
                k: self.data[k]
                for k in ("monthly_income", "annual_income")
                if k in self.data
            }
            if profile_data:
                update_user_profile(user_id, **profile_data)

            # Preferences — risk profile only
            if "risk_profile" in self.data:
                update_user_preferences(user_id, risk_profile=self.data["risk_profile"])

            # Financial plan
            plan_name = f"{self.data['first_name']}'s Financial Plan"
            create_financial_plan(
                user_id,
                plan_name,
                {"description": self.data.get("short_term", "")},
                {"description": self.data.get("long_term", "")},
            )

            first_name = self.data["first_name"]
            risk = self.data.get("risk_profile", "—")
            monthly = self.data.get("monthly_income", 0)
            self.reset()

            return self._reply(
                "complete",
                f"✅ Welcome aboard, {first_name}! Your CoFina account is ready.\n"
                f"  • Financial plan  : {plan_name}\n"
                f"  • Monthly income  : ${monthly:,.2f}\n"
                f"  • Risk profile    : {risk}\n\n"
                "You can update your full profile anytime — just ask me to "
                "'update my profile'.",
                user_id=user_id,
                profile_complete=True,
            )

        except Exception as exc:
            print(f"[RegistrationAgent] Stage-2 save error: {exc}")
            self.reset()
            return self._reply(
                "error",
                "An error occurred while saving your profile. Your account was created "
                "— please contact support if anything is missing.",
            )

    # ── Prompt helper ────────────────────────────────────────────────────

    def _prompt_current(self) -> Dict[str, Any]:
        if self.step_index >= len(STEPS):
            return self._save_stage2()
        step_def = STEPS[self.step_index]
        return self._reply("ask", step_def["prompt"], field=step_def["field"])

    # ── Response factory ─────────────────────────────────────────────────

    @staticmethod
    def _reply(action: str, message: str, **data_kwargs) -> Dict[str, Any]:
        return {"action": action, "message": message, "data": data_kwargs}