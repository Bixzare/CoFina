"""
Registration Agent - Complete two-stage user registration in one seamless guided flow.

Stage 1: Authentication (user_id, name, email, password, secret Q&A)
          → saved to `users` table on completion of stage 1.
Stage 2: Profile, debt, preferences, goals
          → saved to profile / preferences / debts / financial_plans tables.

The agent maintains ALL state internally across multiple orchestrator calls.
The orchestrator checks `agent.is_active()` and routes directly here when a
registration session is in progress.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from db.queries import (
    email_exists, register_user, user_exists,
)
from tools.user_profile import (
    add_user_debt, create_financial_plan,
    update_user_preferences, update_user_profile,
)


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

# Each step:  step (internal name), field (key in self.data / self.current_debt),
#             prompt (shown to user), section (logical group)
STEPS: List[Dict[str, str]] = [
    # ── Stage 1: Authentication ──────────────────────────────────────────
    {"step": "user_id",         "field": "user_id",          "prompt": "Choose a User ID (min 3 chars):",                           "section": "auth"},
    {"step": "first_name",      "field": "first_name",        "prompt": "Your first name:",                                          "section": "auth"},
    {"step": "last_name",       "field": "last_name",         "prompt": "Your last name:",                                           "section": "auth"},
    {"step": "email",           "field": "email",             "prompt": "Your email address:",                                       "section": "auth"},
    {"step": "password",        "field": "password",          "prompt": "Create a password (min 6 characters):",                     "section": "auth"},
    {"step": "secret_question", "field": "secret_question",   "prompt": "Choose a secret question (for account recovery):",          "section": "auth"},
    {"step": "secret_answer",   "field": "secret_answer",     "prompt": "Answer to your secret question:",                           "section": "auth"},

    # ── Stage 2: Personal & Employment ───────────────────────────────────
    {"step": "profession",        "field": "profession",            "prompt": "What is your profession/field?",                                   "section": "personal"},
    {"step": "current_role",      "field": "current_role",          "prompt": "Your current job title / role:",                                   "section": "personal"},
    {"step": "employment_start",  "field": "employment_start_date", "prompt": "Employment start date (YYYY-MM-DD):",                              "section": "personal"},
    {"step": "age",               "field": "age",                   "prompt": "Your age:",                                                        "section": "personal"},
    {"step": "gender",            "field": "gender",                "prompt": "Gender (Male / Female / Other):",                                  "section": "personal"},
    {"step": "civil_status",      "field": "civil_status",          "prompt": "Civil status (Single / Married / Divorced / Widowed):",            "section": "personal"},
    {"step": "children",          "field": "number_of_children",    "prompt": "Number of dependants / children (0 if none):",                     "section": "personal"},
    {"step": "monthly_income",    "field": "monthly_income",        "prompt": "Monthly take-home salary ($):",                                    "section": "personal"},
    {"step": "annual_income",     "field": "annual_income",         "prompt": "Annual gross income ($):",                                         "section": "personal"},
    {"step": "retirement_target", "field": "retirement_age_target", "prompt": "Target retirement age (e.g. 60):",                                 "section": "personal"},

    # ── Stage 2: Debt ─────────────────────────────────────────────────────
    {"step": "has_debt",      "field": "has_debt",         "prompt": "Do you currently have any debt? (yes / no):",                              "section": "debt"},
    # Steps below are only executed when user has debt
    {"step": "debt_type",     "field": "debt_type",        "prompt": "Debt type (Student Loan / Credit Card / Mortgage / Car Loan / Other):",    "section": "debt"},
    {"step": "creditor",      "field": "creditor",         "prompt": "Creditor / bank name:",                                                    "section": "debt"},
    {"step": "total_amount",  "field": "total_amount",     "prompt": "Total original amount ($):",                                               "section": "debt"},
    {"step": "remaining",     "field": "remaining_amount", "prompt": "Remaining balance ($):",                                                   "section": "debt"},
    {"step": "interest_rate", "field": "interest_rate",    "prompt": "Annual interest rate (%):",                                                "section": "debt"},
    {"step": "min_payment",   "field": "minimum_payment",  "prompt": "Minimum monthly payment ($):",                                             "section": "debt"},
    {"step": "due_date",      "field": "due_date",         "prompt": "Payment due day of month (1-31):",                                         "section": "debt"},
    {"step": "add_more_debt", "field": "add_more",         "prompt": "Add another debt? (yes / no):",                                            "section": "debt"},

    # ── Stage 2: Preferences ──────────────────────────────────────────────
    {"step": "risk_profile",     "field": "risk_profile",     "prompt": "Investment risk tolerance (Low / Moderate / High):",         "section": "preferences"},
    {"step": "debt_strategy",    "field": "debt_strategy",    "prompt": "Debt pay-down strategy (Snowball / Avalanche):",             "section": "preferences"},
    {"step": "savings_priority", "field": "savings_priority", "prompt": "Primary savings goal (e.g. Emergency Fund / Retirement):", "section": "preferences"},

    # ── Stage 2: Goals ────────────────────────────────────────────────────
    {"step": "short_term_goal", "field": "short_term", "prompt": "Short-term financial goal (1–2 years, e.g. 'Save $5 000'):",   "section": "goals"},
    {"step": "long_term_goal",  "field": "long_term",  "prompt": "Long-term financial goal (5+ years, e.g. 'Buy a house'):",    "section": "goals"},

    # ── Confirmation ──────────────────────────────────────────────────────
    {"step": "confirm_all", "field": "confirm", "prompt": "All done! Ready to create your account? (yes / no):", "section": "confirm"},
]

# Index for quick look-up
_STEP_INDEX: Dict[str, int] = {s["step"]: i for i, s in enumerate(STEPS)}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RegistrationAgent:
    """
    Stateful, step-driven registration agent.

    Public API used by the orchestrator
    ─────────────────────────────────────
    • is_active() → bool          True while a registration session is open.
    • process(query, context)     Feed one user utterance; get one response dict.
    • reset()                     Abort and clear all state.

    Response dict shape
    ───────────────────
    {
        "action":  "start" | "ask" | "retry" | "complete" | "error" | "clarify",
        "message": str,          # Text to show the user
        "data":    dict          # Extra metadata (field name, user_id, …)
    }
    """

    def __init__(self) -> None:
        self.reset()

    # ── Public ──────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        """True when a registration session is currently in progress."""
        return self.current_flow is not None

    def reset(self) -> None:
        """Clear all registration state."""
        self.data: Dict[str, Any] = {}
        self.debts: List[Dict[str, Any]] = []
        self.current_debt: Dict[str, Any] = {}
        self.current_flow: Optional[str] = None
        self.step_index: int = 0
        self.has_debt: bool = False
        # Stage-1 saved flag prevents double-insert if something goes wrong
        self._stage1_saved: bool = False

    def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point.  Feed one user message; returns next prompt or result.
        """
        clean = query.strip()

        # ── Not yet started ──────────────────────────────────────────────
        if not self.current_flow:
            triggers = ["register", "sign up", "signup", "new account", "create account"]
            if any(kw in clean.lower() for kw in triggers):
                self.current_flow = "registration"
                self.step_index = 0
                return self._reply(
                    "start",
                    "Welcome! Let's set up your CoFina account. "
                    "I'll guide you step by step.\n\n" + STEPS[0]["prompt"],
                    field=STEPS[0]["field"],
                )
            return self._reply(
                "clarify",
                "I can register you and set up your complete financial profile. "
                "Say 'register' or 'sign up' to begin.",
            )

        # ── Active flow ──────────────────────────────────────────────────
        return self._handle_step(clean)

    # ── Internal step router ─────────────────────────────────────────────

    def _handle_step(self, value: str) -> Dict[str, Any]:
        step_def = STEPS[self.step_index]
        step_name = step_def["step"]
        section = step_def["section"]

        # ── Special: has_debt gate ───────────────────────────────────────
        if step_name == "has_debt":
            return self._handle_has_debt(value)

        # ── Special: add_more_debt gate ──────────────────────────────────
        if step_name == "add_more_debt":
            return self._handle_add_more_debt(value)

        # ── Special: final confirmation ──────────────────────────────────
        if step_name == "confirm_all":
            return self._handle_confirm(value)

        # ── Validate & store ─────────────────────────────────────────────
        error = self._validate_and_store(step_def, value)
        if error:
            return self._reply("retry", error, field=step_def["field"])

        # ── Advance ──────────────────────────────────────────────────────
        self.step_index += 1

        # After finishing auth (step_index now points to first personal step)
        # save stage-1 immediately so the user exists in the DB.
        if section == "auth" and STEPS[self.step_index]["section"] != "auth":
            result = self._save_stage1()
            if result is not None:          # error occurred
                return result

        return self._prompt_current()

    # ── Special step handlers ────────────────────────────────────────────

    def _handle_has_debt(self, value: str) -> Dict[str, Any]:
        if value.lower() in ("yes", "y"):
            self.has_debt = True
            self.step_index += 1           # → debt_type
        else:
            self.has_debt = False
            # Skip all debt detail steps; stop when section changes
            while (self.step_index < len(STEPS) and
                   STEPS[self.step_index]["section"] == "debt"):
                self.step_index += 1
        return self._prompt_current()

    def _handle_add_more_debt(self, value: str) -> Dict[str, Any]:
        # Save the debt we just collected
        if self.current_debt:
            self.debts.append(dict(self.current_debt))
            self.current_debt = {}

        if value.lower() in ("yes", "y"):
            # Loop back to debt_type
            self.step_index = _STEP_INDEX["debt_type"]
        else:
            # Skip remaining debt steps
            while (self.step_index < len(STEPS) and
                   STEPS[self.step_index]["section"] == "debt"):
                self.step_index += 1
        return self._prompt_current()

    def _handle_confirm(self, value: str) -> Dict[str, Any]:
        if value.lower() in ("yes", "y"):
            return self._save_stage2()
        # User declined — restart
        self.reset()
        return self._reply(
            "restart",
            "No problem! Let's start over.\n\n" + STEPS[0]["prompt"],
            field=STEPS[0]["field"],
        )

    # ── Validate & store per section ─────────────────────────────────────

    def _validate_and_store(self, step_def: Dict, value: str) -> Optional[str]:
        """
        Validate *value* for *step_def*.  Store the (possibly coerced) value
        in self.data or self.current_debt.  Return an error string on failure,
        or None on success.
        """
        section = step_def["section"]
        field = step_def["field"]

        if section == "auth":
            return self._validate_auth(step_def, value)

        if section == "personal":
            return self._validate_personal(step_def, value)

        if section == "debt":
            return self._validate_debt(step_def, value)

        if section == "preferences":
            return self._validate_preferences(step_def, value)

        if section == "goals":
            if not value:
                return f"{step_def['prompt']} cannot be empty."
            self.data[field] = value
            return None

        return None   # confirm handled separately

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

        # All other auth fields (first_name, last_name, secret_question, secret_answer)
        if not value:
            return f"{step_def['prompt']} cannot be empty."
        self.data[field] = value
        return None

    def _validate_personal(self, step_def: Dict, value: str) -> Optional[str]:
        field = step_def["field"]

        if field == "age":
            try:
                age = int(value)
                if not (18 <= age <= 100):
                    return "Please enter a valid age between 18 and 100:"
                self.data[field] = age
                return None
            except ValueError:
                return "Age must be a whole number. Try again:"

        if field == "number_of_children":
            try:
                n = int(value)
                if n < 0:
                    return "Number of children cannot be negative. Try again:"
                self.data[field] = n
                return None
            except ValueError:
                return "Please enter a whole number (e.g. 0, 1, 2):"

        if field in ("monthly_income", "annual_income"):
            try:
                amount = float(value.replace("$", "").replace(",", ""))
                if amount < 0:
                    return "Income cannot be negative. Try again:"
                self.data[field] = round(amount, 2)
                return None
            except ValueError:
                return "Please enter a valid dollar amount (e.g. 3500 or 3,500):"

        if field == "employment_start_date":
            try:
                datetime.strptime(value, "%Y-%m-%d")
                self.data[field] = value
                return None
            except ValueError:
                return "Please use YYYY-MM-DD format (e.g. 2020-01-15):"

        if field == "retirement_age_target":
            try:
                age = int(value)
                current_age = self.data.get("age", 0)
                if age <= current_age:
                    return f"Retirement age must be greater than your current age ({current_age}):"
                if age > 85:
                    return "Please enter a retirement age of 85 or below:"
                self.data[field] = age
                return None
            except ValueError:
                return "Please enter a whole number (e.g. 60):"

        if field == "gender":
            valid = {"male", "female", "other", "m", "f"}
            if value.lower() not in valid:
                return "Please enter Male, Female, or Other:"
            self.data[field] = value.title()
            return None

        if field == "civil_status":
            valid = {"single", "married", "divorced", "widowed"}
            if value.lower() not in valid:
                return "Please enter Single, Married, Divorced, or Widowed:"
            self.data[field] = value.title()
            return None

        # Catch-all text fields
        if not value:
            return f"{step_def['prompt']} cannot be empty."
        self.data[field] = value
        return None

    def _validate_debt(self, step_def: Dict, value: str) -> Optional[str]:
        field = step_def["field"]

        if field in ("total_amount", "remaining_amount", "interest_rate", "minimum_payment"):
            try:
                num = float(value.replace("$", "").replace(",", "").replace("%", ""))
                if num < 0:
                    return "Value cannot be negative. Try again:"
                self.current_debt[field] = round(num, 2)
                return None
            except ValueError:
                return "Please enter a valid number:"

        if field == "due_date":
            try:
                day = int(value)
                if not (1 <= day <= 31):
                    return "Please enter a day between 1 and 31:"
                self.current_debt[field] = day
                return None
            except ValueError:
                return "Please enter a day number (e.g. 15):"

        if field == "debt_type":
            # Normalise to the exact values the DB CHECK constraint expects
            _map = {
                "student loan": "Student Loan",
                "credit card":  "Credit Card",
                "personal loan": "Personal Loan",
                "mortgage":     "Mortgage",
                "car loan":     "Car Loan",
                "other":        "Other",
            }
            normalised = _map.get(value.lower().strip())
            if not normalised:
                return ("Please enter one of: Student Loan, Credit Card, "
                        "Personal Loan, Mortgage, Car Loan, Other:")
            self.current_debt[field] = normalised
            return None

        # Text fields: creditor
        if not value:
            return f"{step_def['prompt']} cannot be empty."
        self.current_debt[field] = value
        return None

    def _validate_preferences(self, step_def: Dict, value: str) -> Optional[str]:
        field = step_def["field"]

        if field == "risk_profile":
            if value.lower() not in ("low", "moderate", "high"):
                return "Please enter Low, Moderate, or High:"
            self.data[field] = value.title()
            return None

        if field == "debt_strategy":
            if value.lower() not in ("snowball", "avalanche"):
                return "Please enter Snowball or Avalanche:"
            self.data[field] = value.title()
            return None

        # savings_priority — free text
        if not value:
            return f"{step_def['prompt']} cannot be empty."
        self.data[field] = value
        return None

    # ── Save helpers ─────────────────────────────────────────────────────

    def _save_stage1(self) -> Optional[Dict[str, Any]]:
        """
        Insert the user into the `users` table immediately after the auth section.
        Returns an error response dict on failure, None on success.
        """
        if self._stage1_saved:
            return None

        try:
            ok = register_user(
                self.data["user_id"],
                self.data["first_name"],
                self.data["last_name"],
                self.data["email"],
                self.data["password"],
                self.data["secret_question"],
                self.data["secret_answer"],
            )
            if not ok:
                self.reset()
                return self._reply("error", "❌ Could not create your account. "
                                   "The User ID or email may already be in use. "
                                   "Please restart and try different credentials.")
            self._stage1_saved = True
            return None
        except Exception as exc:
            print(f"[RegistrationAgent] Stage-1 save error: {exc}")
            self.reset()
            return self._reply("error", "❌ A database error occurred during account creation. "
                               "Please try again later.")

    def _save_stage2(self) -> Dict[str, Any]:
        """
        Persist profile, preferences, debts and financial plan.
        Called when user confirms at the end of stage 2.
        """
        try:
            user_id = self.data["user_id"]

            # 1. Profile
            profile_fields = [
                "profession", "current_role", "employment_start_date", "age",
                "gender", "civil_status", "number_of_children",
                "monthly_income", "annual_income", "retirement_age_target",
            ]
            profile_data = {k: self.data[k] for k in profile_fields if k in self.data}
            if profile_data:
                update_user_profile(user_id, **profile_data)

            # 2. Preferences
            pref_fields = ["risk_profile", "debt_strategy", "savings_priority"]
            pref_data = {k: self.data[k] for k in pref_fields if k in self.data}
            if pref_data:
                update_user_preferences(user_id, **pref_data)

            # 3. Debts
            for debt in self.debts:
                add_user_debt(user_id, debt)

            # 4. Financial plan
            short_term = self.data.get("short_term", "")
            long_term = self.data.get("long_term", "")
            plan_name = f"{self.data['first_name']}'s Financial Plan"
            create_financial_plan(
                user_id,
                plan_name,
                {"description": short_term},
                {"description": long_term},
            )

            # 5. Retirement estimate
            retirement_str = self._estimate_retirement()

            first_name = self.data["first_name"]
            debt_count = len(self.debts)
            self.reset()

            msg = (
                f"✅ Welcome aboard, {first_name}! "
                f"Your CoFina account and financial profile are ready.\n"
                f"  • Debts recorded : {debt_count}\n"
                f"  • Financial plan  : {plan_name}\n"
                f"  • Est. retirement : {retirement_str}"
            )
            return self._reply("complete", msg,
                               user_id=user_id,
                               retirement_date=retirement_str,
                               profile_complete=True)

        except Exception as exc:
            print(f"[RegistrationAgent] Stage-2 save error: {exc}")
            self.reset()
            return self._reply("error",
                               "❌ An error occurred while saving your profile. "
                               "Your account was created — please contact support "
                               "if your profile is incomplete.")

    # ── Retirement helper ────────────────────────────────────────────────

    def _estimate_retirement(self) -> str:
        try:
            current_age = int(self.data.get("age", 30))
            target_age = int(self.data.get("retirement_age_target", 65))
            years_left = max(target_age - current_age, 0)
            retirement_year = datetime.now().year + years_left
            return f"{retirement_year}-01-01"
        except Exception:
            return f"{datetime.now().year + 30}-01-01"

    # ── Prompt helper ────────────────────────────────────────────────────

    def _prompt_current(self) -> Dict[str, Any]:
        if self.step_index >= len(STEPS):
            # Shouldn't normally happen; trigger save directly
            return self._save_stage2()
        step_def = STEPS[self.step_index]
        return self._reply("ask", step_def["prompt"], field=step_def["field"])

    # ── Response factory ─────────────────────────────────────────────────

    @staticmethod
    def _reply(action: str, message: str, **data_kwargs) -> Dict[str, Any]:
        return {"action": action, "message": message, "data": data_kwargs}