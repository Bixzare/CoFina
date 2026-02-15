"""
Registration Agent - Complete user registration and profile setup in one flow
"""

from typing import Dict, Any, Optional
import re
from datetime import datetime

from db.queries import (
    register_user, user_exists, verify_login, 
    get_secret_question, verify_secret_answer, email_exists
)
from tools.user_profile import (
    update_user_profile, update_user_preferences,
    add_user_debt, create_financial_plan
)

class RegistrationAgent:
    """
    Handles complete user registration + profile setup in one seamless flow
    """
    
    # Complete registration steps (all in one flow)
    STEPS = [
        # Authentication (Step 1-7)
        {"step": "user_id", "field": "user_id", "prompt": "Choose a User ID:", "section": "auth"},
        {"step": "first_name", "field": "first_name", "prompt": "Your first name:", "section": "auth"},
        {"step": "last_name", "field": "last_name", "prompt": "Your last name:", "section": "auth"},
        {"step": "email", "field": "email", "prompt": "Your email address:", "section": "auth"},
        {"step": "password", "field": "password", "prompt": "Create a password (min 6 characters):", "section": "auth"},
        {"step": "secret_question", "field": "secret_question", "prompt": "Secret question (for password recovery):", "section": "auth"},
        {"step": "secret_answer", "field": "secret_answer", "prompt": "Answer to secret question:", "section": "auth"},
        
        # Personal & Employment (Step 8-17)
        {"step": "profession", "field": "profession", "prompt": "What is your profession?", "section": "personal"},
        {"step": "current_role", "field": "current_role", "prompt": "Your current job title/role:", "section": "personal"},
        {"step": "employment_start", "field": "employment_start_date", "prompt": "Employment start date (YYYY-MM-DD):", "section": "personal"},
        {"step": "age", "field": "age", "prompt": "Your age:", "section": "personal"},
        {"step": "gender", "field": "gender", "prompt": "Gender (M/F/Other):", "section": "personal"},
        {"step": "civil_status", "field": "civil_status", "prompt": "Civil status (Single/Married/Divorced):", "section": "personal"},
        {"step": "children", "field": "number_of_children", "prompt": "Number of children (0 if none):", "section": "personal"},
        {"step": "monthly_income", "field": "monthly_income", "prompt": "Monthly take-home salary ($):", "section": "personal"},
        {"step": "annual_income", "field": "annual_income", "prompt": "Annual income ($):", "section": "personal"},
        {"step": "retirement_target", "field": "retirement_age_target", "prompt": "Target retirement age:", "section": "personal"},
        
        # Debt collection (Step 18-27)
        {"step": "has_debt", "field": "has_debt", "prompt": "Do you have any debt? (yes/no):", "section": "debt"},
        {"step": "debt_type", "field": "debt_type", "prompt": "Debt type (Student Loan/Credit Card/Mortgage/Car Loan/Other):", "section": "debt"},
        {"step": "creditor", "field": "creditor", "prompt": "Creditor/bank name:", "section": "debt"},
        {"step": "total_amount", "field": "total_amount", "prompt": "Total original amount ($):", "section": "debt"},
        {"step": "remaining", "field": "remaining_amount", "prompt": "Remaining balance ($):", "section": "debt"},
        {"step": "interest_rate", "field": "interest_rate", "prompt": "Interest rate (%):", "section": "debt"},
        {"step": "min_payment", "field": "minimum_payment", "prompt": "Minimum monthly payment ($):", "section": "debt"},
        {"step": "due_date", "field": "due_date", "prompt": "Due date (day of month, e.g., 15):", "section": "debt"},
        {"step": "add_more_debt", "field": "add_more", "prompt": "Add another debt? (yes/no):", "section": "debt"},
        
        # Preferences (Step 28-30)
        {"step": "risk_profile", "field": "risk_profile", "prompt": "Risk tolerance (Low/Moderate/High):", "section": "preferences"},
        {"step": "debt_strategy", "field": "debt_strategy", "prompt": "Debt payment strategy (Snowball/Avalanche):", "section": "preferences"},
        {"step": "savings_priority", "field": "savings_priority", "prompt": "Primary savings goal:", "section": "preferences"},
        
        # Goals (Step 31-32)
        {"step": "short_term_goal", "field": "short_term", "prompt": "Short-term goal (1-2 years, e.g., 'Save $5000'):", "section": "goals"},
        {"step": "long_term_goal", "field": "long_term", "prompt": "Long-term goal (5+ years, e.g., 'Buy a house'):", "section": "goals"},
        
        # Final confirmation
        {"step": "confirm_all", "field": "confirm", "prompt": "All information collected! Ready to create your account? (yes/no):", "section": "confirm"}
    ]
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all state"""
        self.data = {}
        self.debts = []
        self.current_debt = {}
        self.current_flow = None
        self.current_step = None
        self.step_index = 0
        self.collecting_debts = False
    
    def process(self, query: str, context: Dict) -> Dict[str, Any]:
        """Process complete registration flow"""
        
        # Start new flow
        if not self.current_flow:
            if any(kw in query.lower() for kw in ["register", "sign up", "new account"]):
                self.current_flow = "registration"
                self.step_index = 0
                self.current_step = self.STEPS[0]["step"]
                return {
                    "action": "start",
                    "message": "I'll help you create your complete financial profile. Let's start!",
                    "data": self._get_current_prompt()
                }
            return {
                "action": "clarify",
                "message": "I can help you register and set up your complete financial profile. Would you like to start?",
                "data": {}
            }
        
        # Continue flow
        return self._handle_step(query)
    
    def _handle_step(self, query: str) -> Dict[str, Any]:
        """Handle current step"""
        
        step = self.STEPS[self.step_index]
        field = step["field"]
        clean_query = query.strip()
        
        # Handle special debt collection flow
        if step["section"] == "debt" and step["step"] == "has_debt":
            if clean_query.lower() in ['yes', 'y']:
                self.collecting_debts = True
                self.step_index += 1  # Move to debt_type
                return self._next_step()
            else:
                # Skip all debt steps
                while self.step_index < len(self.STEPS) and self.STEPS[self.step_index]["section"] == "debt":
                    self.step_index += 1
                return self._next_step()
        
        elif step["step"] == "add_more_debt":
            if clean_query.lower() in ['yes', 'y']:
                # Save current debt and start new one
                if self.current_debt:
                    self.debts.append(self.current_debt)
                self.current_debt = {}
                # Go back to debt_type
                self.step_index = self._find_step_index("debt_type")
                return self._next_step()
            else:
                # Done with debts
                if self.current_debt:
                    self.debts.append(self.current_debt)
                # Move to next section after debts
                while self.step_index < len(self.STEPS) and self.STEPS[self.step_index]["section"] == "debt":
                    self.step_index += 1
                return self._next_step()
        
        # Handle confirmation
        elif step["step"] == "confirm_all":
            if clean_query.lower() in ['yes', 'y']:
                return self._save_all()
            else:
                self.reset()
                return {
                    "action": "restart",
                    "message": "Let's start over. Choose a User ID:",
                    "data": {"field": "user_id"}
                }
        
        # Validate and store data
        if step["section"] == "auth":
            validation = self._validate_auth_field(step, clean_query)
            if validation != "ok":
                return validation
        
        elif step["section"] == "personal":
            validation = self._validate_personal_field(step, clean_query)
            if validation != "ok":
                return validation
        
        elif step["section"] == "debt" and step["step"] not in ["has_debt", "add_more_debt"]:
            # Store in current_debt
            validation = self._validate_debt_field(step, clean_query)
            if validation != "ok":
                return validation
            self.current_debt[field] = clean_query
        
        elif step["section"] == "preferences":
            validation = self._validate_preference_field(step, clean_query)
            if validation != "ok":
                return validation
            self.data[field] = clean_query
        
        elif step["section"] == "goals":
            self.data[field] = clean_query
        
        # Move to next step
        self.step_index += 1
        return self._next_step()
    
    def _next_step(self) -> Dict[str, Any]:
        """Move to next step and return prompt"""
        if self.step_index < len(self.STEPS):
            self.current_step = self.STEPS[self.step_index]["step"]
            return {
                "action": "ask",
                "message": self.STEPS[self.step_index]["prompt"],
                "data": {"field": self.STEPS[self.step_index]["field"]}
            }
        
        # Should not reach here - confirmation step should handle
        return self._save_all()
    
    def _get_current_prompt(self) -> Dict[str, Any]:
        """Get current step prompt"""
        step = self.STEPS[self.step_index]
        return {
            "field": step["field"],
            "prompt": step["prompt"]
        }
    
    def _find_step_index(self, step_name: str) -> int:
        """Find index of a step by name"""
        for i, step in enumerate(self.STEPS):
            if step["step"] == step_name:
                return i
        return 0
    
    def _validate_auth_field(self, step: Dict, value: str) -> str:
        """Validate authentication fields"""
        field = step["field"]
        
        if field == "user_id":
            if len(value) < 3:
                return {"action": "retry", "message": "User ID must be at least 3 characters:", "data": {}}
            if user_exists(value):
                return {"action": "retry", "message": f"User ID '{value}' already taken. Choose another:", "data": {}}
            self.data["user_id"] = value
            return "ok"
        
        elif field == "email":
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value):
                return {"action": "retry", "message": "Invalid email format. Try again:", "data": {}}
            if email_exists(value):
                return {"action": "retry", "message": "Email already registered. Use another:", "data": {}}
            self.data["email"] = value
            return "ok"
        
        elif field == "password":
            if len(value) < 6:
                return {"action": "retry", "message": "Password must be at least 6 characters:", "data": {}}
            self.data["password"] = value
            return "ok"
        
        else:
            if not value:
                return {"action": "retry", "message": f"{step['prompt']} cannot be empty:", "data": {}}
            self.data[field] = value
            return "ok"
    
    def _validate_personal_field(self, step: Dict, value: str) -> str:
        """Validate personal/employment fields"""
        field = step["field"]
        
        if field == "age":
            try:
                age = int(value)
                if age < 18 or age > 100:
                    return {"action": "retry", "message": "Please enter a valid age (18-100):", "data": {}}
                self.data["age"] = age
            except:
                return {"action": "retry", "message": "Please enter a number:", "data": {}}
        
        elif field in ["monthly_income", "annual_income"]:
            try:
                amount = float(value.replace('$', '').replace(',', ''))
                self.data[field] = amount
            except:
                return {"action": "retry", "message": "Please enter a valid amount:", "data": {}}
        
        elif field == "employment_start_date":
            try:
                datetime.strptime(value, "%Y-%m-%d")
                self.data["employment_start_date"] = value
            except:
                return {"action": "retry", "message": "Please use YYYY-MM-DD format (e.g., 2020-01-15):", "data": {}}
        
        else:
            if not value:
                return {"action": "retry", "message": f"{step['prompt']} cannot be empty:", "data": {}}
            self.data[field] = value
        
        return "ok"
    
    def _validate_debt_field(self, step: Dict, value: str) -> str:
        """Validate debt fields"""
        field = step["field"]
        
        if field in ["total_amount", "remaining_amount", "interest_rate", "minimum_payment"]:
            try:
                val = float(value.replace('$', '').replace(',', ''))
                self.current_debt[field] = val
            except:
                return {"action": "retry", "message": f"Please enter a valid number:", "data": {}}
        else:
            if not value:
                return {"action": "retry", "message": f"{step['prompt']} cannot be empty:", "data": {}}
            self.current_debt[field] = value
        
        return "ok"
    
    def _validate_preference_field(self, step: Dict, value: str) -> str:
        """Validate preference fields"""
        field = step["field"]
        val_lower = value.lower()
        
        if field == "risk_profile" and val_lower not in ['low', 'moderate', 'high']:
            return {"action": "retry", "message": "Please enter Low, Moderate, or High:", "data": {}}
        
        if field == "debt_strategy" and val_lower not in ['snowball', 'avalanche']:
            return {"action": "retry", "message": "Please enter Snowball or Avalanche:", "data": {}}
        
        return "ok"
    
    def _save_all(self) -> Dict[str, Any]:
        """Save all collected data"""
        try:
            # 1. Register user
            success = register_user(
                self.data['user_id'],
                self.data['first_name'],
                self.data['last_name'],
                self.data['email'],
                self.data['password'],
                self.data['secret_question'],
                self.data['secret_answer']
            )
            
            if not success:
                self.reset()
                return {"action": "error", "message": "❌ Registration failed", "data": {}}
            
            # 2. Save profile
            profile_fields = [
                'profession', 'current_role', 'employment_start_date', 'age',
                'gender', 'civil_status', 'number_of_children', 'monthly_income',
                'annual_income', 'retirement_age_target'
            ]
            profile_data = {k: self.data[k] for k in profile_fields if k in self.data}
            if profile_data:
                update_user_profile(self.data['user_id'], **profile_data)
            
            # 3. Save preferences
            pref_fields = ['risk_profile', 'debt_strategy', 'savings_priority']
            pref_data = {k: self.data[k] for k in pref_fields if k in self.data}
            if pref_data:
                update_user_preferences(self.data['user_id'], **pref_data)
            
            # 4. Save debts
            for debt in self.debts:
                add_user_debt(self.data['user_id'], debt)
            
            # 5. Create plan with goals
            short_term = self.data.get('short_term', '')
            long_term = self.data.get('long_term', '')
            if short_term or long_term:
                create_financial_plan(
                    self.data['user_id'],
                    f"{self.data['first_name']}'s Financial Plan",
                    {"description": short_term},
                    {"description": long_term}
                )
            
            # 6. Calculate retirement
            retirement = "Not calculated"
            if self.data.get('employment_start_date') and self.data.get('retirement_age_target'):
                retirement = self._calculate_retirement(
                    self.data['employment_start_date'],
                    int(self.data['retirement_age_target']),
                    int(self.data.get('age', 30))
                )
            
            user_id = self.data['user_id']
            first_name = self.data['first_name']
            self.reset()
            
            return {
                "action": "complete",
                "message": f"✅ Welcome, {first_name}! Your complete financial profile is ready.",
                "data": {
                    "user_id": user_id,
                    "retirement_date": retirement,
                    "profile_complete": True
                }
            }
            
        except Exception as e:
            print(f"Save error: {e}")
            self.reset()
            return {"action": "error", "message": "❌ An error occurred", "data": {}}
    
    def _calculate_retirement(self, start_date: str, target_age: int, current_age: int) -> str:
        """Calculate estimated retirement date"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            years_to_retirement = target_age - current_age
            retirement = start.replace(year=start.year + years_to_retirement + 35)
            return retirement.strftime("%Y-%m-%d")
        except:
            return datetime.now().replace(year=datetime.now().year + 30).strftime("%Y-%m-%d")