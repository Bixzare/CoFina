"""
Financial Calculator Tools for CoFina - Interest calculations and financial formulas
"""

from typing import Dict, Any, Optional
from langchain.tools import tool
import math

@tool
def calculate_simple_interest(
    principal: float,
    rate: float,
    time_years: float
) -> Dict[str, Any]:
    """
    Calculate simple interest.
    
    Simple Interest Formula: I = P * r * t
    Total Amount: A = P + I
    
    Args:
        principal: Initial amount (P)
        rate: Annual interest rate as percentage (e.g., 5 for 5%)
        time_years: Time in years (t)
    
    Returns:
        Dict with interest and total amount
    """
    rate_decimal = rate / 100
    
    interest = principal * rate_decimal * time_years
    total = principal + interest
    
    return {
        "calculation_type": "Simple Interest",
        "principal": round(principal, 2),
        "rate": f"{rate}%",
        "time_years": time_years,
        "interest_earned": round(interest, 2),
        "total_amount": round(total, 2),
        "formula": f"A = P(1 + rt) = {principal}(1 + {rate_decimal} × {time_years})"
    }

@tool
def calculate_compound_interest(
    principal: float,
    rate: float,
    time_years: float,
    compounds_per_year: int = 12,
    monthly_contribution: float = 0
) -> Dict[str, Any]:
    """
    Calculate compound interest with optional monthly contributions.
    
    Compound Interest Formula: A = P(1 + r/n)^(nt)
    With contributions: FV = P(1 + r/n)^(nt) + PMT * ((1 + r/n)^(nt) - 1) / (r/n)
    
    Args:
        principal: Initial amount (P)
        rate: Annual interest rate as percentage (e.g., 5 for 5%)
        time_years: Time in years (t)
        compounds_per_year: Number of times interest compounds per year (n)
                          Common values: 1 (annual), 4 (quarterly), 12 (monthly), 365 (daily)
        monthly_contribution: Additional monthly contribution (PMT)
    
    Returns:
        Dict with compound interest calculation
    """
    rate_decimal = rate / 100
    rate_per_period = rate_decimal / compounds_per_year
    total_periods = time_years * compounds_per_year
    
    # Base compound interest
    if compounds_per_year > 0:
        compound_factor = (1 + rate_per_period) ** total_periods
        future_value = principal * compound_factor
    else:
        return {"error": "compounds_per_year must be positive"}
    
    # Add monthly contributions if any
    total_contributions = 0
    if monthly_contribution > 0 and rate_per_period > 0:
        # Future value of annuity formula
        annuity_factor = (compound_factor - 1) / rate_per_period
        contributions_fv = monthly_contribution * annuity_factor
        future_value += contributions_fv
        total_contributions = monthly_contribution * total_periods
    elif monthly_contribution > 0:
        # No interest, just add contributions
        future_value += monthly_contribution * total_periods
        total_contributions = monthly_contribution * total_periods
    
    interest_earned = future_value - principal - total_contributions
    
    # Calculate effective annual rate
    effective_annual_rate = ((1 + rate_per_period) ** compounds_per_year - 1) * 100
    
    # Year-by-year projection
    projection = []
    for year in range(1, int(time_years) + 1):
        year_periods = year * compounds_per_year
        year_factor = (1 + rate_per_period) ** year_periods
        year_value = principal * year_factor
        
        if monthly_contribution > 0 and rate_per_period > 0:
            year_annuity = (year_factor - 1) / rate_per_period
            year_value += monthly_contribution * year_annuity
        elif monthly_contribution > 0:
            year_value += monthly_contribution * year_periods
        
        projection.append({
            "year": year,
            "value": round(year_value, 2),
            "gain": round(year_value - principal - (monthly_contribution * year_periods) if monthly_contribution > 0 else year_value - principal, 2)
        })
    
    return {
        "calculation_type": "Compound Interest" + (" with Monthly Contributions" if monthly_contribution > 0 else ""),
        "principal": round(principal, 2),
        "rate": f"{rate}%",
        "time_years": time_years,
        "compounds_per_year": compounds_per_year,
        "monthly_contribution": round(monthly_contribution, 2) if monthly_contribution > 0 else None,
        "total_contributions": round(principal + total_contributions, 2),
        "interest_earned": round(interest_earned, 2),
        "future_value": round(future_value, 2),
        "effective_annual_rate": f"{effective_annual_rate:.2f}%",
        "doubling_time_years": round(math.log(2) / math.log(1 + rate_decimal), 2) if rate > 0 else None,
        "projection": projection,
        "formula_note": "Includes compounding and regular contributions" if monthly_contribution > 0 else "Standard compound interest"
    }

@tool
def calculate_loan_payment(
    loan_amount: float,
    annual_rate: float,
    loan_term_years: float,
    payments_per_year: int = 12
) -> Dict[str, Any]:
    """
    Calculate monthly loan payments and total interest.
    
    Loan Payment Formula: PMT = P * (r(1+r)^n) / ((1+r)^n - 1)
    
    Args:
        loan_amount: Principal loan amount (P)
        annual_rate: Annual interest rate as percentage
        loan_term_years: Loan term in years
        payments_per_year: Number of payments per year (usually 12 for monthly)
    
    Returns:
        Dict with payment schedule and totals
    """
    rate_decimal = annual_rate / 100
    rate_per_period = rate_decimal / payments_per_year
    total_payments = loan_term_years * payments_per_year
    
    if rate_per_period == 0:
        # Zero interest loan
        payment = loan_amount / total_payments
        total_paid = loan_amount
        total_interest = 0
    else:
        # Standard loan payment formula
        payment = loan_amount * (rate_per_period * (1 + rate_per_period) ** total_payments) / ((1 + rate_per_period) ** total_payments - 1)
        total_paid = payment * total_payments
        total_interest = total_paid - loan_amount
    
    # Amortization schedule (first 12 payments)
    schedule = []
    balance = loan_amount
    
    for i in range(1, min(13, int(total_payments) + 1)):
        interest_payment = balance * rate_per_period
        principal_payment = payment - interest_payment
        balance -= principal_payment
        
        schedule.append({
            "payment_number": i,
            "payment": round(payment, 2),
            "principal": round(principal_payment, 2),
            "interest": round(interest_payment, 2),
            "remaining_balance": round(max(balance, 0), 2)
        })
    
    return {
        "calculation_type": "Loan Payment Calculator",
        "loan_amount": round(loan_amount, 2),
        "annual_rate": f"{annual_rate}%",
        "loan_term_years": loan_term_years,
        "payments_per_year": payments_per_year,
        "monthly_payment": round(payment, 2),
        "total_payments": total_payments,
        "total_paid": round(total_paid, 2),
        "total_interest": round(total_interest, 2),
        "interest_percentage": round((total_interest / loan_amount) * 100, 2) if loan_amount > 0 else 0,
        "amortization_schedule": schedule,
        "first_year_summary": {
            "total_payments": round(payment * min(12, total_payments), 2),
            "total_principal": sum(s["principal"] for s in schedule),
            "total_interest": sum(s["interest"] for s in schedule)
        }
    }

@tool
def calculate_retirement_savings(
    current_age: int,
    retirement_age: int,
    current_savings: float,
    monthly_contribution: float,
    expected_return_rate: float = 7.0,
    current_annual_income: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate retirement savings projections.
    
    Args:
        current_age: Current age in years
        retirement_age: Planned retirement age
        current_savings: Current retirement savings
        monthly_contribution: Monthly contribution to retirement
        expected_return_rate: Expected annual return rate as percentage
        current_annual_income: Current annual income (for replacement ratio)
    
    Returns:
        Dict with retirement projections
    """
    years_to_retirement = retirement_age - current_age
    
    if years_to_retirement <= 0:
        return {"error": "Retirement age must be greater than current age"}
    
    # Use compound interest with monthly contributions
    result = calculate_compound_interest(
        principal=current_savings,
        rate=expected_return_rate,
        time_years=years_to_retirement,
        compounds_per_year=12,
        monthly_contribution=monthly_contribution
    )
    
    # Calculate safe withdrawal amount (4% rule)
    future_value = result["future_value"]
    safe_withdrawal_annual = future_value * 0.04
    
    # Income replacement ratio
    replacement_ratio = None
    if current_annual_income and current_annual_income > 0:
        replacement_ratio = (safe_withdrawal_annual / current_annual_income) * 100
    
    # Inflation-adjusted future value (assuming 3% inflation)
    inflation_rate = 3.0
    inflation_adjusted = future_value / ((1 + inflation_rate/100) ** years_to_retirement)
    
    # Recommended savings rate based on age
    recommended_ranges = {
        "30": {"low": 0.10, "high": 0.15},
        "40": {"low": 0.15, "high": 0.20},
        "50": {"low": 0.20, "high": 0.25},
        "60": {"low": 0.25, "high": 0.30}
    }
    
    current_savings_rate = None
    if current_annual_income and current_annual_income > 0:
        current_savings_rate = (monthly_contribution * 12 / current_annual_income) * 100
    
    return {
        "calculation_type": "Retirement Planning",
        "current_age": current_age,
        "retirement_age": retirement_age,
        "years_to_retirement": years_to_retirement,
        "current_savings": round(current_savings, 2),
        "monthly_contribution": round(monthly_contribution, 2),
        "expected_return_rate": f"{expected_return_rate}%",
        "projected_savings_at_retirement": round(future_value, 2),
        "inflation_adjusted_value": round(inflation_adjusted, 2),
        "safe_withdrawal_amount_annual": round(safe_withdrawal_annual, 2),
        "safe_withdrawal_amount_monthly": round(safe_withdrawal_annual / 12, 2),
        "income_replacement_ratio": f"{replacement_ratio:.1f}%" if replacement_ratio else None,
        "current_savings_rate": f"{current_savings_rate:.1f}%" if current_savings_rate else None,
        "recommended_savings_rate": recommended_ranges.get(str(current_age)[0] + "0", {"low": 0.15, "high": 0.20}),
        "projection": result.get("projection", [])
    }

@tool
def calculate_investment_growth(
    initial_investment: float,
    monthly_addition: float,
    years: int,
    expected_return: float,
    volatility: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate investment growth with optional volatility scenarios.
    
    Args:
        initial_investment: Initial investment amount
        monthly_addition: Monthly addition to investment
        years: Investment time horizon in years
        expected_return: Expected annual return as percentage
        volatility: Expected annual volatility as percentage (optional)
    
    Returns:
        Dict with growth projections and scenarios
    """
    # Base calculation
    base_result = calculate_compound_interest(
        principal=initial_investment,
        rate=expected_return,
        time_years=years,
        compounds_per_year=12,
        monthly_contribution=monthly_addition
    )
    
    total_contributions = initial_investment + (monthly_addition * 12 * years)
    
    result = {
        "initial_investment": round(initial_investment, 2),
        "monthly_addition": round(monthly_addition, 2),
        "time_horizon_years": years,
        "expected_return": f"{expected_return}%",
        "total_contributions": round(total_contributions, 2),
        "projected_value": base_result["future_value"],
        "projected_gain": round(base_result["future_value"] - total_contributions, 2),
        "return_multiple": round(base_result["future_value"] / total_contributions, 2),
        "projection": base_result["projection"]
    }
    
    # Add volatility scenarios if requested
    if volatility:
        # Conservative scenario (lower return)
        conservative_return = max(0, expected_return - volatility)
        conservative = calculate_compound_interest(
            principal=initial_investment,
            rate=conservative_return,
            time_years=years,
            compounds_per_year=12,
            monthly_contribution=monthly_addition
        )
        
        # Optimistic scenario (higher return)
        optimistic_return = expected_return + volatility
        optimistic = calculate_compound_interest(
            principal=initial_investment,
            rate=optimistic_return,
            time_years=years,
            compounds_per_year=12,
            monthly_contribution=monthly_addition
        )
        
        result["volatility"] = f"{volatility}%"
        result["scenarios"] = {
            "conservative": {
                "return": f"{conservative_return:.1f}%",
                "value": round(conservative["future_value"], 2),
                "gain": round(conservative["future_value"] - total_contributions, 2)
            },
            "expected": {
                "return": f"{expected_return}%",
                "value": round(base_result["future_value"], 2),
                "gain": round(base_result["future_value"] - total_contributions, 2)
            },
            "optimistic": {
                "return": f"{optimistic_return:.1f}%",
                "value": round(optimistic["future_value"], 2),
                "gain": round(optimistic["future_value"] - total_contributions, 2)
            }
        }
        
        # Calculate range
        result["value_range"] = {
            "low": round(conservative["future_value"], 2),
            "high": round(optimistic["future_value"], 2),
            "spread": round(optimistic["future_value"] - conservative["future_value"], 2)
        }
    
    return result

@tool
def calculate_budget_allocation(
    monthly_income: float,
    needs_percent: float = 50,
    wants_percent: float = 30,
    savings_percent: float = 20
) -> Dict[str, Any]:
    """
    Calculate budget allocation using the 50/30/20 rule.
    
    Args:
        monthly_income: Monthly after-tax income
        needs_percent: Percentage for needs (default 50)
        wants_percent: Percentage for wants (default 30)
        savings_percent: Percentage for savings (default 20)
    
    Returns:
        Dict with budget allocation
    """
    # Validate percentages
    total = needs_percent + wants_percent + savings_percent
    if abs(total - 100) > 0.01:
        return {"error": f"Percentages must sum to 100 (currently {total})"}
    
    needs_amount = monthly_income * (needs_percent / 100)
    wants_amount = monthly_income * (wants_percent / 100)
    savings_amount = monthly_income * (savings_percent / 100)
    
    # Typical needs categories
    needs_breakdown = {
        "Housing": needs_amount * 0.5,
        "Utilities": needs_amount * 0.1,
        "Groceries": needs_amount * 0.2,
        "Transportation": needs_amount * 0.1,
        "Insurance": needs_amount * 0.05,
        "Minimum Debt Payments": needs_amount * 0.05
    }
    
    # Typical wants categories
    wants_breakdown = {
        "Dining Out": wants_amount * 0.3,
        "Entertainment": wants_amount * 0.2,
        "Shopping": wants_amount * 0.2,
        "Travel": wants_amount * 0.2,
        "Subscriptions": wants_amount * 0.1
    }
    
    # Savings allocation
    savings_breakdown = {
        "Emergency Fund": savings_amount * 0.4,
        "Retirement": savings_amount * 0.3,
        "Short-term Goals": savings_amount * 0.2,
        "Investments": savings_amount * 0.1
    }
    
    return {
        "monthly_income": round(monthly_income, 2),
        "allocation": {
            "needs": {
                "percent": needs_percent,
                "amount": round(needs_amount, 2),
                "breakdown": {k: round(v, 2) for k, v in needs_breakdown.items()}
            },
            "wants": {
                "percent": wants_percent,
                "amount": round(wants_amount, 2),
                "breakdown": {k: round(v, 2) for k, v in wants_breakdown.items()}
            },
            "savings": {
                "percent": savings_percent,
                "amount": round(savings_amount, 2),
                "breakdown": {k: round(v, 2) for k, v in savings_breakdown.items()}
            }
        },
        "annual_savings": round(savings_amount * 12, 2),
        "formula": f"{needs_percent}/{wants_percent}/{savings_percent} Rule"
    }

@tool
def calculate_emergency_fund(
    monthly_expenses: float,
    months: int = 6
) -> Dict[str, Any]:
    """
    Calculate recommended emergency fund amount.
    
    Args:
        monthly_expenses: Average monthly essential expenses
        months: Number of months to cover (typically 3-6)
    
    Returns:
        Dict with emergency fund recommendations
    """
    if months < 1:
        return {"error": "Months must be at least 1"}
    
    fund_amount = monthly_expenses * months
    
    # Tiered recommendations
    tiers = {
        "minimum": {
            "months": 3,
            "amount": monthly_expenses * 3,
            "suitable_for": "Stable job, dual income, low expenses"
        },
        "moderate": {
            "months": 6,
            "amount": monthly_expenses * 6,
            "suitable_for": "Most people, moderate job security"
        },
        "comprehensive": {
            "months": 9,
            "amount": monthly_expenses * 9,
            "suitable_for": "Variable income, self-employed, single income"
        },
        "extensive": {
            "months": 12,
            "amount": monthly_expenses * 12,
            "suitable_for": "High-risk profession, health concerns"
        }
    }
    
    # Calculate saving timeline at different rates
    saving_scenarios = {}
    for rate in [100, 250, 500, 1000]:
        months_to_save = fund_amount / rate if rate > 0 else float('inf')
        saving_scenarios[f"${rate}/month"] = {
            "months": round(months_to_save, 1),
            "years": round(months_to_save / 12, 1)
        }
    
    return {
        "monthly_expenses": round(monthly_expenses, 2),
        "target_months": months,
        "recommended_fund": round(fund_amount, 2),
        "tier_recommendations": tiers,
        "saving_scenarios": saving_scenarios,
        "advice": f"Your emergency fund should cover {months} months of essential expenses. This provides a safety net for job loss, medical emergencies, or unexpected expenses."
    }

@tool
def calculate_debt_payoff(
    debts: str,
    monthly_payment: float,
    strategy: str = "avalanche"
) -> Dict[str, Any]:
    """
    Calculate debt payoff timeline using different strategies.
    
    Args:
        debts: JSON string with list of debts (name, balance, rate, min_payment)
        monthly_payment: Total monthly payment available
        strategy: 'avalanche' (highest interest first) or 'snowball' (smallest balance first)
    
    Returns:
        Dict with payoff timeline and interest savings
    """
    import json
    
    try:
        # Parse debts
        if isinstance(debts, str):
            debt_list = json.loads(debts)
        else:
            debt_list = debts
        
        if not debt_list:
            return {"error": "No debts provided"}
        
        # Sort debts based on strategy
        if strategy == "avalanche":
            # Highest interest rate first
            debt_list.sort(key=lambda x: x.get("rate", 0), reverse=True)
        elif strategy == "snowball":
            # Smallest balance first
            debt_list.sort(key=lambda x: x.get("balance", 0))
        else:
            return {"error": f"Unknown strategy: {strategy}. Use 'avalanche' or 'snowball'"}
        
        # Calculate total minimum payments
        total_min_payment = sum(d.get("min_payment", 0) for d in debt_list)
        
        if monthly_payment < total_min_payment:
            return {
                "error": f"Monthly payment (${monthly_payment}) is less than total minimum payments (${total_min_payment:.2f})",
                "suggestion": "Consider debt consolidation or increasing income"
            }
        
        # Extra payment available
        extra_payment = monthly_payment - total_min_payment
        
        # Simulate payoff
        remaining_debts = []
        for d in debt_list:
            remaining_debts.append({
                "name": d.get("name", "Unknown"),
                "balance": d.get("balance", 0),
                "rate": d.get("rate", 0),
                "min_payment": d.get("min_payment", 0)
            })
        
        total_interest_paid = 0
        months = 0
        monthly_schedule = []
        
        while remaining_debts and months < 1200:  # Max 100 years
            months += 1
            payment_available = monthly_payment
            month_interest = 0
            
            # Apply minimum payments to all debts
            for debt in remaining_debts:
                # Calculate interest for this month
                monthly_rate = debt["rate"] / 100 / 12
                interest = debt["balance"] * monthly_rate
                month_interest += interest
                debt["balance"] += interest
                
                # Apply minimum payment
                min_payment = min(debt["min_payment"], debt["balance"])
                debt["balance"] -= min_payment
                payment_available -= min_payment
            
            # Apply extra payment to first debt (snowball/avalanche order)
            if remaining_debts and payment_available > 0:
                extra = min(payment_available, remaining_debts[0]["balance"])
                remaining_debts[0]["balance"] -= extra
                payment_available -= extra
            
            total_interest_paid += month_interest
            
            # Record schedule (every 6 months)
            if months % 6 == 0 or months <= 12:
                monthly_schedule.append({
                    "month": months,
                    "remaining_debt": sum(d["balance"] for d in remaining_debts),
                    "interest_paid_ytd": round(month_interest * 6 if months % 6 == 0 else month_interest * months, 2)
                })
            
            # Remove paid off debts
            remaining_debts = [d for d in remaining_debts if d["balance"] > 0.01]
        
        years = months / 12
        
        # Calculate comparison with minimum payments only
        min_only_months = 0
        min_only_interest = 0
        # Simplified calculation - in production would do full simulation
        
        return {
            "strategy": strategy.capitalize(),
            "total_debt": sum(d.get("balance", 0) for d in debt_list),
            "monthly_payment": monthly_payment,
            "total_min_payment": total_min_payment,
            "extra_payment": extra_payment,
            "months_to_payoff": months,
            "years_to_payoff": round(years, 1),
            "total_interest_paid": round(total_interest_paid, 2),
            "interest_saved_vs_minimum": "Varies based on strategy",
            "payoff_schedule": monthly_schedule[:24],  # First 24 months or until payoff
            "recommendation": f"With the {strategy} strategy, you'll be debt-free in {round(years, 1)} years"
        }
        
    except Exception as e:
        return {"error": f"Failed to calculate debt payoff: {str(e)}"}

# Export all tools
FINANCIAL_CALCULATOR_TOOLS = [
    calculate_simple_interest,
    calculate_compound_interest,
    calculate_loan_payment,
    calculate_retirement_savings,
    calculate_investment_growth,
    calculate_budget_allocation,
    calculate_emergency_fund,
    calculate_debt_payoff
]