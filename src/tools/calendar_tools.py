"""
Calendar Tools for CoFina - Financial calendar and deadline management
"""

from typing import Dict, Any, List, Optional
from langchain.tools import tool
from datetime import datetime, timedelta, date
import calendar
import json

@tool
def get_financial_calendar(year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get important financial dates and deadlines for the year.
    
    Args:
        year: Year to get calendar for (defaults to current year)
    
    Returns:
        Dict with important financial dates
    """
    if year is None:
        year = datetime.now().year
    
    # US Tax deadlines (approximate - would need localization in production)
    tax_deadline = datetime(year, 4, 15)
    # Adjust for weekends
    if tax_deadline.weekday() == 5:  # Saturday
        tax_deadline += timedelta(days=2)
    elif tax_deadline.weekday() == 6:  # Sunday
        tax_deadline += timedelta(days=1)
    
    # Quarterly estimated tax deadlines
    estimated_tax = [
        datetime(year, 4, 15),  # Q1
        datetime(year, 6, 15),  # Q2
        datetime(year, 9, 15),  # Q3
        datetime(year, 1, 15) if year == datetime.now().year else datetime(year + 1, 1, 15)  # Q4 of previous year
    ]
    
    # IRA contribution deadline (typically tax day)
    ira_deadline = tax_deadline
    
    # Financial quarters
    quarters = {
        "Q1": {"start": datetime(year, 1, 1), "end": datetime(year, 3, 31)},
        "Q2": {"start": datetime(year, 4, 1), "end": datetime(year, 6, 30)},
        "Q3": {"start": datetime(year, 7, 1), "end": datetime(year, 9, 30)},
        "Q4": {"start": datetime(year, 10, 1), "end": datetime(year, 12, 31)}
    }
    
    # Format dates
    return {
        "year": year,
        "tax_deadline": tax_deadline.strftime("%B %d, %Y"),
        "ira_contribution_deadline": ira_deadline.strftime("%B %d, %Y"),
        "estimated_tax_dates": [d.strftime("%B %d, %Y") for d in estimated_tax],
        "quarters": {
            q: {
                "start": dates["start"].strftime("%B %d, %Y"),
                "end": dates["end"].strftime("%B %d, %Y")
            } for q, dates in quarters.items()
        },
        "holidays_when_markets_closed": [
            "New Year's Day",
            "Martin Luther King Jr. Day",
            "Presidents' Day",
            "Good Friday",
            "Memorial Day",
            "Juneteenth",
            "Independence Day",
            "Labor Day",
            "Thanksgiving Day",
            "Christmas Day"
        ]
    }

@tool
def calculate_pay_periods(
    start_date: str,
    pay_frequency: str = "bi-weekly",
    num_periods: int = 26
) -> Dict[str, Any]:
    """
    Calculate pay periods and dates.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        pay_frequency: 'weekly', 'bi-weekly', 'semi-monthly', or 'monthly'
        num_periods: Number of periods to calculate
    
    Returns:
        Dict with pay period dates and info
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        
        frequency_map = {
            "weekly": timedelta(days=7),
            "bi-weekly": timedelta(days=14),
            "semi-monthly": "semi",  # Special case
            "monthly": "monthly"      # Special case
        }
        
        if pay_frequency not in frequency_map:
            return {"error": f"Invalid pay frequency. Use: {', '.join(frequency_map.keys())}"}
        
        pay_dates = []
        
        if pay_frequency == "semi-monthly":
            # Semi-monthly: 1st and 15th (or next business day)
            current = start
            for i in range(num_periods):
                pay_dates.append(current.strftime("%Y-%m-%d"))
                
                # Alternate between 1st and 15th
                if current.day <= 15:
                    # Next pay is last day of month or 1st of next month
                    if current.month == 12:
                        next_date = datetime(current.year + 1, 1, 1)
                    else:
                        next_date = datetime(current.year, current.month + 1, 1)
                else:
                    # Next pay is 15th of next month
                    if current.month == 12:
                        next_date = datetime(current.year + 1, 1, 15)
                    else:
                        next_date = datetime(current.year, current.month + 1, 15)
                
                current = next_date
        
        elif pay_frequency == "monthly":
            # Monthly: same day each month
            current = start
            for i in range(num_periods):
                pay_dates.append(current.strftime("%Y-%m-%d"))
                
                # Add a month
                if current.month == 12:
                    current = datetime(current.year + 1, 1, current.day)
                else:
                    # Handle months with fewer days
                    next_month = current.month + 1
                    next_year = current.year
                    if next_month > 12:
                        next_month = 1
                        next_year += 1
                    
                    # Get last day of next month if day doesn't exist
                    last_day = calendar.monthrange(next_year, next_month)[1]
                    day = min(current.day, last_day)
                    
                    current = datetime(next_year, next_month, day)
        
        else:
            # Weekly or bi-weekly
            delta = frequency_map[pay_frequency]
            current = start
            for i in range(num_periods):
                pay_dates.append(current.strftime("%Y-%m-%d"))
                current += delta
        
        # Calculate annual summary
        total_pay_periods = {
            "weekly": 52,
            "bi-weekly": 26,
            "semi-monthly": 24,
            "monthly": 12
        }.get(pay_frequency, 0)
        
        return {
            "pay_frequency": pay_frequency,
            "start_date": start.strftime("%Y-%m-%d"),
            "num_periods": len(pay_dates),
            "pay_dates": pay_dates[:10],  # First 10 dates
            "annual_periods": total_pay_periods,
            "note": f"Showing first {min(10, len(pay_dates))} of {len(pay_dates)} dates" if len(pay_dates) > 10 else None
        }
        
    except ValueError as e:
        return {"error": f"Invalid date format. Use YYYY-MM-DD: {str(e)}"}

@tool
def get_bill_reminders(
    bills: str,
    months_ahead: int = 3
) -> Dict[str, Any]:
    """
    Get bill reminders and due dates.
    
    Args:
        bills: JSON string with bill information
        months_ahead: How many months to look ahead
    
    Returns:
        Dict with upcoming bills and reminders
    """
    try:
        # Parse bills (expects JSON array of bills)
        if isinstance(bills, str):
            bill_list = json.loads(bills)
        else:
            bill_list = bills
        
        if not isinstance(bill_list, list):
            bill_list = [bill_list]
        
        today = datetime.now()
        reminders = []
        
        for bill in bill_list:
            name = bill.get("name", "Unknown")
            amount = bill.get("amount", 0)
            due_day = bill.get("due_day")  # Day of month (1-31)
            due_date_str = bill.get("due_date")  # Specific date
            
            if due_date_str:
                # Specific due date
                due_date = datetime.strptime(due_date_str, "%Y-%m-%d")
                if due_date >= today:
                    days_until = (due_date - today).days
                    reminders.append({
                        "name": name,
                        "amount": amount,
                        "due_date": due_date.strftime("%Y-%m-%d"),
                        "days_until": days_until,
                        "reminder_date": (due_date - timedelta(days=3)).strftime("%Y-%m-%d")
                    })
            
            elif due_day:
                # Recurring monthly bill
                for month_offset in range(months_ahead):
                    # Calculate date for this month
                    month = today.month + month_offset
                    year = today.year
                    while month > 12:
                        month -= 12
                        year += 1
                    
                    # Get valid day (handle months with fewer days)
                    last_day = calendar.monthrange(year, month)[1]
                    day = min(due_day, last_day)
                    
                    due_date = datetime(year, month, day)
                    
                    if due_date >= today:
                        days_until = (due_date - today).days
                        reminders.append({
                            "name": name,
                            "amount": amount,
                            "due_date": due_date.strftime("%Y-%m-%d"),
                            "days_until": days_until,
                            "reminder_date": (due_date - timedelta(days=3)).strftime("%Y-%m-%d"),
                            "recurring": True
                        })
        
        # Sort by due date
        reminders.sort(key=lambda x: x["days_until"])
        
        return {
            "today": today.strftime("%Y-%m-%d"),
            "total_upcoming": len(reminders),
            "total_amount": sum(r["amount"] for r in reminders if "amount" in r),
            "reminders": reminders[:20],  # Limit to 20
            "next_3_days": [r for r in reminders if r["days_until"] <= 3],
            "next_week": [r for r in reminders if r["days_until"] <= 7]
        }
        
    except Exception as e:
        return {"error": f"Failed to process bills: {str(e)}"}

@tool
def calculate_savings_timeline(
    target_amount: float,
    current_savings: float,
    monthly_contribution: float,
    annual_return_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate how long to reach a savings goal.
    
    Args:
        target_amount: Target savings amount
        current_savings: Current savings
        monthly_contribution: Monthly contribution amount
        annual_return_rate: Expected annual return rate (as decimal, e.g., 0.05 for 5%)
    
    Returns:
        Dict with timeline and projections
    """
    if target_amount <= current_savings:
        return {
            "goal_already_met": True,
            "current_savings": current_savings,
            "target_amount": target_amount,
            "surplus": current_savings - target_amount
        }
    
    if monthly_contribution <= 0 and annual_return_rate <= 0:
        return {
            "error": "Cannot reach goal without contributions or returns"
        }
    
    months = 0
    savings = current_savings
    monthly_rate = annual_return_rate / 12
    
    # Project month by month
    projection = []
    
    while savings < target_amount and months < 1200:  # Max 100 years
        months += 1
        
        # Add monthly contribution
        savings += monthly_contribution
        
        # Apply monthly return
        if monthly_rate > 0:
            savings *= (1 + monthly_rate)
        
        # Store projection points (every 6 months)
        if months % 6 == 0 or months < 12:
            projection.append({
                "month": months,
                "savings": round(savings, 2),
                "years": round(months / 12, 1)
            })
    
    years = months / 12
    
    return {
        "target_amount": target_amount,
        "current_savings": current_savings,
        "monthly_contribution": monthly_contribution,
        "annual_return_rate": f"{annual_return_rate * 100:.1f}%",
        "months_to_goal": months,
        "years_to_goal": round(years, 1),
        "total_contributions": current_savings + (monthly_contribution * months),
        "total_earned": round(savings - (current_savings + (monthly_contribution * months)), 2),
        "projection": projection
    }

@tool
def get_next_payday(
    pay_dates: str
) -> Dict[str, Any]:
    """
    Get the next payday from a list of pay dates.
    
    Args:
        pay_dates: JSON array of pay dates in YYYY-MM-DD format
    
    Returns:
        Dict with next payday info
    """
    try:
        if isinstance(pay_dates, str):
            date_list = json.loads(pay_dates)
        else:
            date_list = pay_dates
        
        today = datetime.now().date()
        
        # Parse dates
        parsed_dates = []
        for d in date_list:
            if isinstance(d, str):
                parsed_dates.append(datetime.strptime(d, "%Y-%m-%d").date())
            else:
                parsed_dates.append(d)
        
        # Find next payday
        future_dates = [d for d in parsed_dates if d >= today]
        
        if not future_dates:
            return {
                "message": "No future paydays found in the list",
                "last_payday": max(parsed_dates).strftime("%Y-%m-%d") if parsed_dates else None
            }
        
        next_pay = min(future_dates)
        days_until = (next_pay - today).days
        
        # Calculate estimated amount (simplified)
        # In production, this would come from user's salary info
        estimated_amount = None
        
        return {
            "next_payday": next_pay.strftime("%Y-%m-%d"),
            "days_until": days_until,
            "day_of_week": next_pay.strftime("%A"),
            "is_this_week": days_until <= 7,
            "is_this_month": days_until <= 30,
            "total_upcoming": len(future_dates)
        }
        
    except Exception as e:
        return {"error": f"Failed to process pay dates: {str(e)}"}

# Export all tools
CALENDAR_TOOLS = [
    get_financial_calendar,
    calculate_pay_periods,
    get_bill_reminders,
    calculate_savings_timeline,
    get_next_payday
]