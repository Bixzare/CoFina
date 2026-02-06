# src/tools/dateTime.py

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from langchain.tools import tool
import calendar


@tool
def get_current_time(timezone_offset: Optional[int] = None) -> Dict[str, Any]:
    """
    Get the current time and date.
    
    Args:
        timezone_offset: Optional timezone offset in hours from UTC.
                        Positive for east, negative for west.
    
    Returns:
        Dict with current date and time information.
    """
    from datetime import timezone
    
    if timezone_offset is not None:
        tz = timezone(timedelta(hours=timezone_offset))
        now = datetime.now(tz)
    else:
        now = datetime.now()
    
    return {
        "current_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "current_date": now.strftime("%B %d, %Y"),
        "current_time": now.strftime("%I:%M %p"),
        "day_of_week": now.strftime("%A"),
        "iso_format": now.isoformat(),
        "timezone": str(now.tzinfo) if now.tzinfo else "Local",
        "unix_timestamp": int(now.timestamp())
    }


@tool
def get_date_difference(
    start_date: str,
    end_date: str,
    format: str = "%Y-%m-%d"
) -> Dict[str, Any]:
    """
    Calculate the difference between two dates.
    
    Args:
        start_date: Start date string
        end_date: End date string
        format: Date format (default: YYYY-MM-DD)
    
    Returns:
        Dict with difference in days, weeks, months, etc.
    """
    try:
        start = datetime.strptime(start_date, format)
        end = datetime.strptime(end_date, format)
        
        if end < start:
            start, end = end, start
            swapped = True
        else:
            swapped = False
        
        delta = end - start
        
        # Calculate approximate months and years
        total_days = delta.days
        total_weeks = total_days / 7
        
        # Calculate months and years
        months = (end.year - start.year) * 12 + (end.month - start.month)
        years = months / 12
        
        return {
            "days": total_days,
            "weeks": round(total_weeks, 1),
            "months": months,
            "years": round(years, 1),
            "business_days": _count_business_days(start, end),
            "is_swapped": swapped,
            "start_date": start.strftime("%B %d, %Y"),
            "end_date": end.strftime("%B %d, %Y")
        }
    except ValueError as e:
        return {"error": f"Invalid date format. Use format: {format}", "details": str(e)}


@tool
def add_to_date(
    date: str,
    days: int = 0,
    weeks: int = 0,
    months: int = 0,
    years: int = 0,
    format: str = "%Y-%m-%d"
) -> Dict[str, Any]:
    """
    Add days, weeks, months, or years to a date.
    
    Args:
        date: Starting date string
        days: Number of days to add
        weeks: Number of weeks to add
        months: Number of months to add
        years: Number of years to add
        format: Date format (default: YYYY-MM-DD)
    
    Returns:
        Dict with new date information.
    """
    try:
        base_date = datetime.strptime(date, format)
        
        # Add days and weeks
        new_date = base_date + timedelta(days=days + (weeks * 7))
        
        # Add months and years
        if months != 0 or years != 0:
            month = new_date.month + months + (years * 12)
            year = new_date.year
            while month > 12:
                month -= 12
                year += 1
            while month < 1:
                month += 12
                year -= 1
            
            # Handle day adjustment for months with fewer days
            day = min(new_date.day, calendar.monthrange(year, month)[1])
            new_date = new_date.replace(year=year, month=month, day=day)
        
        return {
            "original_date": base_date.strftime("%B %d, %Y"),
            "new_date": new_date.strftime("%B %d, %Y"),
            "iso_format": new_date.isoformat(),
            "day_of_week": new_date.strftime("%A"),
            "days_added": days + (weeks * 7),
            "months_added": months + (years * 12)
        }
    except ValueError as e:
        return {"error": f"Invalid date format. Use format: {format}", "details": str(e)}


@tool
def get_day_info(date: str, format: str = "%Y-%m-%d") -> Dict[str, Any]:
    """
    Get information about a specific date.
    
    Args:
        date: Date string
        format: Date format (default: YYYY-MM-DD)
    
    Returns:
        Dict with date information.
    """
    try:
        dt = datetime.strptime(date, format)
        
        return {
            "date": dt.strftime("%B %d, %Y"),
            "day_of_week": dt.strftime("%A"),
            "day_of_year": dt.timetuple().tm_yday,
            "week_of_year": dt.isocalendar()[1],
            "quarter": (dt.month - 1) // 3 + 1,
            "is_weekend": dt.weekday() >= 5,
            "is_leap_year": calendar.isleap(dt.year),
            "days_in_month": calendar.monthrange(dt.year, dt.month)[1],
            "iso_format": dt.isoformat()
        }
    except ValueError as e:
        return {"error": f"Invalid date format. Use format: {format}", "details": str(e)}


@tool
def calculate_age(birth_date: str, format: str = "%Y-%m-%d") -> Dict[str, Any]:
    """
    Calculate age from birth date.
    
    Args:
        birth_date: Birth date string
        format: Date format (default: YYYY-MM-DD)
    
    Returns:
        Dict with age information.
    """
    try:
        birth = datetime.strptime(birth_date, format)
        today = datetime.now()
        
        age = today.year - birth.year
        if (today.month, today.day) < (birth.month, birth.day):
            age -= 1
        
        # Calculate months and days
        if today.month >= birth.month:
            months = today.month - birth.month
        else:
            months = 12 + today.month - birth.month
        
        if today.day >= birth.day:
            days = today.day - birth.day
        else:
            # Borrow days from previous month
            last_month = today.replace(day=1) - timedelta(days=1)
            days = (last_month.day - birth.day) + today.day
            months -= 1
        
        # Calculate total days
        total_days = (today - birth).days
        
        # Calculate next birthday
        next_birthday_year = today.year
        next_birthday = birth.replace(year=next_birthday_year)
        if next_birthday < today:
            next_birthday = next_birthday.replace(year=next_birthday_year + 1)
        
        days_until_next = (next_birthday - today).days
        
        return {
            "birth_date": birth.strftime("%B %d, %Y"),
            "current_age": age,
            "age_months": months,
            "age_days": days,
            "total_days_old": total_days,
            "next_birthday": next_birthday.strftime("%B %d, %Y"),
            "days_until_next_birthday": days_until_next
        }
    except ValueError as e:
        return {"error": f"Invalid date format. Use format: {format}", "details": str(e)}


@tool
def get_financial_dates() -> Dict[str, Any]:
    """
    Get important financial dates (tax deadlines, quarters, etc.).
    
    Returns:
        Dict with financial date information.
    """
    today = datetime.now()
    current_year = today.year
    
    # US Tax deadlines (approximate)
    tax_deadline = datetime(current_year, 4, 15)
    if tax_deadline.weekday() == 5:  # Saturday
        tax_deadline += timedelta(days=2)
    elif tax_deadline.weekday() == 6:  # Sunday
        tax_deadline += timedelta(days=1)
    
    # Financial quarters
    quarters = {
        "Q1": {"start": datetime(current_year, 1, 1), "end": datetime(current_year, 3, 31)},
        "Q2": {"start": datetime(current_year, 4, 1), "end": datetime(current_year, 6, 30)},
        "Q3": {"start": datetime(current_year, 7, 1), "end": datetime(current_year, 9, 30)},
        "Q4": {"start": datetime(current_year, 10, 1), "end": datetime(current_year, 12, 31)}
    }
    
    # Current quarter
    current_quarter = (today.month - 1) // 3 + 1
    quarter_key = f"Q{current_quarter}"
    
    # Days until tax deadline
    if today > tax_deadline:
        next_tax = datetime(current_year + 1, 4, 15)
        days_until_tax = (next_tax - today).days
        tax_status = "passed"
    else:
        days_until_tax = (tax_deadline - today).days
        tax_status = "upcoming"
    
    return {
        "current_date": today.strftime("%B %d, %Y"),
        "current_quarter": quarter_key,
        "quarter_dates": {
            q: {
                "start": dates["start"].strftime("%B %d, %Y"),
                "end": dates["end"].strftime("%B %d, %Y")
            } for q, dates in quarters.items()
        },
        "tax_deadline": tax_deadline.strftime("%B %d, %Y"),
        "days_until_tax_deadline": days_until_tax,
        "tax_status": tax_status,
        "year_end": datetime(current_year, 12, 31).strftime("%B %d, %Y"),
        "days_remaining_in_year": (datetime(current_year, 12, 31) - today).days
    }


@tool
def calculate_compounding(
    principal: float,
    annual_rate: float,
    years: int,
    compounds_per_year: int = 12
) -> Dict[str, Any]:
    """
    Calculate compound interest for financial planning.
    
    Args:
        principal: Initial amount
        annual_rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        years: Number of years
        compounds_per_year: Number of times interest compounds per year
    
    Returns:
        Dict with compounding results.
    """
    rate_per_period = annual_rate / compounds_per_year
    total_periods = years * compounds_per_year
    
    # Compound interest formula: A = P(1 + r/n)^(nt)
    final_amount = principal * (1 + rate_per_period) ** total_periods
    interest_earned = final_amount - principal
    
    # Calculate monthly contributions needed for goals
    monthly_rate = annual_rate / 12
    
    return {
        "principal": round(principal, 2),
        "annual_rate": f"{annual_rate * 100:.2f}%",
        "years": years,
        "compounding_frequency": compounds_per_year,
        "final_amount": round(final_amount, 2),
        "interest_earned": round(interest_earned, 2),
        "total_return": f"{(final_amount / principal - 1) * 100:.2f}%",
        "monthly_contribution_for_10k": round(
            10000 * monthly_rate / ((1 + monthly_rate) ** (years * 12) - 1), 2
        ) if annual_rate > 0 else 0
    }


def _count_business_days(start: datetime, end: datetime) -> int:
    """Count business days (Monday-Friday) between two dates."""
    business_days = 0
    current = start
    
    while current <= end:
        if current.weekday() < 5:  # Monday=0, Friday=4
            business_days += 1
        current += timedelta(days=1)
    
    return business_days


# Available tools for export
TIME_TOOLS = [
    get_current_time,
    get_date_difference,
    add_to_date,
    get_day_info,
    calculate_age,
    get_financial_dates,
    calculate_compounding,
]