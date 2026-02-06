# src/tools/generatePlan.py

from typing import Dict, Any
from langchain.tools import tool
from datetime import datetime
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import json


def ensure_plans_directory():
    """Ensure the users_plans directory exists."""
    plans_dir = os.path.join("allPlans")
    if not os.path.exists(plans_dir):
        os.makedirs(plans_dir)
    return plans_dir


def format_currency(amount):
    """Format amount as currency."""
    try:
        return f"${float(amount):,.2f}"
    except (ValueError, TypeError):
        return str(amount)


def format_goals(goals_data):
    """Format goals data into readable text."""
    if isinstance(goals_data, str):
        # Try to parse as JSON, but if it fails, use as-is
        try:
            goals_data = json.loads(goals_data)
        except json.JSONDecodeError:
            # If it's not JSON, just return the string
            return goals_data
    
    if isinstance(goals_data, dict):
        formatted = []
        for goal, value in goals_data.items():
            goal_name = goal.replace('_', ' ').title()
            if isinstance(value, (int, float)):
                formatted.append(f"• {goal_name}: {format_currency(value)}")
            else:
                formatted.append(f"• {goal_name}: {value}")
        return "\n".join(formatted)
    
    # If it's already a string, just return it
    return str(goals_data)


def create_financial_plan_pdf(user_id: str, profile_data: Dict[str, Any], 
                            short_term_goals: str, long_term_goals: str,
                            plan_name: str = "Financial Plan") -> Dict[str, Any]:
    """
    Create a PDF financial plan for a user.
    
    Args:
        user_id: User ID
        profile_data: User profile data (dict or JSON string)
        short_term_goals: Short-term financial goals
        long_term_goals: Long-term financial goals
        plan_name: Name of the financial plan
    
    Returns:
        Dict with success status and file path
    """
    try:
        # Ensure directory exists
        plans_dir = ensure_plans_directory()
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_id}_financial_plan_{timestamp}.pdf"
        filepath = os.path.join(plans_dir, filename)
        
        # Create document
        doc = SimpleDocTemplate(filepath, pagesize=A4, 
                              topMargin=0.5*inch, bottomMargin=0.5*inch,
                              leftMargin=0.75*inch, rightMargin=0.75*inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2980b9')
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#3498db')
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leading=14
        )
        
        # Create story (content elements)
        story = []
        
        # Header
        story.append(Paragraph("CoFina Financial Plan", title_style))
        story.append(Spacer(1, 10))
        
        # Creation date
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                             styles['Normal']))
        story.append(Spacer(1, 20))
        
        # User Information Section
        story.append(Paragraph("Personal Information", heading_style))
        
        user_info_data = [
            ["User ID:", user_id],
            ["Plan Name:", plan_name],
            ["Generated On:", datetime.now().strftime("%Y-%m-%d")]
        ]
        
        user_info_table = Table(user_info_data, colWidths=[1.5*inch, 4*inch])
        user_info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(user_info_table)
        story.append(Spacer(1, 25))
        
        # Financial Profile Section
        story.append(Paragraph("Financial Profile", heading_style))
        
        # Parse profile data if it's a string
        if isinstance(profile_data, str):
            try:
                profile_data = json.loads(profile_data)
            except:
                profile_data = {}
        
        if profile_data and isinstance(profile_data, dict):
            pref = profile_data.get('preferences', {})
            profile_table_data = [
                ["Risk Profile:", pref.get('risk_profile', 'Not specified')],
                ["Debt Strategy:", pref.get('debt_strategy', 'Not specified')],
                ["Savings Priority:", pref.get('savings_priority', 'Not specified')]
            ]
            
            if profile_data.get('income'):
                profile_table_data.append(["Monthly Income:", format_currency(profile_data.get('income'))])
            
            profile_table = Table(profile_table_data, colWidths=[1.5*inch, 4*inch])
            profile_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4fc')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
            ]))
            
            story.append(profile_table)
        
        story.append(Spacer(1, 25))
        
        # Goals Section
        story.append(Paragraph("Financial Goals", heading_style))
        
        # Short-term Goals
        story.append(Paragraph("Short-term Goals (1-2 years)", subheading_style))
        short_term_text = format_goals(short_term_goals)
        story.append(Paragraph(short_term_text, normal_style))
        story.append(Spacer(1, 15))
        
        # Long-term Goals
        story.append(Paragraph("Long-term Goals (5+ years)", subheading_style))
        long_term_text = format_goals(long_term_goals)
        story.append(Paragraph(long_term_text, normal_style))
        story.append(Spacer(1, 25))
        
        # Action Plan Section
        story.append(Paragraph("Recommended Action Plan", heading_style))
        
        action_items = [
            "1. Review your budget monthly and adjust as needed",
            "2. Automate savings for your emergency fund and goals",
            "3. Regularly monitor your investment portfolio",
            "4. Review and update your financial plan annually",
            "5. Consult with a financial advisor for major decisions"
        ]
        
        for item in action_items:
            story.append(Paragraph(item, normal_style))
            story.append(Spacer(1, 5))
        
        story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(Paragraph("Disclaimer", subheading_style))
        disclaimer = Paragraph(
            "This financial plan is for informational purposes only and should not be considered "
            "as financial advice. Please consult with a qualified financial advisor before making "
            "any investment decisions. Past performance is not indicative of future results.",
            styles['Italic']
        )
        story.append(disclaimer)
        
        # Build PDF
        doc.build(story)
        
        return {
            "success": True,
            "filepath": filepath,
            "filename": filename,
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "message": f"Financial plan PDF generated successfully: {filename}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id,
            "message": f"Failed to generate PDF: {str(e)}"
        }


@tool
def generate_financial_plan_pdf(
    user_id: str,
    profile_data: str,
    short_term_goals: str,
    long_term_goals: str,
    plan_name: str = "Financial Plan"
) -> Dict[str, Any]:
    """
    Generate a PDF financial plan for a user.
    
    Args:
        user_id: User ID
        profile_data: JSON string of user profile data
        short_term_goals: Short-term financial goals (1-2 years)
        long_term_goals: Long-term financial goals (5+ years)
        plan_name: Name of the financial plan
    
    Returns:
        Dict with success status and file information
    """
    return create_financial_plan_pdf(
        user_id=user_id,
        profile_data=profile_data,
        short_term_goals=short_term_goals,
        long_term_goals=long_term_goals,
        plan_name=plan_name
    )