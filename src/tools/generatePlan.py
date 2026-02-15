"""
Financial Plan PDF Generator - Creates professional PDF financial plans
"""

from typing import Dict, Any, Optional
from langchain.tools import tool
from datetime import datetime
import os
import json
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import textwrap

def ensure_plans_directory():
    """Ensure the plans directory exists."""
    plans_dir = os.path.join("financial_plans")
    if not os.path.exists(plans_dir):
        os.makedirs(plans_dir)
    return plans_dir

def format_currency(amount):
    """Format amount as currency."""
    try:
        amount_float = float(amount)
        if amount_float >= 0:
            return f"${amount_float:,.2f}"
        else:
            return f"-${abs(amount_float):,.2f}"
    except (ValueError, TypeError):
        return str(amount)

def format_percentage(value):
    """Format as percentage."""
    try:
        return f"{float(value):.1f}%"
    except:
        return str(value)

def create_financial_plan_pdf(
    user_id: str,
    profile_data: Dict[str, Any],
    short_term_goals: str,
    long_term_goals: str,
    plan_name: str = "Financial Plan",
    include_projections: bool = True
) -> Dict[str, Any]:
    """
    Create a professional PDF financial plan.
    
    Args:
        user_id: User ID
        profile_data: User profile data
        short_term_goals: Short-term goals (1-2 years)
        long_term_goals: Long-term goals (5+ years)
        plan_name: Name of the plan
        include_projections: Whether to include future projections
    
    Returns:
        Dict with success status and file path
    """
    try:
        # Ensure directory exists
        plans_dir = ensure_plans_directory()
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "._-")
        filename = f"{safe_user_id}_{plan_name.replace(' ', '_')}_{timestamp}.pdf"
        filepath = os.path.join(plans_dir, filename)
        
        # Create document - professional layout
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch,
            title=f"CoFina Financial Plan - {user_id}",
            author="CoFina AI Assistant"
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom professional styles
        title_style = ParagraphStyle(
            'ProfessionalTitle',
            parent=styles['Title'],
            fontSize=28,
            spaceAfter=20,
            textColor=colors.HexColor('#1a4d8c'),  # Deep blue
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'ProfessionalHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#2c3e50'),  # Dark slate
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=(0, 0, 4, 0)
        )
        
        subheading_style = ParagraphStyle(
            'ProfessionalSubHeading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor('#34495e'),  # Gray-blue
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'ProfessionalNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leading=14,
            textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica'
        )
        
        small_style = ParagraphStyle(
            'ProfessionalSmall',
            parent=styles['Normal'],
            fontSize=8,
            spaceAfter=4,
            leading=10,
            textColor=colors.HexColor('#7f8c8d'),
            fontName='Helvetica-Oblique'
        )
        
        # Create story
        story = []
        
        # Header with logo (text-based since we don't have image)
        story.append(Paragraph("CoFina", title_style))
        story.append(Paragraph("Intelligent Financial Planning", 
                              ParagraphStyle('Subtitle', parent=styles['Normal'], 
                                           fontSize=14, alignment=TA_CENTER,
                                           textColor=colors.HexColor('#7f8c8d'))))
        story.append(Spacer(1, 0.3*inch))
        
        # Plan info bar
        info_text = f"Plan: {plan_name} | Generated: {datetime.now().strftime('%B %d, %Y')} | User: {user_id}"
        story.append(Paragraph(info_text, 
                              ParagraphStyle('InfoBar', parent=styles['Normal'],
                                           fontSize=9, alignment=TA_CENTER,
                                           textColor=colors.HexColor('#ffffff'),
                                           backColor=colors.HexColor('#3498db'),
                                           borderPadding=(6, 10, 6, 10))))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Paragraph(
            "This personalized financial plan is designed to help you achieve your financial goals "
            "while maintaining a healthy balance between spending, saving, and investing. The recommendations "
            "are based on your financial profile and current market best practices.",
            normal_style
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Parse profile data
        if isinstance(profile_data, str):
            try:
                profile_data = json.loads(profile_data)
            except:
                profile_data = {}
        
        preferences = profile_data.get('preferences', {}) if isinstance(profile_data, dict) else {}
        income = profile_data.get('income') if isinstance(profile_data, dict) else None
        
        # Financial Profile Section
        story.append(Paragraph("Financial Profile", heading_style))
        
        profile_data_rows = []
        if income:
            profile_data_rows.append(["Annual Income", format_currency(income)])
        
        if preferences:
            risk_map = {"low": "Conservative", "moderate": "Moderate", "high": "Aggressive"}
            risk = preferences.get('risk_profile', '').lower()
            profile_data_rows.append(["Risk Tolerance", risk_map.get(risk, preferences.get('risk_profile', 'Not set'))])
            profile_data_rows.append(["Debt Strategy", preferences.get('debt_strategy', 'Not set')])
            profile_data_rows.append(["Savings Priority", preferences.get('savings_priority', 'Not set')])
        
        if profile_data_rows:
            profile_table = Table(profile_data_rows, colWidths=[2*inch, 3.5*inch])
            profile_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
            ]))
            story.append(profile_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Goals Section
        story.append(Paragraph("Financial Goals", heading_style))
        
        # Parse goals
        try:
            short_goals_dict = json.loads(short_term_goals) if isinstance(short_term_goals, str) else short_term_goals
            long_goals_dict = json.loads(long_term_goals) if isinstance(long_term_goals, str) else long_term_goals
        except:
            short_goals_dict = {"description": short_term_goals}
            long_goals_dict = {"description": long_term_goals}
        
        # Short-term goals
        story.append(Paragraph("Short-term Goals (1-2 years)", subheading_style))
        if isinstance(short_goals_dict, dict):
            for key, value in short_goals_dict.items():
                if key != "description" or not isinstance(short_goals_dict, dict) or len(short_goals_dict) == 1:
                    story.append(Paragraph(f"• {value if key == 'description' else key.replace('_', ' ').title()}: {value}", normal_style))
        else:
            story.append(Paragraph(f"• {short_goals_dict}", normal_style))
        
        story.append(Spacer(1, 0.1*inch))
        
        # Long-term goals
        story.append(Paragraph("Long-term Goals (5+ years)", subheading_style))
        if isinstance(long_goals_dict, dict):
            for key, value in long_goals_dict.items():
                if key != "description" or not isinstance(long_goals_dict, dict) or len(long_goals_dict) == 1:
                    story.append(Paragraph(f"• {value if key == 'description' else key.replace('_', ' ').title()}: {value}", normal_style))
        else:
            story.append(Paragraph(f"• {long_goals_dict}", normal_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Budget Allocation
        story.append(Paragraph("Recommended Budget Allocation", heading_style))
        
        if income:
            # 50/30/20 rule
            monthly_income = income / 12 if income else 0
            needs = monthly_income * 0.5
            wants = monthly_income * 0.3
            savings = monthly_income * 0.2
            
            budget_data = [
                ["Category", "Percentage", "Monthly Amount", "Annual Amount"],
                ["Needs (Essentials)", "50%", format_currency(needs), format_currency(needs * 12)],
                ["Wants (Lifestyle)", "30%", format_currency(wants), format_currency(wants * 12)],
                ["Savings & Debt", "20%", format_currency(savings), format_currency(savings * 12)],
                ["Total", "100%", format_currency(monthly_income), format_currency(income)]
            ]
            
            budget_table = Table(budget_data, colWidths=[1.5*inch, 0.8*inch, 1.2*inch, 1.5*inch])
            budget_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('GRID', (0, 1), (-1, -1), 0.5, colors.lightgrey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#ecf0f1')),
                ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold')
            ]))
            story.append(budget_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Action Plan
        story.append(Paragraph("Recommended Action Plan", heading_style))
        
        action_items = [
            "1. Build Emergency Fund: Save 3-6 months of essential expenses",
            "2. Automate Savings: Set up automatic transfers to savings and investment accounts",
            "3. Review Budget Monthly: Track spending and adjust categories as needed",
            "4. Pay Down High-Interest Debt: Focus on debts with interest rates above 6-8%",
            "5. Maximize Retirement Contributions: Take advantage of employer matches",
            "6. Review Insurance Coverage: Ensure adequate health, life, and disability coverage",
            "7. Monitor Progress: Track goals quarterly and adjust plan annually"
        ]
        
        for item in action_items:
            story.append(Paragraph(item, normal_style))
            story.append(Spacer(1, 3))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Projections (if requested)
        if include_projections and income:
            story.append(Paragraph("Future Projections", heading_style))
            
            # Simple projections
            years = [1, 3, 5, 10]
            projection_data = [["Time Horizon", "Estimated Savings", "Monthly Investment Needed"]]
            
            for y in years:
                # Assume 20% savings rate with 5% annual return
                annual_savings = income * 0.2
                future_value = annual_savings * ((1.05 ** y - 1) / 0.05)  # FV of annuity
                monthly_needed = (future_value / (12 * y)) if y > 0 else 0
                
                projection_data.append([
                    f"{y} Year{'s' if y > 1 else ''}",
                    format_currency(future_value),
                    format_currency(monthly_needed)
                ])
            
            proj_table = Table(projection_data, colWidths=[1.2*inch, 1.5*inch, 1.8*inch])
            proj_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))
            story.append(proj_table)
            story.append(Spacer(1, 0.2*inch))
            
            story.append(Paragraph(
                "* Projections assume 20% savings rate and 5% annual return. Actual results may vary.",
                small_style
            ))
            story.append(Spacer(1, 0.1*inch))
        
        # Disclaimer
        story.append(Paragraph("Important Disclosures", heading_style))
        disclaimer_text = (
            "This financial plan is for informational and educational purposes only and should not be considered "
            "as financial, investment, or tax advice. CoFina is an AI assistant and not a certified financial planner. "
            "You should consult with qualified financial professionals before making any financial decisions. "
            "Past performance does not guarantee future results. All projections are estimates and may differ from actual outcomes."
        )
        story.append(Paragraph(disclaimer_text, small_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Footer with generation info
        story.append(Spacer(1, 0.3*inch))
        footer_text = f"Generated by CoFina AI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Plan ID: {timestamp}"
        story.append(Paragraph(footer_text, 
                              ParagraphStyle('Footer', parent=styles['Normal'],
                                           fontSize=7, alignment=TA_CENTER,
                                           textColor=colors.HexColor('#95a5a6'))))
        
        # Build PDF
        doc.build(story)
        
        return {
            "success": True,
            "filepath": filepath,
            "filename": filename,
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "message": f"Financial plan generated: {filename}"
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
    plan_name: str = "Financial Plan",
    include_projections: bool = True
) -> Dict[str, Any]:
    """
    Generate a professional PDF financial plan.
    
    Args:
        user_id: User ID
        profile_data: JSON string of user profile
        short_term_goals: Short-term goals (1-2 years)
        long_term_goals: Long-term goals (5+ years)
        plan_name: Name of the plan
        include_projections: Whether to include projections
    
    Returns:
        Dict with success status and file info
    """
    return create_financial_plan_pdf(
        user_id=user_id,
        profile_data=profile_data,
        short_term_goals=short_term_goals,
        long_term_goals=long_term_goals,
        plan_name=plan_name,
        include_projections=include_projections
    )