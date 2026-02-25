"""
Market Agent - Searches for products and compares options
Enhanced with comprehensive fallback categories for young professionals
"""

from typing import Dict, Any, List, Optional, Union
from tools.searchProducts import search_products
import re
from datetime import datetime
import json


class MarketAgent:
    """
    Specialized agent for product research and comparison
    Enhanced with fallback knowledge base for young professional needs
    """

    def __init__(self):
        self.search_history = {}
        
        # Comprehensive knowledge base for young professionals
        self._initialize_fallback_knowledge()
    
    def _initialize_fallback_knowledge(self):
        """Initialize fallback knowledge for common young professional needs"""
        
        # Technology & Electronics (highest spend category)
        self.tech_fallback = {
            "laptop": {
                "professional": [
                    {"item": "MacBook Pro 14\" (M3)", "price_range": "$1,800-2,300", 
                     "suggestion": "Ideal for creative professionals and developers",
                     "alternatives": ["Dell XPS 15", "Lenovo ThinkPad X1 Carbon", "Microsoft Surface Laptop 5"],
                     "financing": "Apple Card 12-month financing available"},
                    {"item": "Dell XPS 13 Plus", "price_range": "$1,200-1,600",
                     "suggestion": "Excellent Windows ultrabook for business",
                     "alternatives": ["HP Spectre x360", "Asus ZenBook 14"],
                     "financing": "Dell Financing with 0% APR offers"}
                ],
                "budget": [
                    {"item": "MacBook Air M1", "price_range": "$900-1,100",
                     "suggestion": "Best value for students and professionals",
                     "alternatives": ["Acer Swift 3", "Lenovo IdeaPad 5"],
                     "financing": "Apple refurbished saves 15-20%"},
                    {"item": "Chromebook Plus", "price_range": "$400-600",
                     "suggestion": "Great for cloud-based work and students"}
                ]
            },
            "smartphone": {
                "flagship": [
                    {"item": "iPhone 15 Pro", "price_range": "$1,000-1,200",
                     "suggestion": "Best ecosystem integration",
                     "alternatives": ["Samsung S24 Ultra", "Google Pixel 8 Pro"],
                     "carrier_deals": "Trade-in offers up to $800"},
                    {"item": "Samsung Galaxy S24+", "price_range": "$900-1,100",
                     "suggestion": "Best Android experience with AI features"}
                ],
                "mid_range": [
                    {"item": "iPhone 14/15", "price_range": "$700-900",
                     "suggestion": "Solid performance at lower price",
                     "alternatives": ["Google Pixel 7a", "Nothing Phone 2"]},
                    {"item": "Samsung A54", "price_range": "$350-450",
                     "suggestion": "Best budget Android option"}
                ]
            },
            "tablet": [
                {"item": "iPad Air", "price_range": "$600-800",
                 "suggestion": "Perfect for note-taking and media",
                 "use_case": "students, professionals"},
                {"item": "Samsung Tab S9", "price_range": "$700-900",
                 "suggestion": "Great Android alternative with Dex mode"},
                {"item": "iPad Pro", "price_range": "$1,000-1,500",
                 "suggestion": "Laptop replacement for creatives"}
            ],
            "headphones": {
                "premium": [
                    {"item": "Sony WH-1000XM5", "price_range": "$350-400",
                     "suggestion": "Best noise cancellation for work"},
                    {"item": "Apple AirPods Max", "price_range": "$500-550",
                     "suggestion": "Premium Apple ecosystem choice"}
                ],
                "mid_range": [
                    {"item": "Bose QuietComfort 45", "price_range": "$250-300",
                     "suggestion": "Excellent comfort for long wear"},
                    {"item": "Sennheiser Momentum 4", "price_range": "$280-350"}
                ],
                "budget": [
                    {"item": "Anker Soundcore Q45", "price_range": "$100-130",
                     "suggestion": "Best value noise cancellation"}
                ]
            },
            "monitors": {
                "productivity": [
                    {"item": "LG 27\" 4K UHD", "price_range": "$350-450",
                     "suggestion": "Perfect for coding and office work"},
                    {"item": "Dell UltraSharp 27\"", "price_range": "$500-700"}
                ],
                "ultrawide": [
                    {"item": "34\" Curved Ultrawide", "price_range": "$400-600",
                     "suggestion": "Great for multitasking and trading"}
                ]
            },
            "accessories": [
                {"item": "Mechanical Keyboard", "price_range": "$80-150",
                 "suggestion": "Improve typing comfort and productivity"},
                {"item": "Ergonomic Mouse", "price_range": "$50-100",
                 "suggestion": "Prevent wrist strain during long work hours"},
                {"item": "Docking Station", "price_range": "$100-200",
                 "suggestion": "Simplify work-from-home setup"}
            ]
        }

        # Career Development & Education
        self.career_fallback = {
            "courses": {
                "tech": [
                    {"item": "Coding Bootcamp", "price_range": "$10,000-20,000",
                     "suggestion": "Career switch to tech in 3-6 months",
                     "platforms": ["General Assembly", "Flatiron School", "Hack Reactor"],
                     "financing": "Income Share Agreements available"},
                    {"item": "AWS Certification", "price_range": "$300-500",
                     "suggestion": "Boost cloud computing career",
                     "platform": "AWS Training + exam"},
                    {"item": "Data Science Certificate", "price_range": "$1,000-3,000",
                     "suggestion": "Coursera/edX professional certificates"}
                ],
                "business": [
                    {"item": "MBA Preparatory Course", "price_range": "$500-1,500",
                     "suggestion": "GMAT prep or business fundamentals"},
                    {"item": "Project Management Certification", "price_range": "$600-1,000",
                     "suggestion": "PMP or CAPM certification"}
                ],
                "creative": [
                    {"item": "UX/UI Design Course", "price_range": "$500-2,000"},
                    {"item": "Digital Marketing Certificate", "price_range": "$400-800"}
                ]
            },
            "books": {
                "professional": [
                    {"item": "Atomic Habits", "price": "$16-25",
                     "suggestion": "Build better professional habits"},
                    {"item": "The 7 Habits of Highly Effective People", "price": "$15-22"},
                    {"item": "Never Split the Difference", "price": "$18-26",
                     "suggestion": "Negotiation skills for career growth"}
                ],
                "financial": [
                    {"item": "The Psychology of Money", "price": "$15-22"},
                    {"item": "Rich Dad Poor Dad", "price": "$12-18"},
                    {"item": "The Simple Path to Wealth", "price": "$14-20"}
                ]
            },
            "professional_services": [
                {"item": "Resume Writing Service", "price_range": "$150-300",
                 "suggestion": "Professional resume makeover"},
                {"item": "LinkedIn Profile Optimization", "price_range": "$100-250"},
                {"item": "Career Coaching Session", "price_range": "$100-200/session"},
                {"item": "Headshot Photography", "price_range": "$150-400",
                 "suggestion": "Professional photos for LinkedIn"}
            ]
        }

        # Fitness & Wellness
        self.fitness_fallback = {
            "gym_memberships": {
                "budget": [
                    {"item": "Planet Fitness", "price_range": "$10-25/month",
                     "perks": "Basic equipment, 24/7 locations"},
                    {"item": "Crunch Fitness", "price_range": "$10-30/month"}
                ],
                "mid_range": [
                    {"item": "LA Fitness", "price_range": "$30-50/month",
                     "perks": "Pool, basketball, classes"},
                    {"item": "Equinox", "price_range": "$180-250/month",
                     "suggestion": "Premium experience with luxury amenities"}
                ],
                "boutique": [
                    {"item": "ClassPass Credits", "price_range": "$50-150/month",
                     "suggestion": "Try different studios each week"},
                    {"item": "SoulCycle", "price_range": "$30-35/class"},
                    {"item": "YogaWorks", "price_range": "$150-200/month"}
                ]
            },
            "home_equipment": [
                {"item": "Peloton Bike", "price_range": "$1,500-2,500",
                 "suggestion": "Popular indoor cycling",
                 "alternatives": ["Echelon ($800-1,500)", "NordicTrack ($1,000-2,000)"]},
                {"item": "Treadmill", "price_range": "$800-2,500"},
                {"item": "Adjustable Dumbbells", "price_range": "$300-600"},
                {"item": "Yoga Mat", "price_range": "$30-80"},
                {"item": "Resistance Bands Set", "price_range": "$25-50"}
            ],
            "wellness_apps": [
                {"item": "Headspace", "price": "$70/year",
                 "suggestion": "Meditation and mental wellness"},
                {"item": "Calm", "price": "$70/year"},
                {"item": "MyFitnessPal Premium", "price": "$50/year"},
                {"item": "Strava Summit", "price": "$60/year"}
            ],
            "meal_prep": [
                {"item": "HelloFresh", "price": "$60-80/week",
                 "suggestion": "Meal kits for busy professionals"},
                {"item": "Factor Meals", "price": "$80-100/week",
                 "suggestion": "Prepared healthy meals"},
                {"item": "Freshly", "price": "$70-90/week"}
            ]
        }

        # Lifestyle & Experiences
        self.lifestyle_fallback = {
            "travel": {
                "weekend_trips": [
                    {"item": "Domestic Flight", "price_range": "$150-400"},
                    {"item": "Hotel Stay (3-4 star)", "price_range": "$150-300/night"},
                    {"item": "Airbnb Entire Place", "price_range": "$100-250/night"}
                ],
                "international": [
                    {"item": "Europe Flight", "price_range": "$600-1,200"},
                    {"item": "Asia Flight", "price_range": "$800-1,500"},
                    {"item": "All-inclusive Resort", "price_range": "$1,500-3,000/week"}
                ],
                "travel_hacks": [
                    {"item": "Travel Credit Card", "benefits": "Points and lounge access",
                     "popular": ["Chase Sapphire Preferred", "Capital One Venture"]},
                    {"item": "Hostel Stay", "price_range": "$30-60/night"},
                    {"item": "Travel Insurance", "price_range": "$50-150/trip"}
                ]
            },
            "dining": {
                "casual": [
                    {"item": "Coffee Shop Daily", "price_range": "$4-7/day"},
                    {"item": "Lunch Delivery", "price_range": "$15-25/meal"},
                    {"item": "Fast Casual Dinner", "price_range": "$20-35/meal"}
                ],
                "fine_dining": [
                    {"item": "Date Night Restaurant", "price_range": "$80-150/couple"},
                    {"item": "Michelin Star Experience", "price_range": "$150-300+/person"}
                ],
                "meal_subscriptions": [
                    {"item": "Grocery Delivery", "price_range": "$10-15/month fee"},
                    {"item": "Wine Club", "price_range": "$50-150/month"}
                ]
            },
            "entertainment": {
                "streaming": [
                    {"item": "Netflix", "price_range": "$7-20/month"},
                    {"item": "Spotify Premium", "price_range": "$10-15/month"},
                    {"item": "YouTube Premium", "price_range": "$12-18/month"},
                    {"item": "Disney+ Bundle", "price_range": "$13-20/month"}
                ],
                "events": [
                    {"item": "Concert Ticket", "price_range": "$50-200"},
                    {"item": "Festival Pass", "price_range": "$300-600"},
                    {"item": "Broadway/Show Ticket", "price_range": "$80-250"},
                    {"item": "Sporting Event", "price_range": "$40-150"}
                ],
                "hobbies": [
                    {"item": "Gaming Console", "price_range": "$300-500"},
                    {"item": "Video Games", "price_range": "$40-70 each"},
                    {"item": "Board Games", "price_range": "$30-60"},
                    {"item": "Photography Gear", "price_range": "$500-2,000+"}
                ]
            },
            "dating": [
                {"item": "Dating App Premium", "price_range": "$20-40/month"},
                {"item": "Coffee Date", "price_range": "$15-25"},
                {"item": "Dinner Date", "price_range": "$60-120"},
                {"item": "Activity Date", "price_range": "$40-100"}
            ]
        }

        # Housing & Living
        self.housing_fallback = {
            "apartment": {
                "studio": {"price_range": "$1,200-2,500/month", "cities": "varies by location"},
                "1_bedroom": {"price_range": "$1,500-3,000/month"},
                "2_bedroom": {"price_range": "$2,000-4,000/month"},
                "utilities": {"price_range": "$150-300/month", "includes": "electricity, water, internet"}
            },
            "furniture": {
                "essential": [
                    {"item": "Bed Frame + Mattress", "price_range": "$500-1,500"},
                    {"item": "Sofa", "price_range": "$400-1,200"},
                    {"item": "Dining Table + Chairs", "price_range": "$300-800"},
                    {"item": "Desk + Chair", "price_range": "$200-600"}
                ],
                "budget_alternatives": [
                    {"item": "IKEA Basics", "price_range": "$100-400/item"},
                    {"item": "Facebook Marketplace", "saving": "50-70% off retail"},
                    {"item": "Wayfair Sales", "price_range": "20-40% off"}
                ]
            },
            "roommate_resources": [
                {"item": "Roommate Finder Service", "price_range": "$30-50"},
                {"item": "Room Rental Agreement Template", "price_range": "$20-40"},
                {"item": "Roommate Matching App", "price_range": "$0-15"}
            ]
        }

        # Financial Products
        self.financial_fallback = {
            "banking": [
                {"item": "High-Yield Savings Account", "rate": "4-5% APY",
                 "popular": ["Ally", "Marcus", "Discover"]},
                {"item": "Checking Account Bonus", "value": "$200-500",
                 "requirement": "Direct deposit setup"},
                {"item": "Certificate of Deposit (CD)", "rate": "4-5% APY",
                 "term": "3-24 months"}
            ],
            "credit_cards": {
                "cash_back": [
                    {"item": "Citi Double Cash", "benefit": "2% on everything"},
                    {"item": "Chase Freedom Unlimited", "benefit": "1.5% + rotating categories"}
                ],
                "travel": [
                    {"item": "Chase Sapphire Preferred", "annual_fee": "$95",
                     "bonus": "60,000 points"},
                    {"item": "Capital One Venture X", "annual_fee": "$395",
                     "benefit": "$300 travel credit"}
                ],
                "student_builder": [
                    {"item": "Discover it Student", "benefit": "5% categories + match"},
                    {"item": "Capital One SavorOne", "benefit": "3% dining"}
                ]
            },
            "investment": [
                {"item": "Roth IRA", "contribution_limit": "$7,000/year",
                 "platforms": ["Vanguard", "Fidelity", "Schwab"]},
                {"item": "Index Fund", "min_investment": "$1-3,000",
                 "popular": ["VTSAX", "FXAIX", "SWPPX"]},
                {"item": "Fractional Shares", "min_investment": "$1-5",
                 "apps": ["Robinhood", "M1 Finance", "Fidelity"]}
            ],
            "insurance": [
                {"item": "Renters Insurance", "price_range": "$10-20/month"},
                {"item": "Life Insurance (Term)", "price_range": "$25-50/month"},
                {"item": "Disability Insurance", "price_range": "$30-80/month"}
            ]
        }

        # Transportation
        self.transport_fallback = {
            "car_buying": {
                "new": {"price_range": "$25,000-40,000", "loan_term": "48-72 months"},
                "used": {"price_range": "$15,000-25,000", "age": "3-5 years"},
                "lease": {"price_range": "$250-450/month", "term": "24-36 months"}
            },
            "car_maintenance": [
                {"item": "Oil Change", "price_range": "$40-80"},
                {"item": "Tire Rotation", "price_range": "$20-50"},
                {"item": "Car Insurance", "price_range": "$100-200/month"},
                {"item": "Parking Pass", "price_range": "$50-200/month"}
            ],
            "alternatives": [
                {"item": "Public Transit Pass", "price_range": "$50-120/month"},
                {"item": "Bike/Scooter", "price_range": "$300-1,500 one-time"},
                {"item": "Rideshare Budget", "price_range": "$100-300/month"},
                {"item": "Car Sharing", "price_range": "$10-15/hour"}
            ]
        }

    def process(self, query: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process product search request with enhanced fallback system
        """
        # Extract product information using improved extraction
        product_info = self._extract_product_info(query, context)
        
        # Check if we need more information
        if not product_info.get("product"):
            return {
                "action": "clarify",
                "message": self._generate_clarification_message(query),
                "data": {
                    "suggested_categories": self._suggest_categories(query),
                    "common_needs": self._get_common_needs()
                }
            }
        
        # Check user's financial capability
        financial_check = self._check_affordability(product_info, context)
        
        # Try real search first
        search_results = self._search_products(product_info)
        
        # If search fails or returns no results, use enhanced fallback
        if search_results.get("status") == "error" or not search_results.get("products"):
            fallback_recommendations = self._get_enhanced_fallback(
                product_info, 
                financial_check,
                context.get("user_profile", {})
            )
            
            return self._format_fallback_response(
                product_info, 
                fallback_recommendations, 
                financial_check,
                query
            )
        
        # Analyze and rank options
        recommendations = self._analyze_options(search_results, financial_check)
        
        return {
            "action": "recommendations",
            "message": f"🔍 I found some options for {product_info['product']}:",
            "data": {
                "product": product_info["product"],
                "options": recommendations["options"],
                "best_value": recommendations["best_value"],
                "affordability": financial_check,
                "budget_tip": self._get_budget_tip(product_info, financial_check),
                "alternatives": self._get_quick_alternatives(product_info),
                "search_query": product_info
            }
        }
    
    def _extract_product_info(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Enhanced extraction with context awareness"""
        info = {
            "product": None,
            "category": None,
            "max_price": None,
            "preferences": [],
            "intent": "research",
            "urgency": "normal"
        }
        
        query_lower = query.lower()
        
        # Check for common product categories
        category_keywords = {
            "tech": ["laptop", "computer", "phone", "smartphone", "tablet", "ipad", 
                     "headphones", "monitor", "keyboard", "accessories", "gadget"],
            "career": ["course", "certification", "book", "career", "resume", "coaching"],
            "fitness": ["gym", "fitness", "workout", "yoga", "exercise", "peloton", "equipment"],
            "lifestyle": ["travel", "vacation", "flight", "hotel", "concert", "event", 
                          "restaurant", "dining", "streaming", "subscription"],
            "housing": ["apartment", "furniture", "rent", "roommate", "decor"],
            "financial": ["credit card", "bank", "savings", "investment", "insurance"],
            "transport": ["car", "vehicle", "bike", "scooter", "transit"]
        }
        
        # Detect category
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                info["category"] = category
                # Extract specific product
                for keyword in keywords:
                    if keyword in query_lower:
                        info["product"] = keyword
                        break
                break
        
        # If no product found, try to extract main noun
        if not info["product"]:
            # Simple noun extraction (in production use NLP)
            words = query_lower.split()
            for word in words:
                if len(word) > 3 and word not in ["what", "where", "when", "how", "much", "best"]:
                    info["product"] = word
                    break
        
        # Extract price constraints
        price_patterns = [
            r"(?:under|below|less than|max|budget of)\s*[$]?(\d+(?:,\d+)?(?:\.\d+)?)",
            r"[$](\d+(?:,\d+)?(?:\.\d+)?)\s*(?:or less|max|budget)",
            r"(?:spend|cost|price).*?(\d+(?:,\d+)?)"
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                try:
                    price_str = match.group(1).replace(',', '')
                    info["max_price"] = float(price_str)
                    break
                except:
                    pass
        
        # Detect intent
        if any(word in query_lower for word in ["buy", "purchase", "get", "acquire"]):
            info["intent"] = "purchase"
        elif any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            info["intent"] = "compare"
        elif any(word in query_lower for word in ["best", "top", "recommend", "suggestion"]):
            info["intent"] = "recommendation"
        
        # Detect urgency
        if any(word in query_lower for word in ["urgent", "asap", "today", "now", "quick"]):
            info["urgency"] = "high"
        elif any(word in query_lower for word in ["planning", "future", "eventually", "save"]):
            info["urgency"] = "low"
        
        return info
    
    def _generate_clarification_message(self, query: str) -> str:
        """Generate helpful clarification message"""
        return (
            "I'd be happy to help you find the right product! 🤔\n\n"
            "To give you the best recommendations, please let me know:\n"
            "• **What** are you looking for? (e.g., laptop, gym membership, investment account)\n"
            "• **Budget range** if any? (e.g., under $1,000)\n"
            "• **How soon** do you need it? (e.g., this week, planning ahead)\n"
            "• Any **specific preferences**? (brand, features, location)\n\n"
            "Or choose from common categories below:"
        )
    
    def _suggest_categories(self, query: str) -> List[str]:
        """Suggest relevant categories based on query"""
        query_lower = query.lower()
        
        suggestions = []
        if any(word in query_lower for word in ["tech", "computer", "phone", "laptop"]):
            suggestions = ["laptop", "smartphone", "tablet", "headphones"]
        elif any(word in query_lower for word in ["fit", "gym", "workout", "health"]):
            suggestions = ["gym membership", "home equipment", "wellness apps", "meal prep"]
        elif any(word in query_lower for word in ["career", "job", "work", "professional"]):
            suggestions = ["courses", "certifications", "career coaching", "resume service"]
        elif any(word in query_lower for word in ["travel", "vacation", "trip"]):
            suggestions = ["flights", "hotels", "travel insurance", "packing gear"]
        else:
            suggestions = ["tech & gadgets", "fitness & wellness", "career development", 
                          "travel & experiences", "housing & furniture", "financial products"]
        
        return suggestions
    
    def _get_common_needs(self) -> List[Dict]:
        """Get list of common young professional needs"""
        return [
            {"category": "💻 Tech", "items": ["MacBook Pro", "iPhone", "Noise-cancelling headphones", "Mechanical keyboard"]},
            {"category": "🏋️ Fitness", "items": ["Gym membership", "Peloton", "Meal prep service", "Yoga mat"]},
            {"category": "📚 Career", "items": ["Online course", "Resume service", "Professional books", "Networking events"]},
            {"category": "✈️ Travel", "items": ["Weekend getaway", "Travel credit card", "Luggage set", "Travel insurance"]},
            {"category": "🏠 Living", "items": ["Apartment furniture", "Renters insurance", "Roommate finder", "Utilities setup"]},
            {"category": "💰 Finance", "items": ["High-yield savings", "Roth IRA", "Credit card", "Investment app"]}
        ]
    
    def _get_enhanced_fallback(self, product_info: Dict, financial_check: Dict, 
                               user_profile: Dict) -> Dict[str, Any]:
        """Get enhanced fallback recommendations from knowledge base"""
        product = product_info.get("product", "").lower()
        category = product_info.get("category", "")
        
        recommendations = {
            "primary_options": [],
            "budget_options": [],
            "alternatives": [],
            "tips": [],
            "financing_options": []
        }
        
        # Match to appropriate fallback category
        if category == "tech" or any(word in product for word in ["laptop", "phone", "computer"]):
            if "laptop" in product:
                recommendations["primary_options"] = self.tech_fallback["laptop"]["professional"]
                recommendations["budget_options"] = self.tech_fallback["laptop"]["budget"]
                recommendations["tips"] = [
                    "💡 Consider refurbished models for 20-30% savings",
                    "💡 Student/educator discounts available",
                    "💡 Best times to buy: Back-to-school sales, Black Friday"
                ]
            elif "phone" in product or "smartphone" in product:
                recommendations["primary_options"] = self.tech_fallback["smartphone"]["flagship"]
                recommendations["budget_options"] = self.tech_fallback["smartphone"]["mid_range"]
                recommendations["tips"] = [
                    "💡 Buy last year's model for best value",
                    "💡 Trade-in your current phone for $200-500 credit",
                    "💡 Consider carrier deals around iPhone release (September)"
                ]
            elif "headphone" in product:
                recommendations["primary_options"] = self.tech_fallback["headphones"]["premium"]
                recommendations["budget_options"] = self.tech_fallback["headphones"]["budget"]
                
        elif category == "fitness" or any(word in product for word in ["gym", "fitness", "workout"]):
            recommendations["primary_options"] = self.fitness_fallback["gym_memberships"]["mid_range"]
            recommendations["budget_options"] = self.fitness_fallback["gym_memberships"]["budget"]
            recommendations["alternatives"] = self.fitness_fallback["home_equipment"][:3]
            
        elif category == "career" or any(word in product for word in ["course", "career", "certification"]):
            recommendations["primary_options"] = self.career_fallback["courses"]["tech"][:2]
            recommendations["budget_options"] = self.career_fallback["books"]["professional"][:2]
            recommendations["alternatives"] = self.career_fallback["professional_services"][:2]
            
        elif category == "lifestyle" or any(word in product for word in ["travel", "vacation", "trip"]):
            recommendations["primary_options"] = self.lifestyle_fallback["travel"]["weekend_trips"]
            recommendations["budget_options"] = self.lifestyle_fallback["travel"]["travel_hacks"]
            recommendations["tips"] = [
                "💡 Book flights 6-8 weeks in advance for best prices",
                "💡 Use incognito mode when searching for flights",
                "💡 Consider Tuesday/Wednesday travel for lowest fares"
            ]
            
        elif category == "financial" or any(word in product for word in ["credit", "bank", "invest"]):
            if "credit" in product:
                recommendations["primary_options"] = self.financial_fallback["credit_cards"]["cash_back"]
                recommendations["alternatives"] = self.financial_fallback["credit_cards"]["travel"]
            elif "invest" in product:
                recommendations["primary_options"] = self.financial_fallback["investment"][:2]
                recommendations["tips"] = [
                    "💡 Start with tax-advantaged accounts (Roth IRA/401k)",
                    "💡 Consider low-cost index funds for beginners",
                    "💡 Robo-advisors like Betterment have low minimums ($0-500)"
                ]
            else:
                recommendations["primary_options"] = self.financial_fallback["banking"][:2]
        
        # Add financial context tips
        if financial_check.get("affordable") == False:
            recommendations["tips"].append(
                f"💰 Based on your finances, consider: {financial_check['suggestions'][0] if financial_check.get('suggestions') else 'building an emergency fund first'}"
            )
        
        # Add financing options
        recommendations["financing_options"] = [
            "✅ 0% APR credit cards (if paid within promo period)",
            "✅ Affirm/Klarna/Afterpay for installment plans",
            "✅ Manufacturer financing (Apple Card, Dell Preferred)",
            "✅ Save in high-yield account for 3-6 months first"
        ]
        
        return recommendations
    
    def _check_affordability(self, product_info: Dict, context: Dict) -> Dict[str, Any]:
        """Enhanced affordability check with smart suggestions"""
        result = {
            "affordable": True,
            "impact": "minimal",
            "suggestions": [],
            "estimated_months_to_save": None,
            "percentage_of_income": None
        }
        
        # Get user's financial data
        user_data = context.get("user_profile", {})
        annual_income = user_data.get("income", 0)
        monthly_income = annual_income / 12 if annual_income else 0
        monthly_expenses = user_data.get("monthly_expenses", monthly_income * 0.7)  # Estimate if not provided
        disposable_income = max(0, monthly_income - monthly_expenses)
        
        # Get product price
        price = product_info.get("max_price")
        if not price:
            # Estimate price based on product category
            category = product_info.get("category")
            if category == "tech":
                price = 1200
            elif category == "fitness":
                price = 300
            elif category == "travel":
                price = 800
            elif category == "career":
                price = 500
            else:
                return result  # Can't determine affordability
        
        if monthly_income > 0:
            price_percentage = (price / monthly_income) * 100
            result["percentage_of_income"] = round(price_percentage, 1)
            
            # Determine affordability
            if price_percentage > 50:
                result["affordable"] = False
                result["impact"] = "high"
                
                # Calculate months to save
                if disposable_income > 0:
                    months = price / disposable_income
                    result["estimated_months_to_save"] = round(months, 1)
                    
                    result["suggestions"] = [
                        f"Save for {round(months, 1)} months at your current disposable income",
                        f"Consider putting aside ${round(price/6):,.0f}/month for 6 months",
                        "Look for refurbished or older models",
                        "Check if your employer offers purchase programs"
                    ]
                else:
                    result["suggestions"] = [
                        "Build an emergency fund first",
                        "Review monthly expenses to find savings opportunities",
                        "Consider if this aligns with your financial goals"
                    ]
                    
            elif price_percentage > 20:
                result["impact"] = "medium"
                result["suggestions"] = [
                    f"This is {price_percentage}% of your monthly income",
                    "Consider financing options or 0% APR credit cards",
                    "Check if you can find a better deal or cashback offers"
                ]
            else:
                result["suggestions"] = [
                    "This is within a reasonable budget",
                    "Check for cashback or rewards when purchasing",
                    "Consider if you need the item now or can wait for a sale"
                ]
        
        return result
    
    def _format_fallback_response(self, product_info: Dict, recommendations: Dict, 
                                  financial_check: Dict, original_query: str) -> Dict:
        """Format fallback response in a helpful, professional way"""
        
        product = product_info.get("product", "item").title()
        
        # Build response message
        message_parts = [
            f"🔍 **I found some information about {product} for you:**\n"
        ]
        
        # Primary options
        if recommendations.get("primary_options"):
            message_parts.append("\n**✨ Top Recommendations:**")
            for item in recommendations["primary_options"][:2]:
                if isinstance(item, dict):
                    name = item.get("item", item.get("name", "Option"))
                    price = item.get("price_range", item.get("price", "Price varies"))
                    suggestion = item.get("suggestion", "")
                    message_parts.append(f"• **{name}** — {price}")
                    if suggestion:
                        message_parts.append(f"  💡 {suggestion}")
        
        # Budget options
        if recommendations.get("budget_options"):
            message_parts.append("\n**💰 Budget-Friendly Options:**")
            for item in recommendations["budget_options"][:2]:
                if isinstance(item, dict):
                    name = item.get("item", "Option")
                    price = item.get("price_range", item.get("price", "Varies"))
                    message_parts.append(f"• {name} — {price}")
        
        # Tips
        if recommendations.get("tips"):
            message_parts.append("\n**💡 Pro Tips:**")
            for tip in recommendations["tips"][:3]:
                message_parts.append(f"• {tip}")
        
        # Financial context
        if not financial_check.get("affordable", True):
            message_parts.append("\n**📊 Financial Context:**")
            for suggestion in financial_check.get("suggestions", [])[:2]:
                message_parts.append(f"• {suggestion}")
        
        # Financing options
        if recommendations.get("financing_options"):
            message_parts.append("\n**💳 Payment Options:**")
            for option in recommendations["financing_options"][:3]:
                message_parts.append(f"• {option}")
        
        # Alternatives
        if recommendations.get("alternatives"):
            message_parts.append("\n**🔄 Alternative Options:**")
            for alt in recommendations["alternatives"][:2]:
                if isinstance(alt, dict):
                    message_parts.append(f"• {alt.get('item', 'Alternative')} — {alt.get('price_range', 'Varies')}")
                else:
                    message_parts.append(f"• {alt}")
        
        # Follow-up question
        message_parts.append("\n\n**❓ Would you like me to:**")
        message_parts.append("• Compare specific models?")
        message_parts.append("• Find current deals/prices?")
        message_parts.append("• Create a savings plan for this?")
        message_parts.append("• Suggest alternatives in different price ranges?")
        
        return {
            "action": "fallback_recommendations",
            "message": "\n".join(message_parts),
            "data": {
                "product": product_info["product"],
                "category": product_info.get("category"),
                "recommendations": recommendations,
                "affordability": financial_check,
                "common_comparisons": self._get_common_comparisons(product_info.get("product", "")),
                "related_searches": self._get_related_searches(product_info.get("product", ""))
            }
        }
    
    def _get_budget_tip(self, product_info: Dict, financial_check: Dict) -> str:
        """Generate budget-specific tip"""
        product = product_info.get("product", "")
        
        tips = {
            "laptop": "Consider refurbished business laptops (ThinkPad, Dell Latitude) for 50% savings",
            "phone": "Buy last year's flagship or consider mid-range (Google Pixel A-series, Samsung A-series)",
            "gym": "Check if employer offers wellness benefits or gym discounts",
            "travel": "Use travel rewards credit cards and book during off-peak",
            "course": "Look for employer tuition reimbursement or free alternatives (Coursera audit)",
            "furniture": "Check Facebook Marketplace for barely-used furniture at 70% off"
        }
        
        for key, tip in tips.items():
            if key in product.lower():
                return tip
        
        return "Consider setting up automatic savings for this purchase"
    
    def _get_quick_alternatives(self, product_info: Dict) -> List[str]:
        """Get quick alternative suggestions"""
        product = product_info.get("product", "").lower()
        
        alternatives = {
            "laptop": ["Chromebook (budget)", "iPad + Keyboard (versatile)", "Desktop (more power for less $)"],
            "phone": ["Refurbished models", "Last year's flagship", "Mid-range (Google Pixel A-series)"],
            "gym": ["ClassPass", "Home workout app", "Local community center", "Outdoor activities"],
            "travel": ["Staycation", "Road trip", "Off-season travel", "Travel rewards cards"],
            "course": ["Coursera audit", "YouTube tutorials", "Library books", "Local workshops"],
            "furniture": ["IKEA", "Facebook Marketplace", "Wayfair sales", "Rent-to-own options"]
        }
        
        for key, alts in alternatives.items():
            if key in product:
                return alts
        
        return ["Refurbished/used options", "Previous model/year", "Alternative brands"]
    
    def _get_common_comparisons(self, product: str) -> List[Dict]:
        """Get common product comparisons"""
        product_lower = product.lower()
        
        comparisons = {
            "laptop": [
                {"title": "MacBook vs Windows", "factors": ["Ecosystem", "Software compatibility", "Price"]},
                {"title": "New vs Refurbished", "factors": ["Warranty", "Price difference", "Condition"]}
            ],
            "phone": [
                {"title": "iPhone vs Android", "factors": ["OS preference", "App ecosystem", "Integration"]},
                {"title": "New vs Last year", "factors": ["Features", "Price difference", "Longevity"]}
            ],
            "gym": [
                {"title": "Commercial vs Boutique", "factors": ["Equipment variety", "Class quality", "Price"]},
                {"title": "Gym vs Home", "factors": ["Convenience", "Cost", "Motivation"]}
            ]
        }
        
        for key, comps in comparisons.items():
            if key in product_lower:
                return comps
        
        return []
    
    def _get_related_searches(self, product: str) -> List[str]:
        """Get related search suggestions"""
        product_lower = product.lower()
        
        related = {
            "laptop": ["Best laptop for programming", "Laptop deals this week", "Student discounts"],
            "phone": ["Phone deals with trade-in", "Best camera phone", "Longest battery life phone"],
            "gym": ["Gym membership discounts", "Home gym setup cost", "Best fitness apps"],
            "travel": ["Last minute deals", "Travel credit card comparison", "Packing lists"],
            "course": ["Free certifications", "Employer tuition reimbursement", "Online vs in-person"]
        }
        
        for key, searches in related.items():
            if key in product_lower:
                return searches
        
        return ["Best time to buy", "Reviews and ratings", "Financing options"]
    
    def _search_products(self, product_info: Dict) -> Dict[str, Any]:
        """Search for products using the search tool"""
        query = product_info.get("product", "")
        if not query:
            return {"status": "error", "message": "No product specified"}
        
        try:
            # Call the search tool
            result = search_products.invoke({
                "query": query,
                "limit": 5,
                "max_price": product_info.get("max_price")
            })
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _analyze_options(self, search_results: Dict, financial_check: Dict) -> Dict:
        """Analyze and rank product options"""
        products = search_results.get("products", [])
        
        if not products:
            return {
                "options": [],
                "best_value": None
            }
        
        # Score each product
        scored = []
        for product in products:
            score = self._score_product(product, financial_check)
            scored.append({
                **product,
                "score": score
            })
        
        # Sort by score
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        # Find best value (price/quality ratio)
        best_value = None
        best_ratio = 0
        for product in products:
            if product.get("price") and product.get("rating"):
                try:
                    price = float(str(product["price"]).replace('$', '').replace(',', ''))
                    rating = float(product["rating"]) if product["rating"] else 3.0
                    ratio = rating / price * 1000  # Normalize
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_value = product
                except:
                    pass
        
        return {
            "options": scored[:3],  # Top 3
            "best_value": best_value
        }
    
    def _score_product(self, product: Dict, financial_check: Dict) -> float:
        """Score a product based on multiple factors"""
        score = 5.0  # Base score
        
        # Price factor
        if product.get("price"):
            try:
                price = float(str(product["price"]).replace('$', '').replace(',', ''))
                # Lower price = higher score when budget matters
                if not financial_check.get("affordable", True):
                    score += max(0, 5 - (price / 500))  # Bonus for cheaper options
            except:
                pass
        
        # Rating factor
        if product.get("rating"):
            try:
                rating = float(product["rating"])
                score += (rating - 3) * 1.5  # Convert 1-5 to -3 to +3
            except:
                pass
        
        # Store factor (prefer trusted stores)
        trusted_stores = ["amazon", "bestbuy", "walmart", "target", "apple"]
        if product.get("store"):
            store = product["store"].lower()
            if any(trusted in store for trusted in trusted_stores):
                score += 1
        
        return max(1, min(10, score))  # Clamp between 1-10