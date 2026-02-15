"""
Product Search Tool with caching and fallback options
"""

import os
import requests
import json
import time
from typing import Dict, Any, List, Optional
from langchain.tools import tool
from datetime import datetime, timedelta

# Simple cache for product searches
_product_cache = {}
CACHE_DURATION = 3600  # 1 hour in seconds

OPENWEBNINJA_API_URL = "https://api.openwebninja.com/realtime-product-search/search-v2"
OPENWEBNINJA_API_KEY = os.getenv("OPENWEBNINJA_API_KEY")

# Fallback product database for common items when API is down
FALLBACK_PRODUCTS = {
    "macbook": [
        {
            "title": "Apple MacBook Air 13-inch (M1)",
            "price": "$999",
            "currency": "USD",
            "store": "Apple",
            "rating": 4.8,
            "url": "https://www.apple.com/macbook-air/",
            "estimated": True
        },
        {
            "title": "Apple MacBook Pro 14-inch (M3)",
            "price": "$1,599",
            "currency": "USD",
            "store": "Apple",
            "rating": 4.9,
            "url": "https://www.apple.com/macbook-pro/",
            "estimated": True
        }
    ],
    "laptop": [
        {
            "title": "Dell XPS 13",
            "price": "$1,199",
            "currency": "USD",
            "store": "Dell",
            "rating": 4.7,
            "url": "https://www.dell.com/xps13",
            "estimated": True
        },
        {
            "title": "HP Spectre x360",
            "price": "$1,299",
            "currency": "USD",
            "store": "HP",
            "rating": 4.6,
            "url": "https://www.hp.com/spectre",
            "estimated": True
        }
    ],
    "iphone": [
        {
            "title": "iPhone 15 Pro",
            "price": "$999",
            "currency": "USD",
            "store": "Apple",
            "rating": 4.9,
            "url": "https://www.apple.com/iphone/",
            "estimated": True
        },
        {
            "title": "iPhone 15",
            "price": "$799",
            "currency": "USD",
            "store": "Apple",
            "rating": 4.8,
            "url": "https://www.apple.com/iphone/",
            "estimated": True
        }
    ],
    "headphones": [
        {
            "title": "Sony WH-1000XM5",
            "price": "$398",
            "currency": "USD",
            "store": "Sony",
            "rating": 4.9,
            "url": "https://www.sony.com/headphones",
            "estimated": True
        },
        {
            "title": "Apple AirPods Pro (2nd gen)",
            "price": "$249",
            "currency": "USD",
            "store": "Apple",
            "rating": 4.8,
            "url": "https://www.apple.com/airpods/",
            "estimated": True
        }
    ]
}

def get_cache_key(query: str, country: str, sort_by: str) -> str:
    """Generate cache key"""
    return f"{query}:{country}:{sort_by}"

def get_cached_search(key: str) -> Optional[Dict]:
    """Get cached search result if valid"""
    if key in _product_cache:
        data = _product_cache[key]
        if time.time() - data['timestamp'] < CACHE_DURATION:
            return data['result']
    return None

def cache_search(key: str, result: Dict):
    """Cache search result"""
    _product_cache[key] = {
        'timestamp': time.time(),
        'result': result
    }

def get_fallback_products(query: str) -> List[Dict]:
    """Get fallback products for common queries"""
    query_lower = query.lower()
    
    # Check exact matches
    for key in FALLBACK_PRODUCTS:
        if key in query_lower:
            return FALLBACK_PRODUCTS[key]
    
    # Check categories
    if any(term in query_lower for term in ["computer", "mac", "pc"]):
        return FALLBACK_PRODUCTS["laptop"]
    
    if any(term in query_lower for term in ["phone", "smartphone", "mobile"]):
        return FALLBACK_PRODUCTS["iphone"]
    
    if any(term in query_lower for term in ["audio", "earbuds", "earphones"]):
        return FALLBACK_PRODUCTS["headphones"]
    
    # Generic fallback
    return [
        {
            "title": f"Popular {query.title()} Options",
            "price": "Varies by model",
            "currency": "USD",
            "store": "Multiple retailers",
            "rating": 4.5,
            "url": "#",
            "estimated": True,
            "note": "Check local retailers for current pricing"
        }
    ]

@tool
def search_products(
    query: str,
    country: str = "us",
    language: str = "en",
    limit: int = 3,
    sort_by: str = "RELEVANCE",
    max_price: Optional[float] = None,
    use_fallback: bool = True
) -> Dict[str, Any]:
    """
    Search for products and compare prices.
    
    This tool searches for products and returns pricing information.
    If the live API is unavailable, it provides estimated prices as fallback.
    
    Args:
        query: Product search query (e.g., "MacBook Pro", "iPhone 15")
        country: Country code (us, uk, ca, etc.)
        language: Language code (en, es, fr, etc.)
        limit: Maximum number of results (1-5)
        sort_by: Sort method (RELEVANCE, LOWEST_PRICE, HIGHEST_PRICE, BEST_MATCH)
        max_price: Maximum price filter (optional)
        use_fallback: Whether to use fallback data if API fails
    
    Returns:
        Dict with product information and recommendations
    """
    
    # Validate limit
    limit = max(1, min(5, limit))
    
    # Check cache
    cache_key = get_cache_key(query, country, sort_by)
    cached = get_cached_search(cache_key)
    if cached:
        cached['cached'] = True
        return cached
    
    # Try live API if key exists
    if OPENWEBNINJA_API_KEY:
        params = {
            "q": query,
            "country": country,
            "language": language,
            "page": 1,
            "limit": limit * 2,  # Get more for filtering
            "sort_by": sort_by,
            "product_condition": "ANY",
        }

        if max_price is not None:
            params["max_price"] = max_price

        headers = {"x-api-key": OPENWEBNINJA_API_KEY}

        try:
            response = requests.get(
                OPENWEBNINJA_API_URL,
                params=params,
                headers=headers,
                timeout=5,  # Shorter timeout
            )

            if response.status_code == 200:
                data = response.json()
                products = data.get("products", [])
                
                if products:
                    clean_products = []
                    for p in products[:limit]:
                        clean_products.append({
                            "name": p.get("title", "Unknown"),
                            "price": p.get("price", "Price unavailable"),
                            "currency": p.get("currency", "USD"),
                            "store": p.get("store", "Unknown retailer"),
                            "rating": p.get("rating", "N/A"),
                            "url": p.get("link", "#"),
                            "image": p.get("thumbnail", None),
                            "estimated": False
                        })
                    
                    result = {
                        "status": "success",
                        "query": query,
                        "source": "live",
                        "results_count": len(clean_products),
                        "products": clean_products,
                        "timestamp": datetime.now().isoformat(),
                        "note": None
                    }
                    
                    # Cache the result
                    cache_search(cache_key, result)
                    return result
                    
            elif response.status_code == 429:
                # Rate limited, will use fallback
                pass

        except Exception as e:
            # API failed, will use fallback
            pass
    
    # Use fallback if enabled
    if use_fallback:
        fallback_products = get_fallback_products(query)
        
        # Apply price filter if needed
        if max_price and fallback_products:
            filtered = []
            for p in fallback_products:
                try:
                    price_str = p.get('price', '$0').replace('$', '').replace(',', '')
                    price = float(price_str) if price_str.replace('.', '').isdigit() else 0
                    if price <= max_price or price == 0:
                        filtered.append(p)
                except:
                    filtered.append(p)
            fallback_products = filtered[:limit]
        
        return {
            "status": "success",
            "query": query,
            "source": "estimated",
            "results_count": len(fallback_products),
            "products": fallback_products[:limit],
            "timestamp": datetime.now().isoformat(),
            "note": "Live prices temporarily unavailable - showing estimated prices"
        }
    
    # No results
    return {
        "status": "no_results",
        "query": query,
        "message": f"No products found for '{query}'",
        "suggestions": [
            "Try a more general search term",
            "Check spelling",
            "Browse popular categories instead"
        ]
    }

@tool
def compare_products(
    products: str
) -> Dict[str, Any]:
    """
    Compare multiple products side by side.
    
    Args:
        products: JSON string with list of products to compare
    
    Returns:
        Dict with comparison table and recommendations
    """
    try:
        # Parse products
        if isinstance(products, str):
            product_list = json.loads(products)
        else:
            product_list = products
        
        if not product_list or len(product_list) < 2:
            return {
                "status": "error",
                "message": "Need at least 2 products to compare"
            }
        
        # Create comparison
        comparison = {
            "products": [],
            "price_comparison": {},
            "best_value": None,
            "best_rated": None
        }
        
        # Track best values
        best_price_value = float('inf')
        best_price_product = None
        best_rating = 0
        best_rated_product = None
        
        for p in product_list[:3]:  # Compare up to 3 products
            name = p.get('name') or p.get('title', 'Unknown')
            price_str = p.get('price', '0').replace('$', '').replace(',', '')
            
            try:
                price = float(price_str) if price_str.replace('.', '').isdigit() else 0
            except:
                price = 0
            
            rating_str = p.get('rating', '0')
            try:
                rating = float(rating_str) if rating_str != 'N/A' else 0
            except:
                rating = 0
            
            comparison['products'].append({
                "name": name,
                "price": p.get('price', 'N/A'),
                "price_value": price,
                "rating": rating,
                "store": p.get('store', 'Unknown'),
                "url": p.get('url', '#')
            })
            
            # Track best price
            if price > 0 and price < best_price_value:
                best_price_value = price
                best_price_product = name
            
            # Track best rating
            if rating > best_rating:
                best_rating = rating
                best_rated_product = name
        
        comparison['price_comparison'] = {
            "lowest_price": best_price_product,
            "highest_rated": best_rated_product
        }
        
        # Value score (rating/price)
        value_scores = []
        for p in comparison['products']:
            if p['price_value'] > 0 and p['rating'] > 0:
                value_score = p['rating'] / p['price_value'] * 1000
                value_scores.append((p['name'], value_score))
        
        if value_scores:
            value_scores.sort(key=lambda x: x[1], reverse=True)
            comparison['best_value'] = value_scores[0][0]
        
        return {
            "status": "success",
            "comparison": comparison,
            "recommendation": f"Based on price and rating, {comparison['best_value'] or comparison['price_comparison']['lowest_price']} offers the best value."
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to compare products: {str(e)}"
        }

@tool
def check_affordability(
    product_price: float,
    monthly_income: float,
    current_savings: Optional[float] = None,
    emergency_fund_target: Optional[float] = None
) -> Dict[str, Any]:
    """
    Check if a purchase is affordable based on income and savings.
    
    Args:
        product_price: Price of the product
        monthly_income: Monthly after-tax income
        current_savings: Current savings (optional)
        emergency_fund_target: Emergency fund target (optional)
    
    Returns:
        Dict with affordability analysis
    """
    analysis = {
        "product_price": product_price,
        "monthly_income": monthly_income,
        "affordability_score": 0,
        "recommendation": "",
        "factors": []
    }
    
    # Check against monthly income
    income_ratio = product_price / monthly_income
    
    if income_ratio <= 0.1:
        analysis["affordability_score"] += 3
        analysis["factors"].append(f"✓ Price is only {income_ratio*100:.1f}% of monthly income")
    elif income_ratio <= 0.25:
        analysis["affordability_score"] += 2
        analysis["factors"].append(f"⚠️ Price is {income_ratio*100:.1f}% of monthly income - consider saving for 1-2 months")
    elif income_ratio <= 0.5:
        analysis["affordability_score"] += 1
        analysis["factors"].append(f"⚠️ Price is {income_ratio*100:.1f}% of monthly income - consider saving for 2-3 months")
    else:
        analysis["affordability_score"] -= 1
        analysis["factors"].append(f"❌ Price exceeds 50% of monthly income - this is a significant purchase")
    
    # Check against savings
    if current_savings is not None:
        savings_ratio = product_price / current_savings if current_savings > 0 else float('inf')
        
        if savings_ratio <= 0.1:
            analysis["affordability_score"] += 2
            analysis["factors"].append(f"✓ Price is only {savings_ratio*100:.1f}% of current savings")
        elif savings_ratio <= 0.25:
            analysis["affordability_score"] += 1
            analysis["factors"].append(f"⚠️ Price would use {savings_ratio*100:.1f}% of your savings")
        else:
            analysis["factors"].append(f"❌ Price would use over 25% of your savings")
        
        # Check emergency fund impact
        if emergency_fund_target:
            emergency_impact = (emergency_fund_target - current_savings + product_price) / emergency_fund_target
            if emergency_impact > 0.2:
                analysis["affordability_score"] -= 2
                analysis["factors"].append(f"❌ This purchase would significantly impact your emergency fund goal")
    
    # Final recommendation
    if analysis["affordability_score"] >= 5:
        analysis["recommendation"] = "✅ This purchase appears affordable based on your current finances."
        analysis["verdict"] = "affordable"
    elif analysis["affordability_score"] >= 2:
        analysis["recommendation"] = "⚠️ This purchase is reasonable but consider saving specifically for it."
        analysis["verdict"] = "caution"
        analysis["saving_suggestion"] = f"Save ${product_price/3:.2f} per month for 3 months"
    else:
        analysis["recommendation"] = "❌ This purchase may strain your finances. Consider alternatives or waiting."
        analysis["verdict"] = "not_recommended"
        analysis["alternatives"] = [
            "Look for refurbished or older models",
            "Wait for seasonal sales",
            "Consider a more affordable alternative"
        ]
    
    return analysis