"""
Market Agent - Searches for products and compares options
"""

from typing import Dict, Any, List, Optional
from tools.searchProducts import search_products

class MarketAgent:
    """
    Specialized agent for product research and comparison
    """
    
    def __init__(self):
        self.search_history = {}
    
    def process(self, query: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process product search request
        
        Args:
            query: User query about products
            user_id: User identifier
            context: Current context including financial situation
        
        Returns:
            Product recommendations and analysis
        """
        # Extract product information
        product_info = self._extract_product_info(query)
        
        if not product_info.get("product"):
            return {
                "action": "clarify",
                "message": "What product are you interested in?",
                "data": {}
            }
        
        # Check user's financial capability
        financial_check = self._check_affordability(product_info, context)
        
        # Search for products
        search_results = self._search_products(product_info)
        
        if search_results.get("status") == "error":
            return {
                "action": "error",
                "message": "Unable to search for products at the moment.",
                "data": {"fallback": self._get_fallback_options(product_info)}
            }
        
        # Analyze and rank options
        recommendations = self._analyze_options(search_results, financial_check)
        
        return {
            "action": "recommendations",
            "message": f"I found some options for {product_info['product']}:",
            "data": {
                "product": product_info["product"],
                "options": recommendations["options"],
                "best_value": recommendations["best_value"],
                "affordability": financial_check,
                "search_query": product_info
            }
        }
    
    def _extract_product_info(self, query: str) -> Dict[str, Any]:
        """Extract product details from query"""
        info = {
            "product": None,
            "max_price": None,
            "preferences": []
        }
        
        # Simple extraction - in production would use LLM
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ["macbook", "laptop", "phone", "tablet", "car"]:
                info["product"] = word
            elif word.lower() in ["under", "below", "max", "budget"] and i+1 < len(words):
                try:
                    # Look for price
                    for j in range(i+1, min(i+4, len(words))):
                        if words[j].replace('$', '').replace(',', '').isdigit():
                            info["max_price"] = float(words[j].replace('$', '').replace(',', ''))
                            break
                except:
                    pass
        
        return info
    
    def _check_affordability(self, product_info: Dict, context: Dict) -> Dict[str, Any]:
        """Check if product is affordable based on user's financial situation"""
        result = {
            "affordable": True,
            "impact": "minimal",
            "suggestions": []
        }
        
        # Get user's financial data from context
        user_data = context.get("user_profile", {})
        monthly_income = user_data.get("income", 0) / 12 if user_data.get("income") else 0
        
        if product_info.get("max_price") and monthly_income > 0:
            price = product_info["max_price"]
            if price > monthly_income * 0.5:
                result["affordable"] = False
                result["impact"] = "high"
                result["suggestions"].append("This would consume over 50% of your monthly income")
            elif price > monthly_income * 0.2:
                result["impact"] = "medium"
                result["suggestions"].append("Consider saving for 2-3 months before purchasing")
        
        return result
    
    def _search_products(self, product_info: Dict) -> Dict[str, Any]:
        """Search for products using the search tool"""
        query = product_info.get("product", "")
        if not query:
            return {"status": "error", "message": "No product specified"}
        
        # Call the search tool
        result = search_products.invoke({
            "query": query,
            "limit": 5,
            "max_price": product_info.get("max_price")
        })
        
        return result
    
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
        
        # Price factor (lower is better if budget-conscious)
        if product.get("price") and financial_check.get("affordable") is False:
            try:
                price = float(str(product["price"]).replace('$', '').replace(',', ''))
                # Higher price = lower score when unaffordable
                score -= min(price / 1000, 3)  # Cap at -3
            except:
                pass
        
        # Rating factor
        if product.get("rating"):
            try:
                rating = float(product["rating"])
                score += rating - 3  # Convert 1-5 to -2 to +2
            except:
                pass
        
        # Store factor (prefer known stores)
        trusted_stores = ["amazon", "bestbuy", "walmart", "target"]
        if product.get("store"):
            store = product["store"].lower()
            if any(trusted in store for trusted in trusted_stores):
                score += 1
        
        return max(1, min(10, score))  # Clamp between 1-10
    
    def _get_fallback_options(self, product_info: Dict) -> List[Dict]:
        """Get fallback options when search fails"""
        product = product_info.get("product", "item")
        return [
            {
                "item": f"New {product}",
                "suggestion": "Check manufacturer's website for deals",
                "type": "new"
            },
            {
                "item": f"Refurbished {product}",
                "suggestion": "Consider certified refurbished for savings",
                "type": "refurbished"
            },
            {
                "item": f"Previous model {product}",
                "suggestion": "Last year's model often offers best value",
                "type": "previous_model"
            }
        ]