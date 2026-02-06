# src/tools/searchProducts.py

import os
import requests
from typing import Dict, Any, List
from langchain.tools import tool

OPENWEBNINJA_API_URL = (
    "https://api.openwebninja.com/realtime-product-search/search-v2"
)
OPENWEBNINJA_API_KEY = os.getenv("OPENWEBNINJA_API_KEY")


@tool
def search_products(
    query: str,
    country: str = "us",
    language: str = "en",
    limit: int = 3,
    sort_by: str = "LOWEST_PRICE",
    max_price: float = None,
) -> Dict[str, Any]:
    """
    Search products via Google Shopping (OpenWebNinja).
    Returns minimal, finance-relevant product info only.
    """

    if not OPENWEBNINJA_API_KEY:
        return {
            "status": "error",
            "message": "OPENWEBNINJA_API_KEY not set in environment variables",
        }

    params = {
        "q": query,
        "country": country,
        "language": language,
        "page": 1,
        "limit": limit,
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
            timeout=10,
        )

        if response.status_code == 429:
            return {
                "status": "limited",
                "message": "Live product prices are temporarily unavailable.",
                "suggestions": [
                    "Try again later",
                    "Consider refurbished or older models",
                    "Check local marketplaces",
                ],
            }

        response.raise_for_status()
        data = response.json()

    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": "Product search service unavailable",
            "details": str(e),
        }

    products = data.get("products", [])

    if not products:
        return {"status": "ok", "query": query, "results_count": 0, "products": []}

    clean_products: List[Dict[str, Any]] = []
    for p in products[:limit]:
        clean_products.append(
            {
                "item": p.get("title"),
                "price": p.get("price"),
                "currency": p.get("currency"),
                "store": p.get("store"),
                "rating": p.get("rating"),
                "url": p.get("link"),
            }
        )

    return {
        "status": "ok",
        "query": query,
        "results_count": len(clean_products),
        "products": clean_products,
    }