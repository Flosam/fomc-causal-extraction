"""
Economic term standardization for FOMC causal triples.

Maps varied natural language expressions to broad economic concept categories
for cleaner knowledge graph construction.
"""

from typing import Dict, Optional
import re


# Canonical economic concept mappings - CONSOLIDATED TO BROAD CATEGORIES
# Format: {canonical_concept: [variant1, variant2, ...]}
ECONOMIC_CONCEPTS = {
    # Inflation and prices (all price-related concepts)
    "inflation": [
        "inflation",
        "rising prices",
        "price increases",
        "price pressures",
        "inflationary pressures",
        "elevated inflation",
        "consumer price inflation",
        "price inflation",
        "inflation pressures",
        "inflation to remain above",
        "upward pressure on inflation",
        "core inflation",
        "inflation forecasts",
        "easing inflationary pressures",
        "disinflationary process",
        "total consumer price increase",
        "upside inflation risks",
        "price stability",
        "stable prices",
        "core inflation edge down",
        "core services excluding shelter inflation",
        "core pce price inflation",
        "reduced pressure on goods prices",
        "lower inflation",
        "inflation expectations",
        "longer-term inflation expectations",
    ],
    
    # Economic activity / GDP / Output / Growth (all output concepts)
    "economic_activity": [
        "economic activity",
        "u.s. economic activity",
        "domestic economic activity",
        "gdp",
        "output",
        "economic growth",
        "growth",
        "expansion",
        "real gdp growth",
        "sustainable growth in output",
        "solid real pce growth",
        "economic expansion",
        "period of below-trend real gdp growth",
        "economic growth moderating",
        "real economy",
        "real economy continued to be in a good place",
        "these gains",
        "strong domestic demand",
        "reduction in economic activity",
        "pickup in exports",
        "production",
        "production of motor vehicles",
        "industrial activity",
        "industrial production",
        "convergence of real gdp growth",
    ],
    
    # Monetary policy and interest rates (all Fed policy)
    "monetary_policy": [
        "monetary policy",
        "policy stance",
        "federal funds rate",
        "target range for the federal funds rate",
        "ongoing increases in the target range for the federal funds rate",
        "more aggressive pace toward reaching a neutral policy stance",
        "maintaining the current highly accommodative stance of monetary policy",
        "proximity of the federal funds rate to the effective lower bound",
        "monetary and financial conditions",
        "revised statement of risks",
        "monetary tightening",
        "policy rate",
        "interest rates",
        "interest rate",
        "higher mortgage interest rates",
        "mortgage rates",
        "policy easing",
        "expectations regarding future monetary policy easing",
        "steady monetary policy",
        "raised assessment of the appropriate path of the federal funds rate",
        "appropriate policy",
        "temporarily raise the federal funds rate",
        "tightening monetary policy",
        "lagged cumulative effect of policy tightening",
    ],
    
    # Financial conditions (markets, credit, banking)
    "financial_conditions": [
        "financial conditions",
        "financial markets",
        "sizable reactions in financial markets",
        "banking-sector developments",
        "recent developments in the banking sector",
        "banking sector",
        "credit conditions",
        "investors seeking safety and liquidity",
        "household wealth",
        "declining stock market prices",
        "financial strains",
        "rise in longer-term yields",
        "vulnerabilities at nonbank financial institutions",
        "vulnerabilities at some banks",
        "unwarranted easing in financial conditions",
    ],
    
    # Consumer spending / household consumption
    "consumer_spending": [
        "consumer spending",
        "household spending",
        "spending by businesses and households",
        "household consumption",
        "consumption",
        "pce",
        "personal consumption expenditures",
        "growth in consumer spending",
    ],
    
    # Business investment / capital spending
    "business_investment": [
        "business investment",
        "capital spending",
        "investment spending",
        "broader weakness in investment spending",
        "investment",
        "business spending",
        "increased investment in technology",
        "investment in technology or in business process improvements",
    ],
    
    # Housing / real estate
    "housing": [
        "housing",
        "housing activity",
        "housing demand",
        "homebuilding",
        "residential construction",
        "housing sector",
        "housing market",
        "strength in homebuilding",
        "underlying strength for the housing sector",
    ],
    
    # Labor market / Employment
    "labor_market": [
        "labor market",
        "employment",
        "unemployment",
        "jobs",
        "job market",
        "labor supply",
        "labor force",
        "hiring",
        "job growth",
        "employment growth",
        "tight labor markets",
        "slower hiring",
        "softening in the growth of labor demand",
        "extended unemployment benefits",
        "unit labor costs",
        "convergence of the unemployment rate",
    ],
    
    # Aggregate demand
    "aggregate_demand": [
        "aggregate demand",
        "demand",
        "total demand",
        "overall demand",
    ],
    
    # Trade / external / international
    "trade": [
        "external conditions",
        "foreign economic activity",
        "global economy",
        "international developments",
        "exports",
        "imports",
        "trade",
        "currency depreciation",
        "previous increases in the exchange value of the dollar",
    ],
    
    # Supply factors / supply chain
    "supply": [
        "supply",
        "supply chain",
        "supply constraints",
        "supply disruptions",
    ],
    
    # Energy / commodities
    "energy_prices": [
        "energy prices",
        "oil prices",
        "crude oil",
        "natural gas",
        "earlier declines in energy prices",
        "earlier declines in oil prices",
        "higher energy prices",
    ],
    
    # Immigration / demographics
    "immigration": [
        "immigration",
        "increased immigration",
        "demographic",
    ],
    
    # Expectations / sentiment
    "expectations": [
        "expectations",
        "inflation expectations",
        "longer-term inflation expectations becoming unanchored",
        "incoming information",
    ],
}


def build_term_to_concept_map() -> Dict[str, str]:
    """Build reverse mapping from term variants to canonical concepts."""
    term_map = {}
    for concept, variants in ECONOMIC_CONCEPTS.items():
        for variant in variants:
            # Normalize: lowercase and strip
            normalized = variant.lower().strip()
            term_map[normalized] = concept
    return term_map


# Pre-build the reverse mapping for efficiency
TERM_TO_CONCEPT = build_term_to_concept_map()


def normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, strip, compress whitespace."""
    if not text or pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def standardize_term(term: str, drop_unmapped: bool = True) -> str:
    """
    Map an economic term to its canonical broad concept category.
    
    Args:
        term: Raw term from cause or effect field
        drop_unmapped: If True, return empty string for unmapped terms (drops non-economic phrases)
        
    Returns:
        Canonical concept name, or empty string if no mapping found and drop_unmapped=True
    """
    if not term or pd.isna(term):
        return ""
    
    normalized = normalize_text(term)
    
    # Exact match
    if normalized in TERM_TO_CONCEPT:
        return TERM_TO_CONCEPT[normalized]
    
    # Partial match: check if any mapped term is substring
    # Prioritize longest matches
    matches = []
    for mapped_term, concept in TERM_TO_CONCEPT.items():
        if mapped_term in normalized or normalized in mapped_term:
            matches.append((len(mapped_term), concept, mapped_term))
    
    if matches:
        # Return concept from longest matching term
        matches.sort(reverse=True)
        return matches[0][1]
    
    # No match - return empty if dropping, otherwise return original
    if drop_unmapped:
        return ""
    else:
        return normalized


def standardize_triple(cause: str, effect: str, drop_unmapped: bool = True) -> tuple[str, str]:
    """
    Standardize both cause and effect terms in a causal triple.
    
    Args:
        cause: Raw cause text
        effect: Raw effect text
        drop_unmapped: If True, drop non-economic terms
        
    Returns:
        (standardized_cause, standardized_effect) tuple
    """
    return standardize_term(cause, drop_unmapped), standardize_term(effect, drop_unmapped)


# Make pandas available for type checking
try:
    import pandas as pd
except ImportError:
    # Fallback for basic usage without pandas
    class _FakePd:
        @staticmethod
        def isna(x):
            return x is None or (isinstance(x, float) and x != x)
    pd = _FakePd()


if __name__ == "__main__":
    # Test with sample terms
    test_terms = [
        "inflation",
        "rising prices",
        "economic activity",
        "GDP",
        "monetary policy",
        "federal funds rate",
        "consumer spending",
        "housing activity",
        "some unknown term",
        "recent developments in the banking sector",
        "elevated inflation",
    ]
    
    print("Term Standardization Tests (Broad Categories):")
    print("-" * 60)
    for term in test_terms:
        standardized = standardize_term(term, drop_unmapped=True)
        if standardized:
            print(f"{term:50} → {standardized}")
        else:
            print(f"{term:50} → [DROPPED]")
    
    print(f"\nTotal broad categories: {len(ECONOMIC_CONCEPTS)}")
    print(f"Categories: {', '.join(sorted(ECONOMIC_CONCEPTS.keys()))}")
