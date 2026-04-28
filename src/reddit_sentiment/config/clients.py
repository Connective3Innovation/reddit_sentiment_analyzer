# src/reddit_sentiment/config/clients.py
"""
Client configuration for multi-tenant competitive intelligence.

Each client has:
- A primary brand to track
- A list of competitors with aliases
- Industry-specific topic patterns
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


@dataclass
class ClientConfig:
    """Configuration for a single client."""
    client_id: str
    client_name: str
    primary_brand: str
    search_keywords: list[str]  # Keywords to search Reddit for
    competitors: dict[str, list[str]]  # competitor_name -> [aliases]
    industry: str = "general"
    days_back: int = 30
    post_limit: int = 500
    enabled: bool = True
    metadata: dict = field(default_factory=dict)
    target_subreddits: list[str] = field(default_factory=list)  # Subreddits to search (empty = all)
    require_keyword_match: bool = True  # Filter posts that don't contain brand keywords


# ─────────────────────────────────────────────────────────────────────────────
# Industry-specific competitor templates
# ─────────────────────────────────────────────────────────────────────────────

FINANCIAL_SERVICES_COMPETITORS = {
    # Major banks
    "chase": ["chase", "jp morgan", "jpmorgan", "chase bank"],
    "bank of america": ["bank of america", "bofa", "boa", "bankofamerica"],
    "wells fargo": ["wells fargo", "wellsfargo", "wells"],
    "citi": ["citi", "citibank", "citigroup"],
    "capital one": ["capital one", "capitalone", "cap one"],
    "us bank": ["us bank", "usbank", "u.s. bank"],
    "pnc": ["pnc bank", "pnc"],
    "td bank": ["td bank", "td ameritrade"],
    # Credit cards
    "amex": ["amex", "american express", "americanexpress"],
    "discover": ["discover card", "discover"],
    # Online banks
    "ally": ["ally bank", "ally"],
    "sofi": ["sofi", "social finance"],
    "chime": ["chime", "chime bank"],
    "marcus": ["marcus", "goldman sachs marcus"],
    # Credit unions
    "navy federal": ["navy federal", "nfcu"],
    "usaa": ["usaa"],
}

TECH_COMPETITORS = {
    "google": ["google", "alphabet", "googl"],
    "microsoft": ["microsoft", "msft", "azure"],
    "amazon": ["amazon", "aws", "amzn"],
    "apple": ["apple", "aapl", "ios"],
    "meta": ["meta", "facebook", "fb", "instagram"],
    "netflix": ["netflix", "nflx"],
    "salesforce": ["salesforce", "sfdc"],
    "oracle": ["oracle", "orcl"],
    "ibm": ["ibm"],
    "adobe": ["adobe", "adbe"],
}

RETAIL_COMPETITORS = {
    "walmart": ["walmart", "wal-mart", "wmt"],
    "target": ["target", "tgt"],
    "amazon": ["amazon", "amzn", "prime"],
    "costco": ["costco", "cost"],
    "kroger": ["kroger", "kr"],
    "walgreens": ["walgreens", "wba"],
    "cvs": ["cvs", "cvs health"],
    "home depot": ["home depot", "hd"],
    "lowes": ["lowes", "lowe's"],
    "best buy": ["best buy", "bby"],
}

INDUSTRY_TEMPLATES = {
    "financial_services": FINANCIAL_SERVICES_COMPETITORS,
    "tech": TECH_COMPETITORS,
    "retail": RETAIL_COMPETITORS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Industry-specific subreddit targeting (reduces noise from irrelevant posts)
# ─────────────────────────────────────────────────────────────────────────────

FINANCIAL_SUBREDDITS = [
    "personalfinance",      # 19M members - credit cards, banking, general finance
    "CreditCards",          # 750K members - primary credit card discussions
    "churning",             # 400K members - credit card rewards optimization
    "FinancialPlanning",    # 200K members - long-term financial decisions
    "Banking",              # 150K members - bank comparisons
    "CRedit",               # 100K members - credit building
    "smallbusiness",        # 1.5M members - business credit cards
    "Entrepreneur",         # 2M members - business banking
    "povertyfinance",       # 1M members - budget banking
    "fatFIRE",              # 500K members - premium cards
]

TECH_SUBREDDITS = [
    "technology",
    "programming",
    "webdev",
    "software",
    "startups",
    "SaaS",
]

RETAIL_SUBREDDITS = [
    "Frugal",
    "deals",
    "shopping",
    "Costco",
    "Target",
    "walmart",
]

INDUSTRY_SUBREDDITS = {
    "financial_services": FINANCIAL_SUBREDDITS,
    "tech": TECH_SUBREDDITS,
    "retail": RETAIL_SUBREDDITS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Capital One Segment-Specific Subreddits
# ─────────────────────────────────────────────────────────────────────────────

CAPITAL_ONE_TECH_SUBREDDITS = [
    # AI & Machine Learning
    "MachineLearning",       # 3M+ - ML research and applications
    "artificial",            # 1M+ - General AI discussions
    "LocalLLaMA",            # 500K - LLM enthusiasts
    "OpenAI",                # 1M+ - AI products and tools
    "ChatGPT",               # 2M+ - AI users
    # Data Science
    "datascience",           # 1M - Data science professionals
    "dataengineering",       # 200K - Data infrastructure
    "analytics",             # 100K - Analytics professionals
    "rstats",                # 100K - R/statistics users
    # Software Development
    "programming",           # 5M+ - General programming
    "webdev",                # 2M - Web developers
    "ExperiencedDevs",       # 200K - Senior engineers
    "cscareerquestions",     # 1M+ - Tech career discussions
    # Tech Finance (where tech workers discuss finances)
    "personalfinance",       # 19M - Personal finance
    "CreditCards",           # 750K - Credit card discussions
    "fatFIRE",               # 500K - High earners (many in tech)
]

CAPITAL_ONE_B2B_SUBREDDITS = [
    # Business subreddits
    "smallbusiness",         # 1.5M - Small business owners
    "Entrepreneur",          # 2M - Business founders
    "startups",              # 1M - Startup founders
    "ecommerce",             # 300K - Online business owners
    "freelance",             # 200K - Freelancers need business accounts
    "selfemployed",          # 100K - Self-employed
    "accounting",            # 300K - Business finance
    "sweatystartup",         # 200K - Service businesses
    # Credit card subreddits (business cards discussed here too)
    "CreditCards",           # 750K - Business card reviews/comparisons
    "churning",              # 400K - Business card signup bonuses
]

CAPITAL_ONE_CONSUMER_SUBREDDITS = [
    "personalfinance",       # 19M - Personal finance
    "CreditCards",           # 750K - Credit card discussions
    "churning",              # 400K - Rewards optimization
    "povertyfinance",        # 1M - Budget-conscious
    "FinancialPlanning",     # 200K - Long-term planning
    "CRedit",                # 100K - Credit building
    "fatFIRE",               # 500K - High-net-worth
    "Frugal",                # 2M - Value-conscious consumers
    "Banking",               # 150K - Bank comparisons
]


# ─────────────────────────────────────────────────────────────────────────────
# Default client configurations
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CLIENTS: dict[str, ClientConfig] = {
    # ─────────────────────────────────────────────────────────────────────────
    # Capital One Segment Configs (3 focused segments instead of 1 general)
    # ─────────────────────────────────────────────────────────────────────────

    "capital_one_tech": ClientConfig(
        client_id="capital_one_tech",
        client_name="Capital One (Technologists)",
        primary_brand="capital one",
        search_keywords=[
            # General brand mentions in tech communities
            "capital one",
            "capital one credit card",
            "capital one venture",
            "capital one savor",
            "capital one quicksilver",
            # Tech-specific financial needs
            "capital one software subscription",
            "capital one aws credits",
            "capital one cloud spending",
            "capital one tech purchases",
            # Banking for tech workers
            "capital one savings",
            "capital one 360",
            "capital one high yield",
            # Rewards relevant to tech workers
            "capital one travel rewards",
            "capital one lounge",
            "venture x lounge",
        ],
        competitors={
            # Credit cards popular with tech workers
            "chase sapphire": ["chase sapphire", "sapphire reserve", "sapphire preferred", "csr", "csp"],
            "amex platinum": ["amex platinum", "amex plat", "platinum card"],
            "amex gold": ["amex gold", "gold card"],
            "citi": ["citi", "citi premier", "citi double cash"],
            # Online banks popular with tech workers
            "sofi": ["sofi", "social finance"],
            "ally": ["ally bank", "ally"],
            "marcus": ["marcus", "goldman marcus"],
            "wealthfront": ["wealthfront"],
            "betterment": ["betterment"],
        },
        industry="tech",
        days_back=30,
        post_limit=500,
        target_subreddits=CAPITAL_ONE_TECH_SUBREDDITS,
        require_keyword_match=True,
    ),

    "capital_one_b2b": ClientConfig(
        client_id="capital_one_b2b",
        client_name="Capital One (Business Banking)",
        primary_brand="capital one",
        search_keywords=[
            # Business cards
            "capital one spark",
            "capital one spark business",
            "capital one business card",
            "capital one business credit card",
            "capital one small business",
            "spark cash",
            "spark miles",
            # Business banking
            "capital one business checking",
            "capital one business account",
            "capital one merchant services",
            "capital one business banking",
            # Business credit
            "capital one business line of credit",
            "capital one business loan",
        ],
        competitors={
            # Business credit cards
            "chase ink": ["chase ink", "ink business", "ink preferred", "ink unlimited", "ink cash"],
            "amex business": ["amex business", "business platinum", "business gold", "blue business", "amex biz"],
            "bank of america": ["bofa business", "bank of america business", "boa business"],
            # Startup/SMB cards
            "brex": ["brex", "brex card", "brex cash"],
            "ramp": ["ramp", "ramp card"],
            "divvy": ["divvy", "divvy card", "bill divvy"],
            # Business banking alternatives
            "mercury": ["mercury", "mercury bank", "mercury business"],
            "bluevine": ["bluevine", "blue vine", "bluevine business"],
            "novo": ["novo", "novo bank", "novo business"],
        },
        industry="financial_services",
        days_back=30,
        post_limit=500,
        target_subreddits=CAPITAL_ONE_B2B_SUBREDDITS,
        require_keyword_match=True,
    ),

    "capital_one_consumer": ClientConfig(
        client_id="capital_one_consumer",
        client_name="Capital One (Consumer)",
        primary_brand="capital one",
        search_keywords=[
            # Consumer credit cards
            "capital one venture",
            "capital one venture x",
            "capital one quicksilver",
            "capital one savor",
            "capital one savor one",
            "capital one platinum",
            "capital one secured",
            # Banking products
            "capital one 360",
            "capital one savings",
            "capital one checking",
            "capital one high yield",
            # General consumer
            "capital one credit card",
            "capital one rewards",
            "capital one approval",
            "capital one credit limit",
            "capital one customer service",
        ],
        competitors={
            k: v for k, v in FINANCIAL_SERVICES_COMPETITORS.items()
            if k.lower() != "capital one"
        },
        industry="financial_services",
        days_back=30,
        post_limit=500,
        target_subreddits=CAPITAL_ONE_CONSUMER_SUBREDDITS,
        require_keyword_match=True,
    ),
}


class ClientRegistry:
    """Registry for managing client configurations."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize registry.

        Args:
            config_path: Optional path to JSON config file
        """
        self._clients: dict[str, ClientConfig] = {}
        self._load_defaults()

        # Use config_path if provided, otherwise fall back to env var (not both)
        resolved_path = config_path or os.getenv("REDDIT_CLIENTS_CONFIG")
        if resolved_path and os.path.exists(resolved_path):
            self._load_from_file(resolved_path)

    def _load_defaults(self):
        """Load default client configurations."""
        self._clients.update(DEFAULT_CLIENTS)

    def _load_from_file(self, path: str):
        """Load clients from JSON config file."""
        with open(path, 'r') as f:
            data = json.load(f)

        for client_data in data.get("clients", []):
            # Get industry subreddits if not specified
            industry = client_data.get("industry", "general")
            default_subreddits = INDUSTRY_SUBREDDITS.get(industry, [])

            client = ClientConfig(
                client_id=client_data["client_id"],
                client_name=client_data["client_name"],
                primary_brand=client_data["primary_brand"],
                search_keywords=client_data.get("search_keywords", [client_data["primary_brand"]]),
                competitors=client_data.get("competitors", {}),
                industry=industry,
                days_back=client_data.get("days_back", 30),
                post_limit=client_data.get("post_limit", 500),
                enabled=client_data.get("enabled", True),
                metadata=client_data.get("metadata", {}),
                target_subreddits=client_data.get("target_subreddits", default_subreddits),
                require_keyword_match=client_data.get("require_keyword_match", True),
            )
            self._clients[client.client_id] = client

    def get(self, client_id: str) -> Optional[ClientConfig]:
        """Get client by ID."""
        return self._clients.get(client_id)

    def list_clients(self, enabled_only: bool = True) -> list[ClientConfig]:
        """List all clients."""
        clients = list(self._clients.values())
        if enabled_only:
            clients = [c for c in clients if c.enabled]
        return clients

    def add_client(self, config: ClientConfig) -> None:
        """Add or update a client."""
        self._clients[config.client_id] = config

    def remove_client(self, client_id: str) -> None:
        """Remove a client."""
        self._clients.pop(client_id, None)

    def create_client(
        self,
        client_id: str,
        client_name: str,
        primary_brand: str,
        industry: str = "general",
        custom_competitors: Optional[dict[str, list[str]]] = None,
    ) -> ClientConfig:
        """
        Create a new client with industry defaults.

        Args:
            client_id: Unique identifier
            client_name: Display name
            primary_brand: Brand to track
            industry: Industry for competitor template
            custom_competitors: Override default competitors

        Returns:
            Created ClientConfig
        """
        # Get industry template or empty dict
        template = INDUSTRY_TEMPLATES.get(industry, {})

        # Use custom competitors or template (excluding primary brand)
        if custom_competitors:
            competitors = custom_competitors
        else:
            competitors = {
                k: v for k, v in template.items()
                if k.lower() != primary_brand.lower()
            }

        config = ClientConfig(
            client_id=client_id,
            client_name=client_name,
            primary_brand=primary_brand,
            search_keywords=[primary_brand],
            competitors=competitors,
            industry=industry,
            target_subreddits=INDUSTRY_SUBREDDITS.get(industry, []),
            require_keyword_match=True,
        )

        self.add_client(config)
        return config


# Thread-safe singleton registry using lru_cache
@lru_cache(maxsize=1)
def get_registry() -> ClientRegistry:
    """Get the global client registry (thread-safe singleton)."""
    return ClientRegistry()


def reset_registry() -> None:
    """Reset the registry cache (useful for testing)."""
    get_registry.cache_clear()


def get_client(client_id: str) -> Optional[ClientConfig]:
    """Convenience function to get a client."""
    return get_registry().get(client_id)
