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
# Default client configurations
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CLIENTS: dict[str, ClientConfig] = {
    "capital_one": ClientConfig(
        client_id="capital_one",
        client_name="Capital One",
        primary_brand="capital one",
        search_keywords=[
            # Business credit cards
            "capital one business credit card",
            "capital one business card",
            "capital one business credit cards",
            "capital one credit card for small business",
            "business credit card capital one",
            "capital one credit card business",
            "capital one credit card for business",
            "capital one small business credit card",
            "capital one small business credit cards",
            "capital one credit cards for business",
            "business credit cards capital one",
            "capital one business rewards card",
            "capital one credit card small business",
            "capital one quicksilver business card",
            "capital one spark business credit card",
            "capital one venture business card",
            # General credit cards
            "capital one credit card",
            "best capital one credit card",
            "is capital one a good credit card",
            "capital one 0 interest credit card",
            "capital one no interest credit card",
            "capital one credit card rewards",
            "secured credit card capital one",
            "credit card capital one",
            # Popular products (high volume)
            "capital one venture",
            "capital one quicksilver",
            "capital one savor",
            "capital one venture x",
            "capital one 360",
            "capital one savings",
            "capital one checking",
            # Brand variations & discussions
            "capitalone",
            "cap one card",
            "capital one rewards",
            "capital one approval",
            "capital one customer service",
        ],
        # Use industry competitors for regex-based threat/opportunity detection
        # LLM analysis still identifies competitors naturally from text
        competitors={
            k: v for k, v in FINANCIAL_SERVICES_COMPETITORS.items()
            if k.lower() != "capital one"
        },
        industry="financial_services",
        days_back=30,
        post_limit=500,
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
            client = ClientConfig(
                client_id=client_data["client_id"],
                client_name=client_data["client_name"],
                primary_brand=client_data["primary_brand"],
                search_keywords=client_data.get("search_keywords", [client_data["primary_brand"]]),
                competitors=client_data.get("competitors", {}),
                industry=client_data.get("industry", "general"),
                days_back=client_data.get("days_back", 30),
                post_limit=client_data.get("post_limit", 500),
                enabled=client_data.get("enabled", True),
                metadata=client_data.get("metadata", {}),
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
