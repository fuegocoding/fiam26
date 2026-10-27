"""Research pod charter identifiers; no built-in strategies."""

import enum


class Charter(str, enum.Enum):
    FUNDAMENTALS_EXPECTATIONS = "business_fundamentals_expectations"
    CORPORATE_EVENTS = "corporate_events_capital_allocation"
    VALUATION_RELATIVE_VALUE = "valuation_fundamental_relative_value"
    MARKET_DYNAMICS = "market_dynamics_information_diffusion"
