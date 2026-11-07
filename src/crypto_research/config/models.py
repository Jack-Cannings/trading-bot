"""Research settings intentionally exclude live trading and leverage."""

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, model_validator

from crypto_research.contracts import (
    Identifier,
    Positive,
    Record,
    StrategyConfig,
    Timeframe,
    TradingCosts,
)


class ResearchConfig(TradingCosts):
    exchange: Identifier
    markets: tuple[Annotated[str, Field(pattern=r"^[A-Z0-9]+/[A-Z0-9]+$")], ...] = Field(
        min_length=1
    )
    primary_timeframe: Timeframe = "1d"
    secondary_timeframe: Timeframe = "4h"
    trading_mode: Literal["spot"] = "spot"
    long_only: Literal[True] = True
    starting_capital: Positive = 10000.0
    base_currency: Annotated[str, Field(pattern=r"^[A-Z0-9]+$")] = "USDT"
    leverage: Literal[False] = False
    live_trading: Literal[False] = False
    seed: Annotated[int, Field(ge=0, strict=True)] = 0

    @model_validator(mode="after")
    def consistent_markets(self) -> Self:
        if len(set(self.markets)) != len(self.markets):
            raise ValueError("markets must be unique")
        if any(pair.split("/")[1] != self.base_currency for pair in self.markets):
            raise ValueError("market quote currencies must match base currency")
        if self.primary_timeframe == self.secondary_timeframe:
            raise ValueError("primary and secondary timeframes must differ")
        return self


class ExperimentConfig(Record):
    experiment_id: Identifier
    research: ResearchConfig
    strategies: tuple[StrategyConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_strategies(self) -> Self:
        ids = [strategy.identity for strategy in self.strategies]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "strategy configuration identifiers must be unique within an experiment"
            )
        return self


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently overwriting settings."""


def _mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict[str, object]:
    result: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("YAML mapping keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(path: str | Path) -> ResearchConfig:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.load(stream, Loader=UniqueKeyLoader)
    return ResearchConfig.model_validate(data)
