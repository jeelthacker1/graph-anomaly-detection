"""Strict ingestion contract: tokens only, integer minor units, no PAN/CVV fields."""
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Token = Annotated[str, Field(min_length=6, max_length=96, pattern=r"^[a-z]+_[A-Za-z0-9_-]+$")]


class Payment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    payment_id: Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_-]+$")]
    occurred_at: datetime
    card_id: Token
    merchant_id: Token
    device_id: Token
    ip_token: Token
    amount_minor: Annotated[int, Field(ge=1, le=100_000_000)]
    currency: Literal["USD"] = "USD"
    country: Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
    home_country: Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
    card_present: bool
    three_ds: Literal["verified", "failed", "not_requested"]

    @field_validator("occurred_at", mode="before")
    @classmethod
    def parse_time(cls, value):
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(UTC)

    @field_validator("card_id", "merchant_id", "device_id", "ip_token")
    @classmethod
    def token_prefix(cls, value, info):
        prefix = {"card_id": "card_", "merchant_id": "merchant_", "device_id": "device_", "ip_token": "ip_"}
        if not value.startswith(prefix[info.field_name]):
            raise ValueError(f"{info.field_name} must use prefix {prefix[info.field_name]}")
        return value


class CaseUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_version: Annotated[int, Field(ge=1)]
    status: Literal["open", "investigating", "closed_fraud", "closed_legitimate"]
    note: Annotated[str, Field(min_length=3, max_length=1000)]
