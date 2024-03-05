from typing import Literal, Union
from dataclasses import dataclass


@dataclass
class ETAGapQuery:
    reference: int
    operation: Literal["increase", "decrease"]


@dataclass
class HeightQuery:
    reference: int
    operation: Literal["lower", "raise"]
    boundary: Literal["min", "max"]
    ip: Literal["1", "2"]


@dataclass
class Intent:
    queryType: Literal["etagap", "height"]
    content: Union[ETAGapQuery, HeightQuery]
