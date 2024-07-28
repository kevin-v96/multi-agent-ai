from typing import TypedDict, List


class MessageHistory(TypedDict):
    weather_agent: List[str]
    currency_agent: List[str]
    user_agent: List[str]
