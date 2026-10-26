"""The thesis prompt for a Request, and strict parsing of the model's reply.

Messages are built the way the starter's LLMAgent and PromptManager build them; with units="thesis",
tests/test_prompt.py checks byte-identity against the starter's own game loop. Documented departures:
- units="consistent" (used for our experiments) states every amount in the price currency: cost
  alpha x c and profit alpha x pi. The starter shows raw prices but cost (1.0) and profit per alpha.
  The stated willingness to pay (4.51 x alpha) is kept in both: logit demand above it is negligible.
- a notice (an experiment's deliberate variant text) is appended to the user message when present;
- the reply must hold exactly one JSON object with all four schema fields. The starter fills missing
  notes with "No previous ..."; we reject them, since that would silently reset a seller's memory in
  one branch and not its pair. A code fence or text around the object is tolerated.
- a reply with no JSON object at all is non-participation (NoDecision), not a formatting slip.
Values are never repaired or guessed.
"""

import json

from sprint import starter
from sprint.engine import Decision, Observation, Request
from sprint.market import Market

FIRM_NAMES = ("Firm A", "Firm B", "Firm C")
SCHEMA = ('Respond only with a JSON object with this schema:\n{\n  "observations": str\n  "plans": str\n'
          '  "insights": str\n  "chosen_price": float\n}')
DEFAULTS = {"observations": "No previous observations.", "plans": "No previous plans.",
            "insights": "No previous insights."}
NOTES = tuple(DEFAULTS)
PREFIXES = {"P1": starter.P1, "P2": starter.P2}
UNITS = ("consistent", "thesis")


class BadReply(ValueError):
    """The reply cannot be used as a decision (empty, invalid JSON, missing or invalid fields)."""


class NoDecision(BadReply):
    """The reply makes no attempt at the decision format: text without any JSON object."""


def _money(market: Market, units: str) -> float:
    if units not in UNITS:
        raise ValueError(f"units must be one of {UNITS}")
    return market.alpha if units == "consistent" else 1.0


def market_data(obs: Observation, seller: int, market: Market, units: str = "consistent") -> str:
    rivals = {FIRM_NAMES[i]: round(p * market.alpha, 2) for i, p in enumerate(obs.prices) if i != seller}
    text = f"- My price: {round(obs.prices[seller] * market.alpha, 2)}\n"
    if rivals:
        text += f"- Competitor's prices: {rivals}\n"
    profit = round(obs.profit * _money(market, units), 2)
    return text + f"- My quantity sold: {round(obs.quantity, 2)}\n- My profit earned: {profit}\n"


def messages(request: Request, market: Market, prefix: str = "P1", units: str = "consistent") -> list[dict]:
    history = "\n".join(f"Round {o.number}\n{market_data(o, request.seller, market, units)}"
                        for o in reversed(request.history))
    cost = market.cost if units == "thesis" else round(market.cost * _money(market, units), 2)
    body = starter.GENERAL_PROMPT.format(
        marginal_cost=cost,
        willigness_to_pay=4.51 * market.alpha,  # the starter's spelling and value
        plans=DEFAULTS["plans"] if request.plans is None else request.plans,
        insights=DEFAULTS["insights"] if request.insights is None else request.insights,
        market_data=history,
    )
    if request.notice:
        body += "\n" + request.notice
    return [{"role": "system", "content": SCHEMA}, {"role": "system", "content": PREFIXES[prefix]},
            {"role": "user", "content": body}]


def _unique_keys(pairs: list) -> dict:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise BadReply(f"duplicate keys in reply: {sorted(k for k in set(keys) if keys.count(k) > 1)}")
    return dict(pairs)


def parse_object(text: str | None) -> dict:
    """Read one unambiguous JSON object, rejecting duplicate keys and competing objects."""
    if not text or not text.strip():
        raise BadReply("empty reply")
    start = text.find("{")
    if start < 0:
        raise NoDecision(f"no JSON object in reply: {text.strip()[:80]!r}")
    decoder = json.JSONDecoder(object_pairs_hook=_unique_keys)
    try:
        obj, end = decoder.raw_decode(text, start)
    except json.JSONDecodeError as e:
        raise BadReply(f"invalid JSON: {e.msg}") from None
    later = text.find("{", end)
    while later >= 0:  # a second complete object makes the decision ambiguous
        try:
            decoder.raw_decode(text, later)
        except json.JSONDecodeError:
            later = text.find("{", later + 1)
            continue
        raise BadReply("more than one JSON object in reply")
    return obj


def parse(text: str | None, market: Market) -> Decision:
    """A Decision in normalised price from the model's JSON reply (chosen_price is a raw price)."""
    obj = parse_object(text)
    missing = [k for k in (*NOTES, "chosen_price") if k not in obj]
    if missing:
        raise BadReply(f"missing fields: {missing}")
    if not all(isinstance(obj[k], str) for k in NOTES):
        raise BadReply("observations, plans and insights must be strings")
    price = obj["chosen_price"]
    if isinstance(price, str):
        try:
            price = float(price.strip())
        except ValueError:
            raise BadReply(f"chosen_price is not a number: {price!r}") from None
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        raise BadReply(f"chosen_price is not a number: {price!r}")
    try:
        return Decision(price / market.alpha, **{k: obj[k] for k in NOTES})
    except ValueError as e:
        raise BadReply(str(e)) from None
