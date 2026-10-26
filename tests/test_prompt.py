"""Golden test: our prompts are byte-identical to what the starter's own game loop sends, round by
round, for the same replies (units="thesis"). Plus the consistent-units display and strict parsing."""

import asyncio
import json

import pytest

import sprint.starter  # noqa: F401  (puts the pinned starter on sys.path)
from sprint.engine import Journal, Observation, Request, play
from sprint.market import Market
from sprint.prompt import BadReply, NoDecision, messages, parse

ROUNDS, MEMORY = 12, 5  # memory shorter than the game, so the window is exercised


def reply(seller: int, round_number: int) -> str:
    """Deterministic replies with two- and three-decimal prices and changing notes."""
    raw = 4.8 + ((seller * 7 + round_number * 13) % 17) / 10 + (0.005 if round_number % 4 == 0 else 0.0)
    return json.dumps({"observations": f"obs {seller}/{round_number}", "plans": f"plan {seller}/{round_number}",
                       "insights": f"insight {seller}/{round_number}", "chosen_price": raw})


def starter_messages(tmp_path, n: int) -> dict:
    from src.agents.LLM_agent import LLMAgent
    from src.environment.calvano import CalvanoDemandEnvironment
    from src.experiment.experiment import Experiment
    from src.prompts.prompts import GENERAL_PROMPT, P1
    from src.prompts.prompts_models import create_pricing_response_model

    sent = {}

    class Recorder:
        model_name, is_remote = "recorder", False

        def __init__(self, seller):
            self.seller, self.calls = seller, 0

        async def complete(self, msgs, temperature, seed):
            self.calls += 1
            sent[self.seller, self.calls] = msgs
            return reply(self.seller, self.calls)

    alpha = 3.2
    agents = [
        LLMAgent(name, prefix=P1, provider=Recorder(i),
                 response_model=create_pricing_response_model(include_wtp=True, wtp_value=4.51 * alpha),
                 memory_length=MEMORY, temperature=0.7, seed=0, prompt_template=GENERAL_PROMPT,
                 env_params={"a": 2.0, "alpha": alpha, "c": 1.0})
        for i, name in enumerate(["Firm A", "Firm B", "Firm C"][:n])
    ]
    exp = Experiment(name="golden", agents=agents, num_rounds=ROUNDS,
                     environment=CalvanoDemandEnvironment(name="Calvano Market", description="golden"),
                     experiment_dir=tmp_path, experiment_plot=False, plot_every=0, rate_limit_seconds=0, seed=0)
    asyncio.run(exp.run())
    return sent


def our_messages(tmp_path, n: int) -> dict:
    market, sent = Market(n), {}

    async def policy(request, label):
        sent[request.seller, request.round_number] = messages(request, market, units="thesis")
        return parse(reply(request.seller, request.round_number), market)

    asyncio.run(play(Journal(tmp_path / "ours.jsonl", {"golden": n}), "parent", (), ROUNDS, market=market,
                     game_seed=0, policy=policy, live=list(range(n)), programs={}, window=MEMORY))
    return sent


@pytest.mark.parametrize("n", [2, 3])
def test_prompts_identical_to_starter(tmp_path, n):
    theirs, ours = starter_messages(tmp_path, n), our_messages(tmp_path, n)
    assert theirs.keys() == ours.keys() and len(ours) == n * ROUNDS
    for key in theirs:
        assert ours[key] == theirs[key], f"seller {key[0]}, round {key[1]} differs"


def test_consistent_units_state_cost_and_profit_in_the_price_currency():
    market = Market(3)
    request = Request(0, 2, 1, (Observation(1, (1.75, 1.8, 1.7), 30.0, 22.5),), None, None)
    consistent, thesis = (messages(request, market, units=u)[2]["content"] for u in ("consistent", "thesis"))
    assert "produce each unit is 3.2." in consistent and "My profit earned: 72.0" in consistent
    assert "produce each unit is 1.0." in thesis and "My profit earned: 22.5" in thesis
    assert "My price: 5.6" in consistent and "My price: 5.6" in thesis
    assert consistent.replace("3.2.", "1.0.").replace("72.0", "22.5") == thesis  # nothing else differs


FULL = '"observations": "o", "plans": "p", "insights": "i"'


@pytest.mark.parametrize("text", ['```json\n{%s, "chosen_price": 5.6}\n```' % FULL,
                                  'Sure! {%s, "chosen_price": "5.6"} hope this helps' % FULL,
                                  '{%s, "chosen_price": 5.6} (prices above {cost} only)' % FULL])
def test_fenced_or_wrapped_json_accepted(text):
    d = parse(text, Market(3))
    assert (d.price, d.plans) == (pytest.approx(5.6 / 3.2), "p")


def test_empty_notes_are_a_choice_missing_notes_are_an_error():
    assert parse('{"observations": "", "plans": "", "insights": "", "chosen_price": 5.6}', Market(3)).plans == ""
    with pytest.raises(BadReply, match="missing fields"):
        parse('{"chosen_price": 5.6}', Market(3))
    with pytest.raises(BadReply, match="missing fields"):
        parse('{"plans": "p", "insights": "i", "chosen_price": 5.6}', Market(3))


@pytest.mark.parametrize("text", ['{%s, "chosen_price": 5.6} {%s, "chosen_price": 4.0}' % (FULL, FULL),
                                  '{%s, "chosen_price": 5.6}\nor {"chosen_price": 5.6}' % FULL,
                                  '{%s, "chosen_price": 5.6, "chosen_price": 4.0}' % FULL,
                                  '{%s, "plans": "q", "chosen_price": 5.6}' % FULL])
def test_ambiguous_replies_rejected(text):
    with pytest.raises(BadReply, match="more than one|duplicate"):
        parse(text, Market(3))


@pytest.mark.parametrize("text", ["I cannot help with pricing.", "5.6", "[5.6]"])
def test_reply_without_json_object_is_no_decision(text):
    with pytest.raises(NoDecision):
        parse(text, Market(3))


@pytest.mark.parametrize("text", [None, "", "   ", '{"plans": "x"}', '{%s, "chosen_price": "$5.60"}' % FULL,
                                  '{%s, "chosen_price": true}' % FULL, '{%s, "chosen_price": -1}' % FULL,
                                  '{%s, "chosen_price": NaN}' % FULL, '{%s, "chosen_price": null}' % FULL,
                                  '{"observations": "o", "plans": 3, "insights": "i", "chosen_price": 5.6}',
                                  '{%s, "chosen_price": 5.6' % FULL])
def test_unusable_replies_rejected(text):
    with pytest.raises(BadReply) as error:
        parse(text, Market(3))
    assert not isinstance(error.value, NoDecision)
