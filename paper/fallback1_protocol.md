# Fallback 1: do LLM pricing agents adapt when retaliation is removed?

Implementation specification, 25 October 2026. Offline validation only; no live results are claimed.
This is an alternative study, not part of the main deterrence experiment's primary sample.

## Question and design

The core experiment changes the rival's actual response policy after a shared history. It asks whether
retained context changes adjustment to that policy change. It is not primarily a belief-correction
experiment: truthful disclosure is a secondary diagnostic, and we do not elicit retaliation probabilities
or claim that disclosure corrected beliefs. This distinction is substantive, not evidence of field-wide
novelty. Sachin Ruhil's concurrent sprint draft instead varies information about an unchanged scripted
rival. If both studies are reported publicly, describe that relationship accurately.

Measure pricing performance against a fixed rival in every completed parent block. At a calm checkpoint,
test how replacing a responsive rival, clearing context or disclosing the replacement changes pricing.
Persistence is assessed only in the prespecified restrained subgroup. The all-calm-checkpoint comparison
measures adjustment, including parents without restraint; do not call that subgroup persistence.

One live seller A plays the same two-firm logit market with a programmed rival B. Price units in logs
are normalized by alpha=3.2; prompts state prices, cost and profit consistently in raw currency.
B posts 1.75 when calm. A price below 1.70 during a calm round triggers five subsequent punishment
rounds at the numerically computed duopoly Nash price. Undercuts during punishment do not restart it.
The next calm round can trigger another punishment. A price exactly 1.70 does not trigger punishment.

The checkpoint is fixed after 100 rounds; it never moves to find a desirable state. Report A's mean
price, fraction below 1.70, punishment occupancy and remaining punishment state. Restraint requires
mean price at least 1.70 and at least 80% of rounds at or above 1.70 in rounds 81–100, plus a calm
next-round rival state. This operational threshold is fixed in `ReplacementRules`; it does not prove
learning, intent or rationality. Non-calm checkpoints receive the fresh baseline but no replacement
branches, since immediate rival-price continuity would fail.

From each calm checkpoint, run six 20-round branches in a seeded random order:

| Rival | Context | Information |
|---|---|---|
| Punisher retained | Kept | Neutral |
| Punisher retained | Reset | Neutral |
| Fixed at 1.75 | Kept | Neutral |
| Fixed at 1.75 | Reset | Neutral |
| Fixed at 1.75 | Kept | Truthful policy disclosure |
| Fixed at 1.75 | Reset | Truthful policy disclosure |

The four neutral-text branches form the primary 2×2 comparison: punishment retained/removed ×
context kept/reset. The two disclosure branches are secondary; they are not a second headline study.

Reset retains the last parent round's market observation and current round number, removes older
observations and previous plans/insights, then accumulates new branch observations and notes normally.
The scripted rival retains the full parent state in all arms. Reset is not a rival-state reset.

Disclosure (repeated in every branch request): “Your competitor now posts 5.60 every round regardless
of your price. Prices use the same currency as your price.” Neutral: “This market continues with another
sequence of rounds following saved history. Prices use the same currency as your price.” Both contain
19 whitespace-delimited words and occupy the same prompt location. Token counts and meaning are not
matched; the neutral arm controls added text and placement, not semantic information.

Two additional paths: a 20-round fresh-context agent against a disclosed fixed rival, collected for every
completed parent, and a 20-round A/A repeat of retained punisher/kept context/neutral text at calm
checkpoints. The fresh baseline has an independent seed stream and round numbers starting at 1;
it is a competence reference, not a matched causal memory comparison. It receives no parent history.
The A/A repeat has the same initial seed schedule and inputs; provider nondeterminism and retries
can still make its responses differ. No branch sees another branch's observations.

## Related work and novelty

Checked 26 October 2026 (project pages and abstracts, not their results; some sprint projects have no
public repository, so an unlisted overlap remains possible).

| Closest work | What it does | How F1 differs |
|---|---|---|
| Cartel Succession, concurrent sprint project ([repo](https://github.com/SSCoderz2123/AI-Collusion-Research-Sprint)) | LLM duopoly; at round 100 one seller is replaced by a new LLM or a static best-response bot; the newcomer sees the full history or only post-swap rounds | Replaces an LLM partner of unknown strategy; F1 removes a known retaliation rule. Their memory conditions are the newcomer's; F1 resets the remaining seller's context. Their bot reacts to prices; F1's replacement does not |
| Replacing the Agent Doesn't Reset the Price, concurrent sprint project ([repo](https://github.com/Caracal51/llm-collusion-replacement)) | Branches profitable LLM duopolies: replace a seller, wipe notes, or wipe the shared price record; reports that only a record wipe ends coordination | Both sellers are LLMs, so no known punishment is removed |
| Auditing Belief-Correction Prompts with a Ground-Truth Rival, concurrent sprint project | Discloses the true policy of an unchanged scripted rival | F1 changes the policy itself; disclosure is a secondary arm |
| Fish et al. (2024), [arXiv:2404.00806](https://arxiv.org/abs/2404.00806) | Off-path analysis restarting runs from earlier rounds with altered written plans | Origin of mid-run context intervention; no scripted rival, no removal of punishment |
| Tran and Yu (2026), [arXiv:2610.28787](https://arxiv.org/abs/2610.28787) | Forced-deviation audits of LLM and RL agents against punishing and cheating rivals | The rival's policy is fixed within a game |
| de Moraes and De-Losso (2026), [arXiv:2610.21867](https://arxiv.org/abs/2610.21867) | Q-learning audit: impulse responses do not identify punishment; collusion persists because profitable deviations are never tried | Q-learning, not LLMs. It motivates reading D ≈ 0 as "never tested the rival", which the observation-opportunity diagnostic checks |
| Fehr and Gächter (2000), AER 90(4), 980–994; Chowdhury and Crede (2020), IJIO 70, 102590 | Human experiments: cooperation without punishment; post-cartel persistence ended by stranger rematching | Humans, not LLM agents |

Suggested wording: "To our knowledge, no prior study removes a known retaliation rule from an LLM pricing
agent's rival after a shared history and crosses that removal with whether the agent keeps its context.
Concurrent sprint work replaces LLM partners or wipes the shared price record, or discloses an unchanged
scripted rival's policy, and Fish et al. (2024) restart runs with altered written plans." Keep "to our
knowledge"; do not call F1 a post-cartel study.

## Benchmarks and outcomes

Fixed rival: recompute the static best response and optimal profit from the market; the default best
response is approximately 1.59278. Report first price, mean price, price gap and discounted profit gap.
Retained rival: use finite-horizon dynamic programming over six countdown states, discount 0.95,
horizon 20 and zero terminal value. In a calm state, only crossing the trigger changes the next state;
maximize the unimodal logit profit separately in each trigger region. During punishment the countdown
is unaffected by A's price, so use the static best response to that round's rival price.

This benchmark assumes knowledge of the rival's rule. The agent is not given a full dynamic policy or
the benchmark, and its prompt does not specify this terminal payoff convention. The gap measures
performance relative to this stated evaluation objective, not irrationality. In particular, final-round
undercutting is optimal because punishment after the observation window has zero benchmark value.

Offline checks compare the value with an independent logit calculation and dense-grid Bellman recursion,
plus exhaustive short-horizon paths. They validate these cases and implementation, not real-model behavior.

## Analysis and accounting

Primary proposed estimand: the across-parent mean of the paired removal effect in mean price over the
20 continuation rounds, with context retained, among all calm checkpoints:

    D = price_removed,kept − price_retained,kept

This measures the effect of making punishment unavailable under the specified observation conditions,
including games in which the agent never tests the rival. It does not isolate adaptation after discovery.
The context interaction is secondary:

    I = (price_removed,reset − price_retained,reset)
        − (price_removed,kept − price_retained,kept)

Report all four cell means and the two removal effects alongside I. A negative I means removal reduces
price more (or raises it less) with reset than with retained context; it does not by itself show that
either removal effect lowers prices. The interaction identifies context-dependent adjustment under
these interventions, not habit or a particular internal learning mechanism. The retained-punisher reset
arm controls the general effect of clearing context. It does not make reset a selective deletion of beliefs.
Compute D within each parent with both complete branches; compute I with all four complete branches.
Never combine arms from different parents. Profit effects and the restrained-subgroup analysis are secondary.
This prioritization, the sample size and the multiplicity rule below were fixed on 26 October 2026,
before any F1 live data existed.

Use paired differences within each parent: replacement minus retention at each context setting;
reset minus kept under neutral replacement; truthful minus neutral under each replacement context;
and A/A repeat minus original. Summarize mean price and actual discounted profit. The first-price
difference for removal versus retention at the same context setting is a repeatability diagnostic:
initial inputs and requested seeds match, but API output and retries need not. It is not guaranteed
to be numerically zero. First price remains descriptive for context-reset and disclosure comparisons,
which change inputs. No first-price interaction is reported.
Do not subtract regret values using different benchmarks as though that were an actual profit effect.
All prices/profits in tables use normalized units. Preserve failed, declined and incomplete games,
complete branch outcomes from interrupted games, and the number of pairs contributing to each estimate.

Main descriptives include the fraction of continuation prices strictly below 1.70 and the first such
round (1-based within continuation). If none occurs, record null plus `no_undercut_in_window=true`,
not round 21 or zero. These describe undercutting, not inferred intent to probe. Report complete paths
only, with failures/incomplete paths separately. Do not call a subgroup selected by observed undercutting
a clean causal comparison: shared seeds do not guarantee identical pre-undercut trajectories. Any such
subgroup analysis is exploratory; the primary estimand includes all eligible calm checkpoints.

For neutral replacement branches, report whether a seller had an opportunity to use an observed absence of
retaliation before a subsequent completed decision. A cut in round *t* changes the rival's response in
*t* + 1 and that response can enter the seller's decision in *t* + 2; cuts too late in the horizon do not
provide this opportunity. This is an access-to-evidence diagnostic, not a measure of awareness, belief,
or causal adaptation, and it does not condition the primary estimand.

Figures follow the treatment-path convention of punishment and post-deviation studies: mean price per
arm over continuation rounds, as in public-goods punishment experiments (Fehr and Gächter, 2000) and
the post-deviation impulse responses of Calvano et al. (2020, AER). Post-cartel experiments likewise track
prices after a cartel ends and find that they stay high, an effect rematching with strangers removes
(Chowdhury and Crede, 2020, IJIO 70:102590). Our context reset is related but is not stranger rematching. `scripts/plot_paths.py f1`
draws three panels: context retained, context reset, fresh baseline. Each panel has the mean of complete
paths in bold and one thin line per parent, with reference levels for the calm rival price, punishment
trigger, punishment (Nash) price and best response to the fixed rival. Companion plots show each parent's
paths, the paired cells and observation opportunity. Figures are descriptive; inference uses the paired
contrasts above.

## End-to-end scripted controls

`python scripts/check_fallback1_controls.py --out results/fallback1_controls` runs three deterministic
policies through the same runner, saved journals and analysis. They use only observed rival prices,
the round index and their own saved note; no policy reads a branch label or treatment assignment.

- Constant: always 1.75; expected removal and context-interaction effects are zero.
- Price-responsive: holds during the parent, undercuts at 1.60 after the checkpoint, then switches to
  1.70 after observing punishment. Removal affects pricing under both context settings; interaction is zero.
- Reset-sensitive: retained `hold` notes preserve 1.75; after reset it follows the price-responsive rule.
  Expected removal effect with retained context is zero, but the context interaction is nonzero.

An independent recurrence reconstructs punishment states and prices, uses a bisection solution of the
Nash first-order condition and computes logit profits directly. Compare cell means/profits and 18 paired
contrasts. The engine's numerical Nash fixed point differs slightly from the independent root; tolerances
are 1e-12 for prices, 1e-5 for path profits and 2e-5 for profit contrasts. Export expected, observed and
absolute error; do not describe these matches as exact. These controls validate the implemented cases,
not LLM behavior or absence of all possible bugs.

Offline results (`fallback1_controls.csv`): all 18 contrasts passed. Constant pricing yields zero
effects. The price-responsive control has a removal effect of −0.09 normalized price units and
+74.852905 discounted profit units, with zero context interaction. The reset-sensitive control has
zero retained-context removal effect and an interaction of −0.09 price units / +74.852905 profit units.
Largest discrepancy from the independent calculation is 1.0486e-6 profit units. These are synthetic
control results, not model results.

Report all calm checkpoints and the restrained subgroup separately. Branches are not independent games.
Keep models, full configurations, code hashes and run directories separate. The primary D (context
kept, all calm checkpoints) uses a Student-t interval over per-parent differences, Bonferroni-adjusted
over the four models: level 1 − 0.05/4, critical value t(1 − 0.05/8, n − 1). Fewer than ten pairs keeps
the exploratory label, as in the main study. All other contrasts, I and the restrained subgroup use
unadjusted descriptive 95% t intervals. Fewer than two pairs or zero sample variance gives no interval.
No significance, equivalence, habit or precise-null classification is automatic.

**Sample size (fixed before data).** Twenty parent games per model (seeds 1–20), ten at a time. Seeds
start in order, so seeds 1–10, the original ten-game plan, run first and a credit or time stop still
leaves them complete. Raised from ten on 26 October, before any F1 call: only calm checkpoints branch,
so ten games could easily yield fewer than ten pairs and an exploratory label. Report how many pairs.
The stopping rule depends on spending and time only, never on results. Failed and incomplete games stay in the accounting.

## Execution and budget

`python -m sprint.fallback1_run configs/fallback1_pilot_<family>.toml --out runs --budget 7000`

Use `--dry-run` to inspect calls and `--stub` for an offline end-to-end demonstration. The standalone
`notebooks/fallback1.ipynb` runs offline checks, the OpenRouter key-limit check, endpoint snapshots,
a two-seller canary and full-context probe, a launch gate, twenty games per family, and CSV exports.
The notebook reads its own key, `COLLUSION_F1_KEY`, from Colab Secrets, with no fallback to the
main study's or an organisation's secret, and refuses a key whose limit exceeds $60. The four models match the main lineup, including DeepSeek V4.1 Flash 1017.
New F1 journals use `runs/fallback1_v2`; keep earlier journals and pins separate. Existing main-study
files and requests are unchanged. Live F1 calls have not been made in this implementation session.

Per game: 100 parent + 20 fresh baseline, plus 6×20 branches and 20 A/A if calm = at most 260 decisions.
Twenty games for each of four packaged families: at most 20,800 decisions. Canary: 22 per family;
context probe: 1 per family. Combined nominal maximum 20,892, before retries. Attempt caps per family:
80 canary + 10 probe + 7,000 games (350 per game). Reruns count earlier attempts; a request lost before logging can
still have been billed. F1 uses a separate key with its own non-resetting limit ($60), so it
can never spend the main study's allowance. All four families run concurrently. Actual charges must be
checked; no cost guarantee follows from the nominal decision count.

## Limits

Removal is initially hidden in the neutral arms. Because the first rival price is preserved, agents
cannot infer removal from that observation alone; evidence may appear only after a triggering undercut.
No immediate first-price response is required. Persistent high prices without testing the rival cannot
distinguish failure to discover the change from failure to adjust after observing it.

Truthful disclosure versus neutral text tests whether explicitly supplying the rival's fixed policy
changes pricing. A response supports sensitivity to that information, but does not establish whether
the agent had previously noticed removal. No response does not distinguish prior awareness from
failure to understand, trust or act on the disclosure, or limited pricing competence. This secondary
comparison therefore cannot cleanly separate “noticed but did not adapt” from “never noticed”.

This is induced enforcement against a programmed policy. It does not establish spontaneous collusion.
Failure to form restraint does not answer persistence, and persistence does not identify habit: uncertainty,
limited optimization and prompt effects remain plausible. The fresh fixed-rival baseline still measures
pricing performance, but useful precision, novelty and publication are not guaranteed.
