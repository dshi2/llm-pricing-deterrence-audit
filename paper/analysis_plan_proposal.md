# Main-study analysis protocol

Status: implemented on 26 October 2026, before any pilot outcome was seen. Pilot (seeds 1–4) and
continuation (seeds 5..N) are two stages of one collection: same run folder, same pinned collection code
(code hash 46e1ff33…, unchanged since the pilot pin 811d4f3), same settings. Their games pool in one
analysis. N is fixed by an automatic rule from cost and time (below), never from S, J, M or S − J,
and is frozen in `continue_plan.json` before any continuation call.

## Primary questions and population

For each model/provider/prompt configuration, among eligible, completed games under the fixed selection
rule, estimate the mean prescribed-cut gain with a randomly selected single rival enabled,
S = (G10 + G01)/2, with both rivals enabled, J = G11, and the smaller realised single-rival gain,
M = min(G10, G01), and the paired difference S − J = R2 − R1: how much more two live rivals reduce
the cut gain than one randomly selected live rival. The game is the independent unit; retain all
branches together. Report all started, ineligible, incomplete and failed games alongside these conditional
estimates. Complete-case inference can be biased by outcome-dependent failures; do not generalise it to
all started games without additional justification.

For S and J, an interval wholly below zero supports "the prescribed cut is unprofitable on average
in this response condition"; wholly above zero supports "the cut remains profitable on average";
otherwise its sign is unresolved. These statements concern the fixed ten-round cut and 20-round payoff
window, not all deviations, every game, intent, or equilibrium collusion.

Do not label S > 0 and J < 0 as joint-only deterrence. For example, G10 = -10, G01 = 30 and G11 = -20
give S = 10 although B alone already makes the cut unprofitable. Similarly, J > 0 does not rule out
G10 < 0: two responses need not deter more than one.

Report G10 and G01 separately, plus per-game sign patterns. A realised joint-only pattern requires
G10 > 0, G01 > 0 and G11 < 0 in the same game. Label these observed branch outcomes, not established
expected-payoff classifications of that game: there is only one stochastic continuation per cell.

For M, use only a lower interval bound above zero as directional evidence that even the smaller
realised single-rival gain is positive on average. If the cell estimators are conditionally unbiased
given the saved state, reference and selected roles, then E[min(G10,G01) | state] is no greater than
min(E[G10 | state], E[G01 | state]). Independence between the cells is not required for this inequality.
Under that assumption, a positive mean M is conservative evidence for a positive average of the
smaller state-conditional expected gains. A negative mean M can reflect sampling variation within
cells and does not establish that a single rival reliably deters. This inequality does not guarantee
coverage of the t interval, nor remove bias from failed games or outcome-dependent selection.

M > 0 and J < 0 on average still do not identify how often both responses are necessary in the same
state. For example, two equally weighted deterministic games (G10,G01,G11) = (1,1,1) and (-0.1,100,-100)
have mean M = 0.45 and mean J = -49.5, but neither has a joint-only pattern. Report the two mean
statements explicitly; do not convert them into a joint-only prevalence or universal mechanism claim.

For S − J, an interval wholly above zero supports "two live rivals reduce the cut gain more than one on
average"; wholly below zero, "less than one". It is computed within each game, so the shared G00 and
state cancel. It does not identify strategic complements, intent or a joint-only mechanism.

## Intervals and multiplicity

Use Student-t intervals for the four primary means, calculated from game-level values. Control the
sixteen planned primary intervals (four outcomes times four configurations) with Bonferroni: each
interval has level 1 - 0.05/16 = 99.6875%, with critical value t(1 - 0.05/32, n - 1). Retain this
allocation if fewer families finish or continue; do not reassign it after inspecting results. This gives nominal simultaneous coverage
only to the extent that the marginal t intervals are valid.
Use study-wide correction because the report will present results from all four configurations and
may highlight any of them. Separate family tables do not by themselves remove this selection risk.

Independent normal observations give the exact t result. Small, skewed or outlier-dominated samples
can invalidate its approximation. Plot every game; show leave-one-game-out point estimates as a
stability diagnostic, without deleting observations or switching the primary method after inspection.
Fewer than ten completed eligible pooled games is explicitly exploratory; ten is not a validity
certificate. No meaningful uncertainty estimate is available from fewer than two games or from a
zero sample variance treated as proof of population certainty.

Keep 95% percentile-bootstrap intervals as a labelled secondary sensitivity analysis, not an independent
confirmation. Do not claim a universal percentage by which bootstrap intervals are too narrow.

## Secondary effects

Report R1, R2, nonadditivity, A/A differences, response diagnostics and observed sign frequencies
descriptively. If reporting the 50% scale, calculate
D1 = R1 - 0.5*G00 and D2 = R2 - 0.5*G00 within
each game before summarising. This propagates the paired variation in G00 without dividing by a small
gain. D > 0 concerns removal of more than half of the mean unopposed gain; it does not establish that
the remaining cut gain is negative. No automatic equivalence or precise-null classification is planned.

## Late-parent price diagnostics

For every available main-game journal, report the fixed parent window 100–150 (51 rounds),
separately for each seller and the market mean, in normalised price units. Export the linear
slope, fitted change over 50 round intervals, mean price in 126–150 minus mean price in
100–124, sample standard deviation and mean absolute adjacent-round price change. A missing
or duplicated round makes the window incomplete; do not silently shorten it or report zeros.
Include ineligible games and games that failed after finishing this window.

These are descriptive diagnostics with no round-level significance tests, convergence cutoff,
eligibility gate or adaptive change to the parent horizon. Seller-level results expose offsetting
trends; variability exposes oscillations that a flat fitted slope can hide. A flat window does
not establish equilibrium, convergence or that 150 rounds were sufficient. Persistent notes
can retain information from before the explicit 100-round history window. This saved-data
addition does not alter collection or the primary estimands.

## Sampling and budget: one protocol, two stages, pooled

The pilot has four full-size parent games per family (seeds 1–4). The continuation reuses each family's
pilot config unchanged except for the seed list (1..N) and concurrency, writes into the same run folder,
and resumes past the finished pilot games. The launcher rejects any pilot journal whose saved settings or
collection-code hash differ from the continuation config. Pooling is therefore pooling of games collected
under one protocol, not a post-hoc merge of different designs.

Continuation rule, fixed in code (`sprint/continuation.py`) before any outcome was seen:

1. Wait until every pilot has finished and its folder has been quiet for 10 minutes.
2. A family continues only if its launch checks passed and all pilot games finished as eligible or
   ineligible. A family with a failed pilot game or failed launch checks stays pilot-only and is reported.
3. Measure per family the mean recorded cost per attempt, the game time (slowest pilot pace per reply round
   × the 310 reply rounds of a full eligible game, so short ineligible pilot games cannot understate it) and
   the share of pilot attempts answered HTTP 429. Concurrency c is 8 games, or the pilot's 4 if more than 1% of
   attempts were rate-limited (429 retries also use up attempts).
4. Time cap: 4 + c × floor(hours to the collection cutoff, 15:00 IST 26 October (moved from 12:30, then 14:00, before any pilot data was seen), / (1.5 × game time)),
   at most 24 seeds. Cost per new game: mean attempt cost × 875 attempts.
5. Add one game per family in turns, cheapest family first, while the total stays within 90% of the
   key's remaining allowance, read after the pilots finish, and each family's time cap.
6. Freeze seeds, attempt caps (875 × N per family, pilot attempts included) and configs; a rerun reuses
   them and never re-sizes. Then rehearse the full-run report on the pilot's real data (N is already fixed),
   so report failures surface before the final tables; it never blocks collection.

Eligibility status, S, J, M and S − J play no part in N. Do not stop or extend collection because an
estimate looks favourable. Credit or deadline interruption leaves games incomplete; they stay in the
accounting as incomplete or not started, never replaced. The count is parent games, not eligible games.
The shared non-resetting $100 key cap covers the main study. The cost projection uses recorded pilot
costs; it is not a bound, because future tokens, prices and unrecorded billing can differ.

The pilot's own tables are a quick look, labelled exploratory. The primary analysis is the pooled one,
labelled `pooled` per family; fewer than ten eligible games keeps the exploratory label. Pooled output
has accounting, descriptive intervals, the S/J/M/S − J analysis, per-game outcomes and figures. The
50%-of-unopposed-gain contrast remains secondary. No prespecified equivalence test is claimed.
Parallel runtimes must never write the same game; the quiet-folder wait is a guard, not a lock.

## Sources and implementation status

- [NIST mean confidence limits](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)
- [NIST normality and small-sample limitations](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/conflimi.htm)
- [NIST Bonferroni intervals](https://itl.nist.gov/div898/handbook/prc/section4/prc473.htm)

Implemented in `sprint/study.py` and frozen by `sprint/continuation.py`. Tests cover both
misleading-average examples, the one-direction interpretation of M, the paired S − J interval, zero
variance, the closed-form 16-interval critical value, full protocol parity with the pilot, waiting for
running pilots, pilot-only exclusions, and refusal to re-size or change a frozen plan. Existing collection modules are not modified. Live execution remains unverified.
