# Research process

The project progressed from a standalone strategy to a pod research platform
and finally a constrained fixed-blend portfolio. The final presentation
describes both the exploration and the final book; their experiment counts
refer to different stages.

## From hypothesis to signal

The working process was:

1. Search dated literature and available data, with LLM assistance.
2. State an economic mechanism before testing it.
3. Turn the mechanism into a measurable formation-time signal.
4. Evaluate it only on evidence preceding the test year.
5. Admit or reject it through a recorded evidence gate.
6. Apply fixed portfolio construction, risk controls and accounting checks.

The admission gate included a spread t-statistic of at least 2, a shuffled
control, a correlation limit and a minimum score. The final discovery
implementation also requires a score of at least 0.60 and limits admissions
to at most two per desk. Promising IC alone did not justify admission when
the top-minus-bottom return spread was weak.

## Walk-forward testing

The characteristics panel starts in 2015. For test year Y, models train
through Y-3, validate on Y-2 and Y-1, and then predict year Y. Examples:

| Test period | Training | Validation |
| :--- | :--- | :--- |
| 2021 | 2015-2018 | 2019-2020 |
| 2026, through August | 2015-2023 | 2024-2025 |

Features available at month t forecast the next holding month. Filing-based
availability uses the next trading day where implemented. The final saved
construction replay reused frozen forecasts; it did not refit the stock or
graph models. Checkpoints record that distinction.

## Two agent experiments

| Experiment | Reported outcome | Relevance to the final book |
| :--- | :--- | :--- |
| Earlier recovered research portfolio, presentation slide 4 | 321 candidate variants, 12 admitted | Records the broader exploration; not the final v3 portfolio |
| Final v3 discovery, slides 7 and 12 | 144 proposals over six test years, 0 admitted | Discovery did not change the final historical fixed blend |

The earlier portfolio explored ideas such as cash-value quality, liquid
reversal, earnings surprise/weak response, profitable capex, foot traffic
relative to sales, profitability, asset growth, momentum, operating leverage
and short-sale volume. The presentation's per-signal returns are averages
over their active spans and should not be read as final-book contributions.

The final v3 research desk's best of 51 valid pre-2021 proposals achieved a
spread t-statistic of 1.05, below the admission threshold. This is useful
evidence that the gate rejected weak suggestions, but it does not establish
that agents added performance. With no admissions, the recorded agent-enabled
and fixed-rule histories held the same book.

The deck includes a model/cutoff schedule for recovered research. A stated
training cutoff is not proof of contemporaneous model availability or an
independent leakage audit. The public documentation does not claim such proof.

## Data exploration

Sources tested or explored included foot traffic, card spending and user
breadth, website visits, app downloads, SEC 8-K/Form 4/13F filings, FINRA
short-sale volume, prediction markets, job openings and trade claims.
**Exploration is not use in the final portfolio.** The final deck discloses
I/B/E/S revisions for one alpha sleeve and no Carbon Arc predictive asset in
the final book. FRED is for reporting; saved Arcana betas inform sizing.

Carbon helpers are preserved under [tools/](../tools/README.md), while vendor
payloads stay local. The earlier platform remains under
[final_v2/](../final_v2/README.md), with its single `run.py` entrypoint.

## What we learned

Fundamentals offset the other pods: the presentation reports monthly pod
correlations of approximately -0.46 to -0.54 against the other three. Events
and dynamics were more correlated (+0.75). These are descriptive statistics
from the presented run, rather than a guarantee of future diversification.

Longs were the main return source; shorts provided crisis protection but
introduced thematic and squeeze risk during rallies. Beta constraints did
not remove those style risks. The final discussion prioritizes crowded-short
caps, daily accounting reconciliation, certified volatility, regime tests and
twice-modeled-cost sensitivity. See [the results discussion](results.md).
