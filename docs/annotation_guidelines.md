# Annotation Guidelines — Geopolitical Event Categories

Version **1.1**. These are the rules a human reviewer follows when labelling the
review sheets, and they are also the rules the keyword baseline approximates. Read
them before labelling, and update this file (with a version bump and a roadmap entry)
if a rule changes — the evaluation report records the taxonomy version it was run against.

> **Rule zero:** our categories are *proposed project labels*. They do not exist in the
> news dataset. Classify what the headline describes, never what you assume the market
> did. A keyword match is not evidence that an event happened, and a correct label is not
> a prediction that prices will move.

---

## 1. The categories

Assign exactly **one** category per headline, from this list:

| Category | Definition | Typical signals |
|---|---|---|
| `armed_conflict` | Kinetic hostilities between organised forces: war, strikes, invasions, troop movements, ceasefires, casualties. | war, airstrike, missile, troops, invasion, shelling, ceasefire, hostages, killed |
| `sanctions` | Coercive economic measures targeting a state, entity, or person: sanctions, embargoes, asset freezes, export controls used as punishment. | sanctions, embargo, blacklist, asset freeze, OFAC, punitive measures |
| `trade_restriction` | Barriers to, or bargains over, cross-border trade: tariffs, import/export bans, quotas, trade deals and disputes. | tariff, trade war, import ban, quota, WTO, customs duty, trade deal |
| `energy_or_supply_disruption` | Physical disruption to energy or goods supply: shortages, pipelines, OPEC decisions, port or shipping problems, blackouts. | shortage, pipeline, OPEC, refinery, blackout, supply chain, port congestion |
| `diplomatic_development` | Statecraft without coercion: summits, talks, treaties, ambassadors, alliances, prisoner exchanges. | summit, talks, negotiations, treaty, ambassador, G7, security council |
| `other_or_uncertain` | Everything else — including domestic politics — or anything you genuinely cannot place. | elections, courts, campaigns, crime, celebrity, health policy |

`other_or_uncertain` is a **first-class result**, not a failure. Forcing a domestic
politics story into `diplomatic_development` because it mentions a foreign country
produces worse data than honestly choosing `other_or_uncertain`.

---

## 2. Tie-break rules (when several categories seem to fit)

Apply in order:

1. **Coercion outranks talk.** A headline describing both a threat of force and
   negotiations is `armed_conflict`.
2. **Specific policy instrument outranks general diplomacy.** "Sanctions imposed after
   summit collapse" → `sanctions`.
3. **Policy outranks supply effect.** "Tariffs on oil imports" → `trade_restriction`
   (the policy is the subject), whereas "oil supply cut after pipeline damage" →
   `energy_or_supply_disruption`.
4. **If still tied, choose the category of the event the headline leads with** — usually
   the subject of the main clause.
5. **If you still cannot decide after 20 seconds, use `other_or_uncertain` and write the
   alternatives in the `notes` column.** Ambiguity is data; note it, don't hide it.

The priority order the baseline uses is: `armed_conflict` → `sanctions` →
`trade_restriction` → `energy_or_supply_disruption` → `diplomatic_development` →
`other_or_uncertain`. Rules 1–3 above are what that order encodes.

### 2.1 Context phrases that veto a match (added in version 1.1)

High-frequency tokens fire on metaphors and unrelated senses. After the first review pass, the
baseline learned to **suppress** — not delete — the following contexts, and it records the
suppression in the `excluded_evidence` column so a reviewer can always see it:

| Vetoed token | Suppressing phrase | Why |
|---|---|---|
| `war` | `trade war`, `price war`, `culture war`, `war of words`, `war on drugs`, `war on poverty`, `war on coal`, `war on women`, `non-war`, `war pay` | Not kinetic conflict; a trade story must stay `trade_restriction`. |
| `invasion` | `invasion of privacy` | Legal charge, not armed conflict. |
| `missile` | `missile alert` | A false-alarm civil-defence story is not a strike. |

Keeping the bare token (rather than deleting it) preserves genuine conflict headlines such as
"Ukraine war continues", which the baseline still labels `armed_conflict`. Suppression is a
precision fix with a small recall cost — exactly the kind of trade-off the review sample is meant
to measure.

---

## 3. Hard cases — apply these explicitly

| Case | Rule | Rationale |
|---|---|---|
| **Threats and plans** ("Trump threatens tariffs", "EU may sanction X") | Label the **event type** mentioned: `trade_restriction`, `sanctions`. Do not invent a "threat" category. | The taxonomy classifies what kind of event is being discussed; severity and follow-through are separate questions the project explicitly does not assume. |
| **Denials and absence** ("No new sanctions", "White House denies tariff plan") | Label the event type mentioned — `sanctions`, `trade_restriction` — and note the denial in `notes`. | The headline is *about* that policy instrument. The baseline flags these rows (`evidence_negated`) so the review can see them. |
| **Relief and reversal** ("US lifts sanctions on X", "tariffs lifted") | Label `sanctions` / `trade_restriction`; note it is a relief. | Our taxonomy has no "relief" class; adding one would be scope growth. The relief direction is captured in the notes and in the sentiment score, not the category. |
| **Anniversaries and retrospectives** ("Two years since the war began") | Label by the event type (`armed_conflict`) and note it is historical. | The dataset has no "new event" signal, so treat this as a documented limitation rather than guessing freshness. |
| **Domestic politics** (elections, courts, budgets, party disputes) | `other_or_uncertain`. | Not geopolitical, even when it mentions trade or foreign policy in passing. |
| **Sports, entertainment, lifestyle, health** | `other_or_uncertain`. | Outside project scope. |
| **Multiple countries, one policy** ("US and China agree trade deal") | `trade_restriction`. | Trade bargains are in scope; record both actors in `notes` if useful. |
| **Crime or terrorism that is not state conflict** ("Suspect charged in bombing") | Usually `other_or_uncertain`; use `armed_conflict` only when organised armed conflict is described. | Avoids inflating `armed_conflict` with unrelated violence. |
| **Purely economic news** ("Inflation falls to 3%") | `other_or_uncertain`. | Macro data is not a geopolitical event. |
| **Empty or unusable headline** | Leave `event_category` **blank** and say why in `notes`. | A blank label is honest; a guessed one is not. |

---

## 4. What *not* to do

- Do not label on the basis of assumed market impact. Impact is a hypothesis we test in
  Phase 4, not a labelling criterion.
- Do not use severity as a category. "A small border clash" and "a world war" are both
  `armed_conflict`.
- Do not create new categories ad hoc. Propose them in the roadmap instead.
- Do not leave the label blank simply because the headline is hard — only leave it blank
  when the text is unusable (empty, truncated, non-English).

---

## 5. Sentiment is a separate task

Sentiment is the **tone of the text**, not severity and not market direction:

- `positive` — approving, optimistic, resolving tone ("deal reached", "ceasefire holds").
- `neutral` — factual, procedural tone ("officials met on Tuesday").
- `negative` — alarming, conflictual, or loss-bearing tone ("stocks plunge", "sanctions bite").

A negative-sounding headline about a conflict is not automatically bearish for the market,
and our event labels never imply sentiment. Phase 2 computes sentiment with a small
transparent lexicon (config `nlp.sentiment_mode: lexicon`) and stores the matched terms so
any score can be audited. A pretrained financial model (FinBERT) is a planned optional
upgrade — it needs the `transformers`/`torch` stack and model weights, so it is deliberately
not a default; asking for it raises a clear error rather than silently falling back.

---

## 6. The labelling workflow

```bash
# 1. Classify (writes data/processed/news_classified_*.csv and the Phase 2 report)
python -m stock_project.nlp.classify_news --config configs/config.example.yaml

# 2. Build the sheets (labels left empty; existing labels are never overwritten)
python -m stock_project.nlp.annotate --config configs/config.example.yaml

# 3. Human step: label the sheet. The labeller serves a local page with one headline
#    at a time (keys 1-6 pick a category, s skips, u undoes); labels are written to the
#    sheet immediately, next to an append-only audit log, and skips stay blank.
python -m stock_project.nlp.labeller --config configs/config.example.yaml --reviewer "your name"
#    Allowed values: armed_conflict, sanctions, trade_restriction,
#    energy_or_supply_disruption, diplomatic_development, other_or_uncertain
#    (--check prints progress; --sheet data/annotations/news_review_focus.csv labels the
#     stratified sheet instead, for qualitative error review)

# 4. Evaluate (writes metrics + a misclassification listing)
python -m stock_project.nlp.evaluate_classifier --config configs/config.example.yaml
```

### The two sheets

| Sheet | Sampling | Use it for | Do **not** use it for |
|---|---|---|---|
| `news_review_sample.csv` | seeded **random** sample | the reported precision/recall/F1/balanced accuracy | inspecting rare failure modes (too few rows) |
| `news_review_focus.csv` | **stratified**: flagged evidence + small predicted classes | reading actual error cases | headline metrics — its metrics are biased by construction |

Label the random sheet **blind**: the sheets omit the baseline's predictions by default so
your judgement is not anchored, and the labeller never sends them to the browser either —
pass `--show-predictions` to `annotate` only for a second, explicitly non-blind pass.

Practical guidance for a two-person team: 150–200 randomly sampled headlines is roughly one
sitting per person; split the sheet, each review your half, and record your name in the
`reviewer` column so agreement between reviewers can be checked later.

---

## 7. Known limitations to state in any write-up

1. **Headline-scope.** Classification uses the headline (optionally plus the source's short
   description); the dataset has no article body, so events described only in the body are
   missed.
2. **General-news source.** The corpus is general-interest political news, not a curated
   geopolitical feed; a large share of `other_or_uncertain` is expected and informative.
3. **Day-level timestamps.** See `docs/dataset_notes.md`: features must use `available_from`,
   never `published_at`.
4. **Keyword opacity to context.** The baseline cannot tell a new event from an anniversary,
   a threat from an action, or a sanctions *relief* from a sanctions *imposition* beyond the
   flags it records. Metaphor remains a known weakness: the veto list in §2.1 covers the cases
   the review pass actually found, and new ones will appear.
5. **Small samples.** Per-class metrics from a few dozen labels are noisy; report support
   next to every metric, and never present this baseline's numbers as predictive skill.
