# XAI framework for credit-card attrition ANNs

An end-to-end Python framework that trains a neural network to predict which cardholders
will close their account, then shows **how it thinks**, **why it flagged each customer**, and
**whether those explanations can be trusted**. Each part is explained with the
**Feynman technique**: name the idea, explain it simply, say where the simple version breaks,
and then give the precise version.

```bash
pip install -r requirements.txt
python -m xai_attrition                          # simulated data, about 30 s
python -m xai_attrition --csv BankChurners.csv   # the real Kaggle dataset
python -m xai_attrition --fast                   # quick smoke run, about 10 s
python -m pytest tests                           # the explainers' math guarantees
```

Output: `outputs/report.md` (the full illustrated report), `outputs/figures/*.png`, and
`outputs/summary.json`.

---

## The problem in one paragraph (Feynman step 1 + 2)

A bank wants to know which customers are about to leave so it can act early. A neural network
is very good at spotting them. The trouble is that the network's "reasons" are spread across
thousands of numbers (weights), and none of them means anything on its own. If the bank calls
a customer, cuts a fee or reports to a regulator, it has to be able to say **why**. XAI asks
the network questions from the outside, and looks at its sensitivities from the inside, until
its behaviour can be told as a plain story.

## The framework at a glance

```
              ┌───────────────────────── GLOBAL: how does it think? ───────────────────────┐
              │ Permutation importance │ SHAP summary │ PDP / ICE curves │ Surrogate tree   │
 customers ─► │                                                                             │
   ANN        ├───────────────────────── LOCAL: why THIS customer? ────────────────────────┤
 (black box)  │ SHAP waterfall │ LIME │ Integrated Gradients │ Counterfactual "what-if"    │
              ├───────────────────────── TRUST: believe the explanations? ─────────────────┤
              │ Deletion test │ Agreement between methods │ Additivity checks │ Ground truth │
              └──────────────────────────────── Narrator: plain-language story ────────────┘
```

| Question | Method | One-line Feynman version |
|---|---|---|
| What does the model rely on? | Permutation importance | Shuffle a column; if accuracy drops, it mattered. |
| Which way does each fact push, and how far? | Shapley values (SHAP) | Split the prediction fairly among the facts, like splitting a prize across a team. |
| What shape is the learned rule? | PDP / ICE | Turn one dial and watch the risk change. |
| Can a human follow the logic? | Surrogate tree | A short flow-chart that copies the ANN, scored by how often it agrees. |
| Why this customer, locally? | LIME | The Earth is round, but your street looks flat. Fit a straight line near this one customer. |
| What does the network itself respond to? | Integrated Gradients | Walk from "typical customer" to "this customer" and add up the slope at each step. |
| What would change the outcome? | Counterfactual | "You would have stayed if you had made 20 more transactions." |
| Are the explanations honest? | Trust checks | Erase the facts SHAP blames. If its story is true, the risk should collapse. |

Every explainer class has a `FEYNMAN` card (`concept / simple / analogy / gaps / refined`),
and the report prints that card above the matching chart.

## Where the simple story breaks (Feynman step 3)

These are the gaps we went back and studied. The framework deals with each one in code:

1. **Explanations describe the model, not the world.** A result like "few transactions raises
   risk" tells you what the ANN uses. It does not prove that pushing transactions will keep
   customers. The report says this plainly.
2. **Correlated features split credit between them.** `Avg_Open_To_Buy` is exactly
   `Credit_Limit − Revolving_Bal`, so the model never sees it. Without that, the
   explanations would split credit between two copies of the same fact.
3. **Perturbations create unrealistic customers.** Counterfactuals only change features the
   bank can influence (*actionable* features), and they recompute derived fields such as
   utilisation after each change.
4. **Methods have settings that change the answer** (baseline, kernel width, sample size).
   That is why there is a trust layer: you can see whether the methods agree, whether
   LIME's local fit holds (R²), and whether SHAP passes the deletion test.
5. **How do we know an explanation is correct?** Run without a CSV, the framework simulates
   BankChurners-style data with **causes we planted ourselves**, so it can check whether
   the XAI finds them again.

## The precise version (Feynman step 4)

| Method | Guarantee checked in `tests/` |
|---|---|
| Shapley (Monte-Carlo permutations with antithetic sampling) | Efficiency: Σφⱼ = f(x) − E[f] exactly |
| Integrated Gradients (back-propagation written by hand, midpoint Riemann sum) | Completeness: ΣIGⱼ = F(x) − F(baseline), in log-odds |
| Back-propagation | Matches finite differences; the logit matches sklearn's probability |
| Counterfactual | Never changes non-actionable features, and always lowers the risk |
| Surrogate tree | Fidelity to the ANN is above 75% |

## Results from the default run (simulated data, seed 7)

These numbers come from `outputs/report.md`:

- ANN: ROC-AUC **0.90**, PR-AUC **0.68** (base attrition rate about 16%).
- Surrogate tree (depth 4) matches the ANN on **88%** of held-out customers.
- Faithfulness: for **100%** of the top-25 riskiest customers, erasing the facts SHAP blames
  drops their risk faster than erasing facts in random order.
- Methods agree on **81–86%** of each customer's top-5 drivers.
- Ground truth: SHAP's top 7 contain **6 of the 7 planted drivers**.

## Package layout

```
xai_attrition/
  data.py            load the real CSV or simulate BankChurners; preprocessing; plain-English names
  model.py           the ANN (sklearn MLP) + forward/backward pass written by hand
  explainers/
    base.py          Feynman cards, LocalExplanation, the "mix rows" engine
    shapley.py       Monte-Carlo Shapley values
    lime.py          LIME with weighted ridge regression
    gradients.py     Integrated Gradients
    global_methods.py  permutation importance, PDP/ICE, surrogate tree + IF-THEN rules
    counterfactual.py  greedy, cost-aware "what-if" search over actionable features
  trust.py           deletion test, agreement between methods, ground-truth recovery
  narrator.py        turns numbers into a 5-part plain-language story per customer
  plots.py, report.py, framework.py, __main__.py
```

## Use it as a library

```python
from xai_attrition import AttritionXAI

xai = AttritionXAI(csv_path=None).fit()          # or csv_path="BankChurners.csv"
case = xai.explain_customer(xai.data.X_test.iloc[42], customer_id="42")
print(case.story)                                # verdict, starting point, pushes, what-if, confidence
case.explanations["shapley"].top(5)              # raw numbers, if you want them
```
