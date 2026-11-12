"""
Data layer: load (or simulate) credit-card customers and prepare them for the ANN.

FEYNMAN NOTE
------------
Simple idea:  A bank keeps a spreadsheet. One row = one cardholder. Columns say
              how old they are, how often they swipe the card, how many times they
              phoned the bank, ... and the last column says whether they left.
Why simulate: The public "BankChurners" dataset (Kaggle) is the usual benchmark.
              If you have it, drop ``BankChurners.csv`` in and we use it. If not,
              we generate a look-alike whose *true* causes of leaving we wrote
              ourselves. That is a gift for XAI: we can check whether the
              explanations rediscover the causes we planted. A teacher who knows
              the answer can grade the student.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Attrition_Flag"
POSITIVE_LABEL = "Attrited Customer"  # 1 = the customer left

NUMERIC_FEATURES = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]
CATEGORICAL_FEATURES = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Things the bank can actually influence (used for counterfactual "what-if" advice).
# Age, gender, education ... are facts about a person, not levers.
ACTIONABLE_FEATURES = [
    "Total_Trans_Ct",
    "Total_Trans_Amt",
    "Total_Revolving_Bal",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
]

# Human-friendly names so explanations read like sentences, not column headers.
PLAIN_NAMES = {
    "Customer_Age": "age",
    "Dependent_count": "number of dependents",
    "Months_on_book": "months as a customer",
    "Total_Relationship_Count": "number of bank products held",
    "Months_Inactive_12_mon": "inactive months (last year)",
    "Contacts_Count_12_mon": "contacts with the bank (last year)",
    "Credit_Limit": "credit limit",
    "Total_Revolving_Bal": "revolving balance",
    "Total_Amt_Chng_Q4_Q1": "spend change Q4 vs Q1",
    "Total_Trans_Amt": "total spend (last year)",
    "Total_Trans_Ct": "number of transactions (last year)",
    "Total_Ct_Chng_Q4_Q1": "transaction-count change Q4 vs Q1",
    "Avg_Utilization_Ratio": "credit utilisation",
    "Gender": "gender",
    "Education_Level": "education",
    "Marital_Status": "marital status",
    "Income_Category": "income band",
    "Card_Category": "card tier",
}

# The columns we planted as real drivers in the simulator (ground truth for validation).
SIMULATED_TRUE_DRIVERS = [
    "Total_Trans_Ct",
    "Total_Revolving_Bal",
    "Total_Ct_Chng_Q4_Q1",
    "Contacts_Count_12_mon",
    "Months_Inactive_12_mon",
    "Total_Relationship_Count",
    "Total_Trans_Amt",
]


def plain(feature: str) -> str:
    return PLAIN_NAMES.get(feature, feature)


# --------------------------------------------------------------------------- #
# Loading / simulating
# --------------------------------------------------------------------------- #
def simulate_bank_churners(n: int = 10_000, seed: int = 7) -> pd.DataFrame:
    """Generate a BankChurners-shaped dataset with *known* attrition drivers.

    The recipe (our planted "truth"): customers leave when they stop using the
    card (few transactions, falling usage), carry no revolving balance (nothing
    keeps them), hold few products, are inactive, and keep calling the bank
    (complaints). Demographics have ~no effect - so a good explainer should rank
    them low.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(46, 8, n).clip(26, 73).round()
    gender = rng.choice(["F", "M"], n, p=[0.53, 0.47])
    dependents = rng.integers(0, 6, n)
    education = rng.choice(
        ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"],
        n, p=[0.15, 0.20, 0.10, 0.31, 0.05, 0.04, 0.15],
    )
    marital = rng.choice(["Married", "Single", "Divorced", "Unknown"], n, p=[0.46, 0.39, 0.07, 0.08])
    income = rng.choice(
        ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"],
        n, p=[0.35, 0.18, 0.14, 0.15, 0.07, 0.11],
    )
    card = rng.choice(["Blue", "Silver", "Gold", "Platinum"], n, p=[0.93, 0.055, 0.012, 0.003])
    months_on_book = (age - 26 + rng.normal(10, 6, n)).clip(13, 56).round()
    relationships = rng.integers(1, 7, n)
    inactive = rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.01, 0.22, 0.32, 0.38, 0.04, 0.02, 0.01])
    contacts = rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.04, 0.15, 0.32, 0.33, 0.13, 0.02, 0.01])

    income_mult = pd.Series(income).map(
        {"Less than $40K": 0.5, "$40K - $60K": 0.8, "$60K - $80K": 1.2,
         "$80K - $120K": 1.6, "$120K +": 2.2, "Unknown": 0.9}).to_numpy()
    card_mult = pd.Series(card).map({"Blue": 1.0, "Silver": 2.0, "Gold": 2.8, "Platinum": 3.5}).to_numpy()
    credit_limit = (rng.lognormal(8.6, 0.6, n) * income_mult * card_mult).clip(1438, 34516).round()

    # Latent "engagement" drives usage; it is what really keeps customers.
    engagement = rng.normal(0, 1, n)
    revolving = np.where(rng.random(n) < 0.25, 0,
                         rng.uniform(300, 2517, n) * (1 + 0.15 * engagement)).clip(0, 2517).round()
    trans_ct = (65 + 20 * engagement + rng.normal(0, 8, n)).clip(10, 139).round()
    trans_amt = (trans_ct * rng.lognormal(4.0, 0.35, n) + 0.02 * credit_limit).clip(510, 18484).round()
    ct_chng = (0.71 + 0.18 * engagement + rng.normal(0, 0.18, n)).clip(0, 3.7).round(3)
    amt_chng = (0.76 + 0.12 * engagement + rng.normal(0, 0.2, n)).clip(0, 3.4).round(3)
    utilization = (revolving / credit_limit).clip(0, 1).round(3)

    # ---- the planted truth: log-odds of leaving --------------------------------
    z = (
        -3.2
        - 0.075 * (trans_ct - 65)               # fewer transactions  -> leave
        - 0.0011 * (revolving - 1160)           # no revolving balance -> leave
        - 2.6 * (ct_chng - 0.71)                # usage dropping       -> leave
        + 0.55 * (contacts - 2.45)              # keeps calling        -> leave
        + 0.45 * (inactive - 2.34)              # inactive months      -> leave
        - 0.35 * (relationships - 3.8)          # few products         -> leave
        + 0.00012 * (trans_amt - trans_amt.mean())  # big spenders with few txns are flight risks
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-z))

    df = pd.DataFrame({
        "CLIENTNUM": rng.choice(np.arange(708_000_000, 828_000_000), n, replace=False),
        TARGET: np.where(churn, POSITIVE_LABEL, "Existing Customer"),
        "Customer_Age": age.astype(int),
        "Gender": gender,
        "Dependent_count": dependents,
        "Education_Level": education,
        "Marital_Status": marital,
        "Income_Category": income,
        "Card_Category": card,
        "Months_on_book": months_on_book.astype(int),
        "Total_Relationship_Count": relationships,
        "Months_Inactive_12_mon": inactive,
        "Contacts_Count_12_mon": contacts,
        "Credit_Limit": credit_limit,
        "Total_Revolving_Bal": revolving.astype(int),
        "Total_Amt_Chng_Q4_Q1": amt_chng,
        "Total_Trans_Amt": trans_amt.astype(int),
        "Total_Trans_Ct": trans_ct.astype(int),
        "Total_Ct_Chng_Q4_Q1": ct_chng,
        "Avg_Utilization_Ratio": utilization,
    })
    df["Avg_Open_To_Buy"] = df["Credit_Limit"] - df["Total_Revolving_Bal"]
    return df


def load_data(csv_path: str | Path | None = None, n_simulated: int = 10_000, seed: int = 7
              ) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_description). Uses the real CSV when available."""
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        # The Kaggle file ships two leaked Naive-Bayes columns - remove them.
        df = df.loc[:, ~df.columns.str.startswith("Naive_Bayes")]
        return df, f"real data: {Path(csv_path).name}"
    return simulate_bank_churners(n_simulated, seed), "simulated BankChurners look-alike (ground truth known)"


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bookkeeping identities true after a what-if edit (balance -> utilisation)."""
    df = df.copy()
    if {"Total_Revolving_Bal", "Credit_Limit", "Avg_Utilization_Ratio"} <= set(df.columns):
        df["Avg_Utilization_Ratio"] = (df["Total_Revolving_Bal"] / df["Credit_Limit"]).clip(0, 1)
    return df


# --------------------------------------------------------------------------- #
# Splitting & preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    source: str
    ids_test: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def base_rate(self) -> float:
        return float(self.y_train.mean())


def make_preprocessor() -> ColumnTransformer:
    """Numbers -> z-scores, categories -> one-hot. ANNs learn badly from raw scales.

    Note: ``Avg_Open_To_Buy`` is deliberately *not* used. It is exactly
    Credit_Limit - Revolving_Bal; feeding the network the same fact twice makes
    explanations split the credit between twins and confuses the story.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def prepare(df: pd.DataFrame, source: str, test_size: float = 0.25, seed: int = 7) -> Dataset:
    y = (df[TARGET] == POSITIVE_LABEL).astype(int).to_numpy()
    ids = df["CLIENTNUM"].to_numpy() if "CLIENTNUM" in df else np.arange(len(df))
    X = df[FEATURES].copy()
    X_tr, X_te, y_tr, y_te, _, id_te = train_test_split(
        X, y, ids, test_size=test_size, stratify=y, random_state=seed)
    pre = make_preprocessor().fit(X_tr)
    return Dataset(X_tr.reset_index(drop=True), X_te.reset_index(drop=True),
                   y_tr, y_te, pre, source, id_te)


def encoded_to_raw_map(preprocessor: ColumnTransformer) -> list[str]:
    """For each encoded column, the raw feature it came from (one-hot columns fold back)."""
    owners: list[str] = list(NUMERIC_FEATURES)
    ohe: OneHotEncoder = preprocessor.named_transformers_["cat"]
    for feat, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
        owners.extend([feat] * len(cats))
    return owners
