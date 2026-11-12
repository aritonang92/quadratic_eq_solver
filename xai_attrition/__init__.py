"""Explainable AI (XAI) framework for credit-card attrition neural networks.

Quick start::

    from xai_attrition import AttritionXAI
    xai = AttritionXAI().fit()
    results = xai.run()                      # global + local + trust
    print(results.cases[0].story)            # plain-language explanation

    # explain any single customer
    case = xai.explain_customer(xai.data.X_test.iloc[0], customer_id="demo")
    print(case.story)
"""
from .framework import AttritionXAI, XAIResults

__all__ = ["AttritionXAI", "XAIResults"]
