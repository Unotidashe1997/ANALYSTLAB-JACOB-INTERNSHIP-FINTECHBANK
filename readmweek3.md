
Week 3 of the AnalystLab Africa Experience Lab: done ✅ — and this was the week the project really earned the word "development."

Week 1 was about understanding the problem. Week 2 was about building a working baseline. Week 3 was about pressure-testing everything I'd built and making it genuinely better — with evidence, not guesswork.

A few highlights from the FinTrust Digital Bank risk-review model this week:

🤖 Trained 2 additional models (Decision Tree, Gradient Boosting) alongside my Week 2 baselines, and compared all 4 on Accuracy, Precision, Recall, F1, and ROC-AUC — not just one metric in isolation

⚠️ Caught a classic trap: my Gradient Boosting model looked like the best performer at 78% accuracy... until I checked recall, which was 2.7%. It was barely predicting any risk at all. Tuning the decision threshold took recall from 2.7% to 69%, which became my concrete example of real, evidence-based model refinement

🔍 Ran a proper error analysis — not just "here's the accuracy," but actually looking at which transactions the model gets wrong and why

🧩 Used permutation importance (a more reliable method than standard feature importance) to finally settle a question that had been open since Week 1: do customer-level features like tenure and engagement actually matter? Turns out: barely at all. Transaction-level context is what drives the signal

✅ Selected and validated a candidate model on a completely untouched test set, confirming the results generalise rather than just looking good on paper

The biggest lesson this week: a strong accuracy number can hide a model that's quietly useless. Learning to distrust a single metric — and to actually dig into *why* a model is wrong, not just *that* it's wrong — feels like a genuine levelling-up moment in how I think about this work.

Thank you to AnalystLab Africa for structuring this programme so that "improve it and prove it" is a required step, not an afterthought. Heading into Week 4 with a validated model and a much clearer picture of its real strengths and limitations.

#AnalystLabAfrica #DataScience #FinTech #MachineLearning #TechAfrica
