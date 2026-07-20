# Titanic Survival Prediction — Decision Tree Classifier

An end-to-end, fully commented **Decision Tree classification pipeline** predicting Titanic passenger survival — from raw CSV through cleaning, encoding, and training to an **interpretable tree visualization** showing exactly how the model decides. Test accuracy: **79.9%**, a 21-point lift over the majority baseline.

## Objective

Decision trees are the most *explainable* nonlinear model: every prediction is a readable chain of if/else questions. This project builds one on the classic Titanic dataset and treats the tree diagram itself as the deliverable — the model's logic, visible and auditable, mirroring the historical "women and children first" reality in its very first split.

## Dataset

| Property | Detail |
|---|---|
| Records | 891 passengers × 12 columns |
| Target | `Survived` — 38.4% survived (342 / 549) |
| Cleaning | Age: median-filled (177 missing) · Embarked: mode-filled (2) · Cabin: dropped (77% missing) |
| Encoding | Sex mapped 0/1 · Embarked one-hot with `drop_first=True` |
| Features (8) | Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S |

## Model

`DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)` on an 80/20 split (712 train / 179 test). **`max_depth=4` is the key choice** — an unconstrained tree memorizes the training data; capping depth at 4 (15 leaves) trades a little training fit for genuine generalization.

## Performance (179-passenger test set)

| Metric | Value |
|---|---|
| Accuracy | **79.9%** |
| Majority-class baseline | 58.7% |
| Precision (survived) | 0.839 |
| Recall (survived) | 0.635 |
| Confusion matrix | TN 96 · FP 9 · FN 27 · TP 47 |

The model beats "predict everyone died" by **+21.2 points**. Precision is strong (when it predicts survival, it's right 84% of the time); recall of 0.635 shows it misses some survivors — the classic trade-off of a shallow, conservative tree.

## What the Tree Learned (feature importances)

| Feature | Importance |
|---|---|
| **Sex** | **58.0%** |
| **Pclass** | **20.0%** |
| Fare | 8.1% |
| Age | 7.6% |
| SibSp | 4.6% |
| Others | <2% |

**The root split is Sex** — the single most informative question. For men, the next question is Age (≤6.5 years: the "children first" branch); for women, it's Pclass (the class-privilege effect). Two features carry 78% of the predictive power, and the tree's structure reads as a data-derived retelling of the evacuation's social rules.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` (DecisionTreeClassifier, plot_tree, train_test_split, accuracy_score) · `matplotlib`

## Repository Structure

```
├── Titanic_Decision_Tree.ipynb   # 11-step commented pipeline + tree visualization
├── Titanic-Dataset.csv           # Dataset (891 passengers)
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook Titanic_Decision_Tree.ipynb
```

## Related Repos

- [`handling-missing-data-ml`](../handling-missing-data-ml) — the imputation strategies used here, studied in depth
- [`iris-flower-classification-ml`](../iris-flower-classification-ml) — multi-model benchmarking companion

---

*Author: Kailas Wadje — MSc Data Science & AI, University of Liverpool*
