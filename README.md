# Product Review Sentiment Classifier

## Overview

This project develops and compares machine learning models for
classifying customer product reviews into three sentiment categories:
**negative, neutral, and positive**.

It compares a traditional TF-IDF machine learning baseline with a
fine-tuned Transformer, investigates class imbalance, analyzes
misclassified reviews, and considers deployment trade-offs.

## Objectives

-   Build a sentiment classifier using real customer reviews.
-   Compare TF-IDF with Logistic Regression against fine-tuned
    DistilBERT.
-   Evaluate accuracy, precision, recall, and F1-score.
-   Investigate class imbalance using class weighting.
-   Analyze misclassified examples and plausible causes.
-   Assess model size and inference speed.

## Dataset

**Dataset:** [Kaggle Women's E-Commerce Clothing
Reviews](https://huggingface.co/datasets/chibifire/kaggle-womens-ecom-clothing-reviews)

After removing missing and empty review text, the dataset contained
**22,641 reviews**.

### Sentiment labeling

Star ratings were mapped to sentiment labels as follows:

  Rating       Sentiment
  ------------ -----------
  1--2 stars   Negative
  3 stars      Neutral
  4--5 stars   Positive

### Class distribution

  Sentiment        Reviews   Percentage
  ----------- ------------ ------------
  Positive          17,448       77.06%
  Neutral            2,823       12.47%
  Negative           2,370       10.47%
  **Total**     **22,641**     **100%**

**Labeling limitation:** Ratings are used as proxies for sentiment. A
review may contain mixed opinions that do not perfectly match its
rating-derived label.

## Data preparation

A stratified train/test split was created with `random_state=42`:

-   Training set: 18,112 reviews
-   Test set: 4,529 reviews

The Transformer training data was further split into:

-   Training: 16,300 reviews
-   Validation: 1,812 reviews
-   Test: 4,529 reviews

The same original train/test split was used for all model approaches.

## Models

### 1. TF-IDF + Logistic Regression

-   TF-IDF with up to 20,000 features
-   Unigrams and bigrams
-   `min_df=2`
-   Logistic Regression with `max_iter=1000`

### 2. Balanced Logistic Regression

The same TF-IDF pipeline with `class_weight="balanced"` to increase the
influence of minority-class examples during training.

### 3. Fine-tuned DistilBERT

The pretrained `distilbert/distilbert-base-uncased` checkpoint was
fine-tuned for three-class sequence classification.

-   Maximum sequence length: 256 tokens
-   Learning rate: `2e-5`
-   Training epochs: 3
-   Training batch size: 16
-   Validation batch size: 32
-   Weight decay: `0.01`
-   Best checkpoint selected using validation Macro F1

### 4. Weighted DistilBERT

A second DistilBERT experiment used class-weighted cross-entropy loss.
Class weights were calculated from the training labels:

-   Negative: 3.1848
-   Neutral: 2.6739
-   Positive: 0.4325

## Results

### Test-set comparison

  -----------------------------------------------------------------------------
  Model            Accuracy     Macro F1  Negative F1   Neutral F1  Positive F1
  ------------ ------------ ------------ ------------ ------------ ------------
  Logistic           82.51%       57.57%       49.94%       30.82%       91.94%
  Regression                                                       

  Balanced           79.95%       62.71%       53.92%       42.80%       91.40%
  Logistic                                                         
  Regression                                                       

  DistilBERT         84.79%       66.87%       61.99%       44.22%       94.39%

  Weighted           83.31%       68.64%       65.56%       47.24%       93.12%
  DistilBERT                                                       
  -----------------------------------------------------------------------------

### Key findings

-   Standard DistilBERT achieved the highest accuracy: **84.79%**.
-   Weighted DistilBERT achieved the highest Macro F1: **68.64%**.
-   Class weighting improved negative and neutral F1 compared with
    standard DistilBERT.
-   Neutral sentiment remained the most difficult category.
-   Balanced Logistic Regression improved Macro F1 over the unweighted
    baseline, although its accuracy decreased.

Macro F1 is useful here because it gives equal weight to each class,
rather than allowing the majority positive class to dominate the
evaluation.

**Evaluation note:** Test-set predictions were inspected during
development and error analysis. The test set is therefore not a fully
untouched final evaluation. An independent holdout set or external
dataset would provide a stronger estimate of generalization.

## Error analysis

Five misclassified reviews were examined. Common patterns included:

-   **Mixed sentiment:** A review praised some product features while
    criticizing others.
-   **Positive keywords:** Words such as "soft," "comfortable," and
    "cute" could coexist with dissatisfaction.
-   **Contextual meaning:** Fit, sizing, material, and return decisions
    required interpreting the whole review.
-   **Label ambiguity:** Text sentiment did not always align with the
    rating-derived label.

These are plausible hypotheses based on the review text, not definitive
explanations of the model's internal reasoning. See
`results/error_analysis.csv` for examples.

## Deployment evaluation

A sample of 100 reviews was timed in Google Colab.

  Measurement                                         Result
  ----------------------------------- ----------------------
  Model parameters                                66,955,779
  Parameter memory size                            255.42 MB
  Average inference time per review           0.0140 seconds
  Measured throughput                   71.44 reviews/second

These measurements were obtained in Colab using individual predictions.
They may differ on other hardware or with batch inference. Parameter
memory is not the same as the complete checkpoint size or total runtime
memory.

### Deployment recommendation

Weighted DistilBERT is a candidate when balanced performance across
sentiment classes matters: it achieved the highest Macro F1 among the
evaluated models. However, neutral-class performance remains limited.
TF-IDF with Logistic Regression is a simpler alternative for
resource-constrained environments. Further testing on target hardware
and an independent dataset is recommended before production deployment.

## Limitations

-   Labels are derived from star ratings rather than manually annotated
    text.
-   Mixed-sentiment reviews can be difficult to represent with a single
    label.
-   The dataset is imbalanced.
-   Neutral-class performance needs improvement.
-   Test-set results were inspected during development.
-   Colab inference measurements may not represent production
    performance.

## Technologies

-   Python
-   pandas
-   NumPy
-   scikit-learn
-   Hugging Face Transformers
-   Hugging Face Datasets
-   PyTorch
-   Google Colab

## Project structure

``` text
product-review-sentiment-classifier/
├── Product_Review_Sentiment_Classifier.ipynb
├── README.md
├── requirements.txt
└── results/
    ├── model_comparison.csv
    └── error_analysis.csv
```

## How to run

1.  Clone or download this repository.
2.  Open `Product_Review_Sentiment_Classifier.ipynb` in Google Colab or
    Jupyter.
3.  Install the dependencies listed in `requirements.txt`.
4.  Load the dataset from Hugging Face.
5.  Run the notebook cells in order to reproduce preprocessing,
    training, evaluation, and inference experiments.

Transformer fine-tuning is computationally intensive; a GPU-enabled
environment is recommended.

## Conclusion

The experiments show that class weighting can improve minority-class
performance and Macro F1, while potentially reducing overall accuracy.
They also highlight the importance of class imbalance, label quality,
error analysis, and deployment constraints in text classification.
