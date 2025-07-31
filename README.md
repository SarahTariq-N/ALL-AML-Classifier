
# Gene Expression-Based Classification of AML vs ALL using Machine Learning

## 🧬 1. Background: Gene Expression and Data Collection

This project uses a publicly available gene expression dataset originally published by Golub et al. in 1999, which contains expression levels of 7,129 genes measured in bone marrow and peripheral blood samples from leukemia patients. The goal is to distinguish between two major subtypes of acute leukemia:

- **AML**: Acute Myeloid Leukemia  
- **ALL**: Acute Lymphoblastic Leukemia

The dataset was obtained using **DNA microarray technology**, where gene-specific DNA probes are used to quantify the mRNA expression of thousands of genes simultaneously. In the lab, samples from both healthy and cancerous tissue are labeled with fluorescent markers and hybridized to microarray chips. The resulting fluorescence intensities correspond to gene expression levels, which form the raw dataset used for this project.

---

## 🔬 2. Dimensionality Reduction: PCA vs T-Test

Gene expression datasets are **high-dimensional**, with far more features (genes) than samples. This makes dimensionality reduction or feature selection crucial before applying machine learning models.

I explored two different approaches:

- **Principal Component Analysis (PCA)**: An unsupervised method that reduces dimensionality by finding orthogonal components that capture the most variance in the data. While useful for visualization, PCA does **not use class labels** and transforms features into unrecognizable combinations of genes, making interpretation difficult.

- **T-Test (Univariate Hypothesis Testing)**: A supervised method that tests each gene individually for statistically significant differences in expression between AML and ALL samples. This allows direct selection of biologically meaningful genes.

In this case, models trained on **t-test-selected genes outperformed PCA-based models** in classification accuracy. This suggests that the discriminatory information is concentrated in a subset of genes rather than distributed across many components.

---

## 🤖 3. Classification Models Used

After selecting features using the t-test, I trained three classification models to predict AML vs ALL:

1. **Logistic Regression**: A linear model suitable for binary classification. I tuned hyperparameters (`C`, `penalty`) using grid search with 3-fold cross-validation.

2. **Support Vector Machine (SVM)**: Effective for high-dimensional spaces, especially with linear kernels. Used grid search to tune `C`.

3. **Naive Bayes**: A probabilistic model based on Bayes’ theorem, assuming feature independence. It is fast, simple, and often surprisingly effective for high-dimensional data like gene expression.

All models were trained on the same selected features and evaluated on a held-out test set.

---

## 📏 4. Evaluation Metrics

To assess the performance of each model, I used:

- **Accuracy**: Overall proportion of correct predictions.

- **Confusion Matrix**: Shows counts of true positives, false positives, true negatives, and false negatives.

- **True Positive Rate (Recall for AML)**:  
  TPR = TP/(TP + FN)
  Measures how well the model identifies AML patients.

- **True Negative Rate (Specificity for ALL)**:  
  TNR = TN/(TN + FP)
  Measures how well the model correctly identifies ALL patients.

---

## 📊 5. Results Summary

The models trained using **t-test-selected genes** achieved higher classification performance than those using PCA-reduced components. Among classifiers, `Logistic Regression` and `SVM` yielded the best accuracy, followed closely by `Naive Bayes`.

- Logistic Regression Accuracy: 0.765
- SVM Accuracy: 0.794
- Naive Bayes Accuracy: 0.735

- TPR (AML Recall): 0.714
- TNR (ALL Specificity): 0.85

These results confirm that **statistical feature selection** and **interpretable gene-level analysis** offer advantages for gene expression-based classification.

---

## 📂 6. Repository Structure

```
├── data/
│   └── actual.csv
|   └── data_set_ALL_AML_independent.csv
|   └── data_set_ALL_AML_train.csv
│
├── notebook/
│   ├── Hypothesis Testing.ipynb
│   └── Dimensionality Reduction and Clustering.ipynb
│
├── docs
|   └── Gene Expression Background.pdf
│
├── README.md
```

---

## 📚 7. References

- Golub et al. (1999), *Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring*
- Dataset Source: https://www.kaggle.com/datasets/andrewmvd/leukemia-gene-expression-dataset
- Background biology from:  
  - https://www.cancer.gov/types/leukemia  
  - https://www.onlinebiologynotes.com/dna-microarray-principle-types-and-steps-involved-in-cdna-microarrays/
