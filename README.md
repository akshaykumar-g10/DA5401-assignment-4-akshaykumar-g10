# DA5401 Assignment 4: GMM-Based Synthetic Sampling for Imbalanced Data
- AKSHAY KUMAR G
- ME22B044

  
## **Objective**  
To use GMM based sampling to fix class imbalance. Evaluate the impact on a Logistic Regression model's performance.

## **Problem Statement**
The dataset is a **credit card fraud detection dataset** (from Kaggle). It is **highly imbalanced**:  
- Majority class → non-fraud transactions  
- Minority class → fraud transactions
  
Build a fraud detection model on a highly imbalanced dataset. Use sampling to create better training data and improve performance.

Task: Compare a Logistic Regression model trained on three different datasets:
- Original Imbalanced Data
- GMM Sampling
- GMM + CBU Sampling

## **2. Tasks & Solutions**

### **Part A: Data Exploration and Baseline Model**

1. **Load & Inspect Dataset**  
   - Dataset: `creditcard.csv`
    
2. **Class Distribution**  
   - Extreme imbalance between non-fraud and fraud cases.  
   - Visualized using a bar chart and observed the degree of imbalance.    

3. **Baseline Model**  
   - Data split into **train (80%) and test (20%)**, preserving imbalance in test set.  
   - Logistic Regression trained on original imbalanced training data.  
   - Evaluated using **Precision, Recall, and F1-score** (accuracy alone is misleading due to imbalance).  

### **Part B: Gaussian Mixture Model (GMM) for Synthetic Sampling**

#### **1. Theoretical Foundation**  
- GMM-based sampling is a **probabilistic generative approach**, unlike SMOTE which relies on linear interpolation between neighbors.  
- GMM models the minority data as a **mixture of multiple Gaussian distributions**, each representing a potential subgroup of the minority class.  
- This allows it to capture **multi-modal and complex shapes** in the feature space, which SMOTE may fail to represent.  
- By sampling from the learned Gaussian distributions, GMM produces **more realistic and diverse synthetic samples** that better reflect the true underlying distribution.  

#### **2. GMM Implementation**  
- A **Gaussian Mixture Model** was fitted to the **minority class training data**.  
- The optimal number of components (`k`) was selected by comparing **Akaike Information Criterion (AIC)** and **Bayesian Information Criterion (BIC)** scores across different values of `k`.  
- The model with the **lowest BIC score** was chosen, ensuring a balance between model fit and complexity.  
- This guarantees that the GMM captures meaningful substructures of the minority data without overfitting.  

#### **3. Synthetic Data Generation**  
- After fitting, the GMM was used to generate **new synthetic samples** for the minority class.  
- Sampling process:  
  1. Randomly select a Gaussian component according to its weight (π).  
  2. Draw a sample from the corresponding Gaussian distribution (mean µ, covariance Σ).  
- This process was repeated until the minority class size matched the majority (or desired ratio).  
- The **original minority data + generated synthetic samples** formed the new balanced minority dataset.  

#### **4. Rebalancing with CBU + GMM**  
- To control the majority class, **Clustering-Based Undersampling (CBU)** was applied:  
  - Majority class samples were clustered using K-Means.  
  - Samples were then undersampled **proportionally from each cluster** to reduce size while preserving diversity.  
- On the minority side, **GMM-based synthetic sampling** was applied to increase its size until it matched the undersampled majority.  
- This combination ensured a **balanced dataset** with both:  
  - A compact yet diverse majority set.  
  - A realistic, distribution-aware synthetic minority set.  
- A Logistic Regression classifier was then trained on the rebalanced dataset.  


### **Part C: Performance Evaluation with Logistic Regression**

### **Benefits and Drawbacks of Each Method**

1. Baseline (without resampling)

- Precision is very high (~0.83): When the model predicts fraud, it is usually correct.
- Recall is moderate (~0.64): The model misses a significant fraction of fraud cases.
- F1-score (~0.72): Balanced performance.

2. GMM Oversampling

- Recall shoots up to ~0.91: The model becomes much better at detecting fraud.
- Precision drops drastically (~0.09): Most of the predicted frauds are false alarms.
- F1-score drops (~0.15): Overall performance worsens despite high recall.

3. CBU + GMM

- Very similar to GMM-only: high recall (0.91), low precision (0.08), poor F1 (~0.14).
- Clustering-based undersampling didn’t fix the precision-recall tradeoff.

<img width="846" height="598" alt="image" src="https://github.com/user-attachments/assets/63a27b0f-a726-4479-bcee-d1b0bf1dd1d2" />

## **Conclusion**  

Based on both results and theoretical understanding:

1. Effectiveness of GMM in this context:

- GMM oversampling does improve recall significantly (good for not missing frauds).
- However, the drop in precision and F1 shows that synthetic samples from GMM may not represent the true minority distribution well. They might confuse the classifier by overlapping too much with majority class space.
- Thus, the tradeoff is poor: you detect almost all frauds but at the cost of flagging many normal transactions.

2. Recommendation:

- Do not rely on GMM oversampling alone in this fraud detection setting.
- It can be useful if high recall is absolutely critical (e.g., catching all frauds and letting humans investigate flagged cases).
- But for a balanced automated system, GMM oversampling is not recommended due to poor precision and F1.

---
