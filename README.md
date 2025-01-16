# Predicting Online Shopper Purchasing Intentions

Welcome to the **Online Shopper Purchasing Intentions Prediction** project! This repository contains the complete pipeline for developing a machine learning model and an interactive dashboard to predict whether a customer visiting an online shopping website will make a purchase.

---

## 🛍️ **Project Overview**

The goal of this project is to assist businesses in optimizing their e-commerce platforms and improving user experience by providing real-time insights into user behavior. By analyzing shopper activity data, we aim to help sellers:

- **Understand customer behavior** through predictive analytics.
- **Identify key performance indicators (KPIs)** dynamically.
- **Enhance marketing strategies** based on user patterns and trends.

---

## 📂 **Key Components**

1. **Dataset**  
   - **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)  
   - **Description**: We have made enhancements to the original dataset by adding a few features to enrich the analysis and improve predictive modeling. The modified dataset is available in this repository for your convenience. The modified dataset includes 12,000+ rows and 22 features capturing user interactions, such as page visits, session duration, and traffic sources.

2. **Data Cleaning and Wrangling**
   - Handled missing values using mean imputation.
   - Removed 17 duplicate rows for cleaner data.
   - Corrected data inconsistencies (e.g., standardizing month abbreviations).
   - Retained outliers to capture valuable customer behaviors.

3. **SQL Database Design**
   - Transformed raw data into a normalized relational database with tables for Metrics, Customers, and Analytics.
   - Ensured compliance with 1NF and 2NF to improve data integrity and storage efficiency.

4. **Exploratory Data Analysis (EDA)**
   - Visualized trends using histograms, heatmaps, and scatterplots.
   - Addressed imbalanced data with insights for model selection.

5. **Machine Learning Model**
   - Adopted a **Random Forest Classifier** for robust predictions.
   - Evaluated model using accuracy, precision, recall, and F1-score.
   - Conducted feature importance analysis to highlight key factors.

6. **Interactive Dashboard** [Click Here to Access](https://shoppingintentionpredictor.streamlit.app/)
   - Built using **Streamlit** for seamless exploration and prediction.
   - Key Features:
     - Real-time shopper intent classification.
     - Dynamic KPI dashboards.
     - Customizable data visualizations (e.g., histograms, bar charts, heatmaps).

---

## 🚀 **End-to-End Pipeline**

1. **Data Preparation**
   - Downloaded dataset and performed cleaning (missing values, duplicates, normalization).
2. **Database Integration**
   - Normalized data into SQLite relational database.
   - Queried aggregated subsets for feature analysis.
3. **Model Training**
   - Developed Random Forest model on cleaned data.
   - Addressed multicollinearity using VIF analysis.
4. **App Deployment**
   - Created a [Streamlit app](https://shoppingintentionpredictor.streamlit.app/) for real-time predictions.
   - Deployed online for accessible usage.

---

## 📊 **Insights and Findings**

- **Behavior Patterns**:
  - Returning visitors exhibit higher conversion rates compared to new visitors.
  - Weekend activity correlates with increased purchases.
- **Key Features**:
  - Session duration and product page visits strongly influence purchasing intentions.
  - High-value pages (low exit/bounce rates) are critical for conversions.

---

## 📋 **Technologies Used**

- **Languages**: Python, SQL
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, Streamlit
- **Tools**: SQLite, Jupyter Notebook, Streamlit Sharing

---

## 🛠️ **Usage**

1. Clone the repository:
   ```bash
   git clone https://github.com/ShreyasKadam77/Shopping_Intention_Predictor

