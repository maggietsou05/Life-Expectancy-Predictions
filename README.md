# Life-Expectancy-Predictions

This repository contains a reproducible analysis comparing Principal Component Regression (PCR) and Elastic Net for predicting country-level life expectancy in the presence of strong multicollinearity. The project uses WHO and World Bank indicators and was developed as part of *FEM11149 – Introduction to Data Science*.



## **Project Structure**

```
life-expectancy-prediction/
├── data/                      Raw datasets
├── src/                       Reproducible analysis script
├── report/                    Original assignment (Rmd + PDF)
└── .gitignore
```

**Key files**

* **src/analysis.R** — full analysis pipeline (preprocessing, PCR, Elastic Net, evaluation)
* **report/IDS_Final_Assignment.Rmd** — original report source (R Markdown)
* **report/IDS_Final_Assignment.pdf** — rendered final report
* **data/** — input datasets used in the analysis

## **Summary**

#### 1. Overview

Life expectancy is widely used to assess public health and societal development. However, many relevant predictors—such as healthcare expenditure, immunization coverage, income level, and disease burden—tend to be strongly correlated, creating substantial multicollinearity. Traditional linear regression becomes unstable in this setting.

This project evaluates two solutions to multicollinearity:

* Principal Component Regression (PCR): Reduces correlated predictors into orthogonal components.
* Elastic Net Regression: Combines L1 and L2 penalties to shrink coefficients and perform variable selection.

The objective is to determine which method produces the most accurate and generalizable predictions.

#### 2. Data Description

The dataset includes 160 countries and 28 predictors, compiled from the World Health Organization (WHO) and the World Bank. 

#### 3. Methodology

Preprocessing:

* Train-test split: 80% training, 20% testing
* Median imputation for numeric missing values
* Log transformations for right-skewed variables
* Logit transformations for variables bounded between 0 and 1

Elastic Net Regression:

Elastic Net was implemented using glmnet with 10-fold cross-validation. A grid search over alpha ∈ [0, 1] identified an optimal balance between L1 (lasso) and L2 (ridge) penalties.


Principal Component Regression:

PCR was implemented using standardized predictors and evaluated using:

* Kaiser’s rule
* Scree plot inspection
* 70% cumulative variance explained threshold
* Permutation testing (1,000 iterations)
* Bootstrap validation of eigenvalues (1,000 resamples)

Five principal components were selected, explaining approximately 75% of the total variance. These components aligned with interpretable dimensions such as: Healthcare and economic development, immunization systems, population structure, demographic transition, behavioral health risk.

## **Results (Test Set)**

<img width="859" height="137" alt="image" src="https://github.com/user-attachments/assets/6d9f1886-6d01-4356-b733-c55d1ae3be56" />


**Key findings:**

* Multicollinearity was substantial even after transformations, justifying specialized methods.
* Elastic Net with minimal penalty achieved the lowest test RMSE, identifying key predictors such as fertility rate, sex ratio at birth, TB mortality, sanitation access, and government health expenditure.
* PCR achieved nearly identical accuracy using only five components, providing a more compact model and the most consistent residuals across countries with different income levels.
* Stability analysis showed that PCR exhibited no income-related prediction bias, while the more heavily penalized Elastic Net model did.



## **Reproducibility**

### Install required packages

```r
install.packages(c(
  "dplyr", "tidyr", "ggplot2", "corrplot", "naniar",
  "car", "glmnet", "pls", "reshape2", "e1071"
))
```

### Run the full analysis

```r
source("src/analysis.R")
```

### Regenerate the report

Open and knit:

```
report/IDS_Final_Assignment.Rmd
```



## **Data Sources**

Data files are stored in the `data/` directory:

* `health_nutrition_population.csv`
* `life_expectancy_data-1.csv`
* `predictions-1.csv`

These datasets contain WHO and World Bank indicators used as model inputs.
