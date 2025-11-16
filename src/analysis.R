## src/analysis.R
## Predicting Life Expectancy Under Multicollinearity:
## Principal Component Regression vs Elastic Net

## 0. Setup ----
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(corrplot)
  library(naniar)
  library(car)
  library(glmnet)
  library(pls)
  library(reshape2)
  library(e1071)  # for skewness
})

set.seed(100)

## 1. Load data ----
health_data <- read.csv("data/health_nutrition_population.csv", stringsAsFactors = FALSE)
life_exp    <- read.csv("data/life_expectancy_data.csv", stringsAsFactors = FALSE)
pred_data   <- read.csv("data/predictions.csv", stringsAsFactors = FALSE)

# Assumes there is a common 'Country' column in all files
full_data <- health_data %>%
  inner_join(life_exp, by = "Country")

# optional: check structure
# str(full_data)

## 2. Train/test split ----
n <- nrow(full_data)
train_index <- sample(seq_len(n), size = 0.8 * n)

train_data <- full_data[train_index, ]
test_data  <- full_data[-train_index, ]

## 3. Median imputation for numeric predictors ----
numeric_cols <- sapply(train_data, is.numeric)

# Do not impute the outcome
if ("LifeExpectancy" %in% names(numeric_cols)) {
  numeric_cols["LifeExpectancy"] <- FALSE
}

train_medians <- vapply(
  names(train_data)[numeric_cols],
  function(col) median(train_data[[col]], na.rm = TRUE),
  numeric(1)
)

for (col in names(train_medians)) {
  train_data[[col]][is.na(train_data[[col]])] <- train_medians[[col]]
  if (col %in% names(test_data)) {
    test_data[[col]][is.na(test_data[[col]])]  <- train_medians[[col]]
  }
}

## 4. Drop variables not used & build numeric subset ----
# If there is an 'UrbanPopulation' column (as in your report), drop it for consistency
if ("UrbanPopulation" %in% names(train_data)) {
  train_data <- train_data %>% select(-UrbanPopulation)
  test_data  <- test_data  %>% select(-UrbanPopulation)
}

# Keep numeric predictors (excluding Country and outcome)
train_numeric <- train_data %>%
  select(-Country) %>%
  select(where(is.numeric))

## 5. Multicollinearity & skewness diagnostics ----
# Baseline linear model
lm_baseline <- lm(LifeExpectancy ~ ., data = train_data %>% select(-Country))
vif_values  <- car::vif(lm_baseline)

# Skewness
skewness_values <- sapply(train_numeric, skewness, na.rm = TRUE)
skewed_vars     <- names(skewness_values[abs(skewness_values) > 1])

message("Mean |skewness| (before): ", round(mean(abs(skewness_values)), 3))
message("Variables with severe skewness: ", paste(skewed_vars, collapse = ", "))

## 6. Transformations: log/logit ----
logit_transform <- function(x) {
  # Assumes x is 0–100 or 0–1. If 0–100, scale to 0–1 first.
  x <- as.numeric(x)
  if (max(x, na.rm = TRUE) > 1) x <- x / 100
  eps <- 1e-3
  x <- pmax(pmin(x, 1 - eps), eps)
  log(x / (1 - x))
}

train_transformed <- train_data
test_transformed  <- test_data

for (col in skewed_vars) {
  if (!col %in% names(train_transformed)) next
  
  x_train <- train_transformed[[col]]
  x_test  <- test_transformed[[col]]
  
  # Heuristic: if it's bounded and within [0, 100], treat as percentage
  if (all(x_train >= 0, na.rm = TRUE) && max(x_train, na.rm = TRUE) <= 100) {
    train_transformed[[paste0(col, "_logit")]] <- logit_transform(x_train)
    test_transformed[[paste0(col, "_logit")]]  <- logit_transform(x_test)
  } else {
    train_transformed[[paste0(col, "_log")]] <- log(x_train + 1e-3)
    test_transformed[[paste0(col, "_log")]]  <- log(x_test + 1e-3)
  }
  
  # Optionally keep the original too; for simplicity here we keep both.
}

## Rebuild numeric matrices for modeling ----
train_x <- train_transformed %>%
  select(-Country, -LifeExpectancy)
test_x  <- test_transformed %>%
  select(-Country, -LifeExpectancy)

train_y <- train_transformed$LifeExpectancy
test_y  <- test_transformed$LifeExpectancy

train_x_matrix <- as.matrix(train_x)
test_x_matrix  <- as.matrix(test_x)
train_y_vec    <- as.numeric(train_y)
test_y_vec     <- as.numeric(test_y)

## 7. Correlation plot (for figures/) ----
if (!dir.exists("figures")) dir.create("figures")

corr_mat <- cor(train_x, use = "pairwise.complete.obs")
png("figures/correlation_plot.png", width = 900, height = 900)
corrplot(corr_mat, method = "color", tl.cex = 0.6)
dev.off()

## 8. Elastic Net: alpha grid search + cross-validation ----
alpha_grid <- seq(0, 1, by = 0.01)
cv_list    <- vector("list", length(alpha_grid))
cv_min     <- numeric(length(alpha_grid))

for (i in seq_along(alpha_grid)) {
  cv_fit <- cv.glmnet(
    x = train_x_matrix,
    y = train_y_vec,
    family = "gaussian",
    alpha = alpha_grid[i],
    nfolds = 10,
    type.measure = "mse",
    standardize = TRUE
  )
  cv_list[[i]] <- cv_fit
  cv_min[i]    <- min(cv_fit$cvm)
}

best_i      <- which.min(cv_min)
best_alpha  <- alpha_grid[best_i]
cv_best     <- cv_list[[best_i]]
lambda_min  <- cv_best$lambda.min
lambda_1se  <- cv_best$lambda.1se

message("Best alpha: ", best_alpha)
message("Lambda min: ", signif(lambda_min, 4), "; Lambda 1se: ", signif(lambda_1se, 4))

png("figures/elasticnet_cv.png", width = 800, height = 600)
plot(cv_best)
dev.off()

# Fit models at lambda.min and lambda.1se
enet_min <- glmnet(
  x = train_x_matrix,
  y = train_y_vec,
  family = "gaussian",
  alpha = best_alpha,
  lambda = lambda_min,
  standardize = TRUE
)

enet_1se <- glmnet(
  x = train_x_matrix,
  y = train_y_vec,
  family = "gaussian",
  alpha = best_alpha,
  lambda = lambda_1se,
  standardize = TRUE
)

## 9. Principal Component Regression (PCR) ----
pcr_train <- data.frame(life_expectancy = train_y, train_x)
pcr_test  <- data.frame(life_expectancy = test_y, test_x)

pca_result  <- prcomp(train_x, center = TRUE, scale. = TRUE)
eigenvalues <- pca_result$sdev^2
prop_var    <- eigenvalues / sum(eigenvalues)
cumvar      <- cumsum(prop_var)

png("figures/pcr_scree_plot.png", width = 800, height = 600)
plot(prop_var, type = "b", xlab = "Principal Component",
     ylab = "Proportion of Variance Explained", main = "Scree Plot")
abline(v = 5, col = "gray")
dev.off()

# In your report you chose k = 5
k_final <- 5

pcr_model <- pcr(
  life_expectancy ~ .,
  data  = pcr_train,
  ncomp = k_final,
  scale = TRUE
)

## 10. Test performance: PCR vs Elastic Net ----
# PCR
pcr_pred_test <- as.numeric(predict(pcr_model, newdata = pcr_test, ncomp = k_final))

rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae  <- function(y, yhat) mean(abs(y - yhat))
r2   <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

pcr_rmse <- rmse(test_y_vec, pcr_pred_test)
pcr_mae  <- mae(test_y_vec, pcr_pred_test)
pcr_r2   <- r2(test_y_vec, pcr_pred_test)

# Elastic Net (lambda.min)
enet_min_pred <- as.numeric(predict(enet_min, newx = test_x_matrix, s = lambda_min))
enet_min_rmse <- rmse(test_y_vec, enet_min_pred)
enet_min_mae  <- mae(test_y_vec, enet_min_pred)
enet_min_r2   <- r2(test_y_vec, enet_min_pred)

# Elastic Net (lambda.1se)
enet_1se_pred <- as.numeric(predict(enet_1se, newx = test_x_matrix, s = lambda_1se))
enet_1se_rmse <- rmse(test_y_vec, enet_1se_pred)
enet_1se_mae  <- mae(test_y_vec, enet_1se_pred)
enet_1se_r2   <- r2(test_y_vec, enet_1se_pred)

perf_table <- data.frame(
  Model      = c("PCR (5 PCs)", "Elastic Net (lambda.min)", "Elastic Net (lambda.1se)"),
  Test_RMSE  = c(pcr_rmse, enet_min_rmse, enet_1se_rmse),
  Test_MAE   = c(pcr_mae,  enet_min_mae,  enet_1se_mae),
  Test_R2    = c(pcr_r2,   enet_min_r2,   enet_1se_r2)
)

print(perf_table)

## 11. Residual stability vs income (optional) ----
# If GNI_PerCapita_log (or similar) exists, you can test the income bias
gni_col <- grep("GNI", names(test_x), value = TRUE)[1]

if (!is.na(gni_col)) {
  # heuristic: if log-transformed exists, use it; otherwise log the raw one
  gni_raw <- test_x[[gni_col]]
  gni_log <- log(gni_raw + 1e-3)
  
  resid_df <- data.frame(
    GNI_log        = gni_log,
    pcr_resid      = test_y_vec - pcr_pred_test,
    enet_min_resid = test_y_vec - enet_min_pred,
    enet_1se_resid = test_y_vec - enet_1se_pred
  )
  
  p_pcr      <- summary(lm(pcr_resid ~ GNI_log, data = resid_df))$coefficients[2, 4]
  p_enet_min <- summary(lm(enet_min_resid ~ GNI_log, data = resid_df))$coefficients[2, 4]
  p_enet_1se <- summary(lm(enet_1se_resid ~ GNI_log, data = resid_df))$coefficients[2, 4]
  
  message("Residual ~ log(GNI) p-values:")
  message("  PCR:           ", signif(p_pcr, 4))
  message("  Elastic Net min:  ", signif(p_enet_min, 4))
  message("  Elastic Net 1se:  ", signif(p_enet_1se, 4))
}

## 12. Predict life expectancy for new countries ----
# pred_data should have the same predictor structure as train_x
# Apply the same transformations as above (simplified here)

pred_merged <- pred_data %>%
  inner_join(health_data, by = "Country")

# Simple median imputation using train medians
for (col in names(train_medians)) {
  if (col %in% names(pred_merged)) {
    pred_merged[[col]][is.na(pred_merged[[col]])] <- train_medians[[col]]
  }
}

pred_x <- pred_merged %>%
  select(-Country) %>%
  select(where(is.numeric))

# Align columns to training design (drop unknowns, fill missing with 0)
missing_cols <- setdiff(names(train_x), names(pred_x))
for (col in missing_cols) pred_x[[col]] <- 0
pred_x <- pred_x[, names(train_x)]

pred_pcr <- as.numeric(predict(pcr_model, newdata = data.frame(life_expectancy = 0, pred_x), ncomp = k_final))
pred_table <- data.frame(
  Country           = pred_merged$Country,
  LifeExpectancy_PCR = round(pred_pcr, 2)
)

print(pred_table)

## 13. Save performance figure ----
png("figures/pcr_vs_elasticnet_performance.png", width = 800, height = 600)
ggplot(perf_table, aes(x = Model, y = Test_RMSE)) +
  geom_col() +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Test RMSE: PCR vs Elastic Net",
    y = "RMSE (years)",
    x = NULL
  )
dev.off()

message("Analysis complete.")
