load("data/d_AI_full_causality.RData")

library(dplyr)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)
library(ggplot2)

d <-
  d |>
  mutate_if(is.factor, as.numeric) |>
  mutate(
    TRT_USvEU = TRT_USvEU - 1,
    Employment = Employment - 1,
    LEarnCode = LearnCode - 1,
    EdLevel = ifelse(EdLevel >= 2, 1, 0),
    DataScientist = DataScientist - 1,
  ) |>
  select(-JobSat_NS) |>
  as.data.table()

features_base <- c(
  "MainBranch", "Age",
  "DataScientist",
  "Employment", "LearnCode", "YearsCode",
  "AI_impr_comp_prod_effc", "AI_complex_confidence", "AI_ethics_gov",
  "AI_use_con_lea_sea", "AI_use_production", "AI_use_doc_test",
  "AI_use_proj_plan"
)

formula_flex <- formula(
  " ~ -1 + poly(Age, 2, raw=TRUE) +
  poly(DataScientist, 2, raw=TRUE) +
  poly(Employment, 2, raw=TRUE) +
  poly(LearnCode, 2, raw=TRUE) + MainBranch + YearsCode +
  AI_impr_comp_prod_effc + AI_complex_confidence + AI_ethics_gov +
  AI_use_con_lea_sea + AI_use_production + AI_use_doc_test + AI_use_proj_plan"
)

features_flex <- data.frame(model.matrix(formula_flex, d))

data_dml_base_iv <-
  DoubleMLData$new(
    d,
    y_col = "JobSat",
    d_cols = "TRT_USvEU",
    x_cols = features_base,
    z_cols = "EdLevel"
  )
data_dml_base_iv

model_data <-
  data.table(
    "JobSat" = d[, JobSat],
    "TRT_USvEU" = d[, TRT_USvEU],
    "EdLevel" = d[, EdLevel],
    features_flex
  )
data_dml_flex_iv <-
  DoubleMLData$new(
    model_data,
    y_col = "JobSat",
    d_cols = "TRT_USvEU",
    z_cols = "EdLevel"
  )

set.seed(666)
lasso <- lrn("regr.cv_glmnet", nfolds = 5, s = "lambda.min")
lasso_class <- lrn("classif.cv_glmnet", nfolds = 5, s = "lambda.min")
dml_iivm_lasso <-
  DoubleMLIIVM$new(
    data_dml_flex_iv,
    ml_g = lasso,
    ml_m = lasso_class,
    ml_r = lasso_class,
    n_folds = 5,
    trimming_threshold = 0.01,
    subgroups = list(
      always_takers = FALSE,
      never_takers = TRUE
    )
  )
dml_iivm_lasso$fit()
dml_iivm_lasso$summary()

set.seed(666)
trees <- lrn("regr.rpart")
trees_class <- lrn("classif.rpart")
dml_iivm_tree <-
  DoubleMLIIVM$new(
    data_dml_base_iv,
    ml_g = trees,
    ml_m = trees_class,
    ml_r = trees_class,
    n_folds = 5,
    trimming_threshold = 0.01,
    subgroups = list(
      always_takers = FALSE,
      never_takers = TRUE
    )
  )
dml_iivm_tree$set_ml_nuisance_params(
  "ml_g0", "TRT_USvEU",
  list(cp = 0.0016, minsplit = 74)
)
dml_iivm_tree$set_ml_nuisance_params(
  "ml_g1", "TRT_USvEU",
  list(cp = 0.0018, minsplit = 70)
)
dml_iivm_tree$set_ml_nuisance_params(
  "ml_m", "TRT_USvEU",
  list(cp = 0.0028, minsplit = 167)
)
dml_iivm_tree$set_ml_nuisance_params(
  "ml_r1", "TRT_USvEU",
  list(cp = 0.0576, minsplit = 55)
)
dml_iivm_tree$fit()
dml_iivm_tree$summary()

set.seed(666)
randomForest <- lrn("regr.ranger")
randomForest_class <- lrn("classif.ranger")
dml_iivm_forest <-
  DoubleMLIIVM$new(
    data_dml_base_iv,
    ml_g = randomForest,
    ml_m = randomForest_class,
    ml_r = randomForest_class,
    n_folds = 5,
    trimming_threshold = 0.01,
    subgroups = list(
      always_takers = FALSE,
      never_takers = TRUE
    )
  )
dml_iivm_forest$set_ml_nuisance_params(
  "ml_g0", "TRT_USvEU",
  list(max.depth = 6, mtry = 4, min.node.size = 7)
)
dml_iivm_forest$set_ml_nuisance_params(
  "ml_g1", "TRT_USvEU",
  list(max.depth = 6, mtry = 3, min.node.size = 5)
)
dml_iivm_forest$set_ml_nuisance_params(
  "ml_m", "TRT_USvEU",
  list(max.depth = 6, mtry = 3, min.node.size = 6)
)
dml_iivm_forest$set_ml_nuisance_params(
  "ml_r1", "TRT_USvEU",
  list(max.depth = 4, mtry = 7, min.node.size = 6)
)
dml_iivm_forest$fit()
dml_iivm_forest$summary()

set.seed(666)
boost <- lrn("regr.xgboost", objective = "reg:squarederror")
boost_class <- lrn(
  "classif.xgboost",
  objective = "binary:logistic",
  eval_metric = "logloss"
)
dml_iivm_boost <-
  DoubleMLIIVM$new(
    data_dml_base_iv,
    ml_g = boost,
    ml_m = boost_class,
    ml_r = boost_class,
    n_folds = 5,
    trimming_threshold = 0.01,
    subgroups = list(
      always_takers = FALSE,
      never_takers = TRUE
    )
  )

dml_iivm_boost$set_ml_nuisance_params(
  "ml_g0", "TRT_USvEU",
  list(nrounds = 9, eta = 0.1, objective = "reg:squarederror", verbose = 0)
)
dml_iivm_boost$set_ml_nuisance_params(
  "ml_g1", "TRT_USvEU",
  list(nrounds = 33, eta = 0.1, objective = "reg:squarederror", verbose = 0)
)
dml_iivm_boost$set_ml_nuisance_params(
  "ml_m", "TRT_USvEU",
  list(
    nrounds = 12, eta = 0.1, objective = "binary:logistic",
    eval_metric = "logloss", verbose = 0
  )
)
dml_iivm_boost$set_ml_nuisance_params(
  "ml_r1", "TRT_USvEU",
  list(
    nrounds = 25, eta = 0.1, objective = "binary:logistic",
    eval_metric = "logloss", verbose = 0
  )
)
dml_iivm_boost$fit()
dml_iivm_boost$summary()

confints <- rbind(
  dml_iivm_lasso$confint(), dml_iivm_forest$confint(),
  dml_iivm_tree$confint(), dml_iivm_boost$confint()
)
estimates <- c(
  dml_iivm_lasso$coef, dml_iivm_forest$coef,
  dml_iivm_tree$coef, dml_iivm_boost$coef
)
result_iivm <-
  data.table(
    "model" = "IIVM",
    "ML" = c("lasso", "random forest", "decision trees", "xgboost"),
    "Estimate" = estimates,
    "lower" = confints[, 1],
    "upper" = confints[, 2]
  )
result_iivm |>
  ggplot(aes(x = ML, y = Estimate, color = ML)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower, ymax = upper, color = ML)) +
  geom_hline(yintercept = 0, color = "grey") +
  theme_minimal() +
  labs(
    x = "Algorithm", y = "Coefficient Estimate with CI",
    title = "DoubleML Local average treatment effects",
    subtitle = "with an instrumental variable",
    caption = "Target Feauture: Job Satisfaction
    Treatment: Residing in the US. vs. Europe
    Instrumental: Education Level: Low (no university degree) vs. High (univeristy degree & higher)"
  ) +
  theme(
    axis.text.x = element_text(angle = 90), legend.position = "none",
    text = element_text(size = 14)
  )
