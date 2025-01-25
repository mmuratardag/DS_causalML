load("data/d_AI_full_causality.RData")

library(dplyr)
library(data.table)
d <-
  d |>
  mutate_if(is.factor, as.numeric) |>
  mutate(
    TRT_USvEU = TRT_USvEU - 1,
    Employment = Employment - 1,
    LearnCode = LearnCode - 1,
    EdLevel = ifelse(EdLevel >= 2, 1, 0),
    DataScientist = DataScientist - 1,
  ) |>
  select(-JobSat_NS) |>
  as.data.table()

library(sl3)
library(sherlock)

lrn_ranger100 <- Lrnr_ranger$new(num.trees = 100)
xgb_fast <- Lrnr_xgboost$new()
xgb_50 <- Lrnr_xgboost$new(nrounds = 50, max_depth = 6, eta = 0.001)
lrn_glm <- Lrnr_glm$new()
lrn_lasso <- Lrnr_glmnet$new()
lrn_ridge_interaction <- Lrnr_glmnet$new(alpha = 0)
lrn_enet.5_interaction <- Lrnr_glmnet$new(alpha = 0.5)

or_interactions <-
  list(
    c("TRT_USvEU", "Employment"),
    c("TRT_USvEU", "EdLevel")
  )
ps_learner_spec <- xgb_50
or_learner_spec <- list(
  lrn_ranger100, xgb_fast,
  list(or_interactions, lrn_lasso)
)
cate_learner_spec <- Lrnr_sl$new(
  learners = list(lrn_ranger100, xgb_fast),
  metalearner = Lrnr_cv_selector$new()
)
set.seed(666)
sherlock_results <-
  sherlock_calculate(
    data_from_case = d,
    baseline = c(
      "MainBranch", "Age", "Employment",
      "EdLevel", "LearnCode", "YearsCode",
      "DataScientist",
      "AI_impr_comp_prod_effc", "AI_complex_confidence", "AI_ethics_gov",
      "AI_use_con_lea_sea", "AI_use_production", "AI_use_doc_test",
      "AI_use_proj_plan"
    ),
    exposure = "TRT_USvEU",
    outcome = "JobSat",
    segment_by = c("EdLevel", "Employment"),
    cv_folds = 5L,
    ps_learner = ps_learner_spec,
    or_learner = or_learner_spec,
    cate_learner = cate_learner_spec
  )
print(sherlock_results)
plot(sherlock_results, plot_type = "cate")
