load("data/d_AI_full_causality.RData")

library(tidyverse)
library(grf)

d <-
  d |>
  mutate_if(is.factor, as.numeric) |>
  mutate(
    TRT_USvEU = TRT_USvEU - 1,
    DataScientist = DataScientist - 1,
    Employment = Employment - 1,
    LearnCode = LearnCode - 1
  )

d |>
  group_by(TRT_USvEU) |>
  summarise(
    JobSat_median = median(JobSat),
    JobSat_mean = mean(JobSat),
    JobSat_sd = sd(JobSat),
    .groups = "drop"
  )

W <- d$TRT_USvEU
Y <- d$JobSat
X <-
  d |>
  select(
    AI_impr_comp_prod_effc,
    AI_complex_confidence,
    AI_ethics_gov,
    AI_use_con_lea_sea,
    AI_use_production,
    AI_use_doc_test,
    AI_use_proj_plan,
    DataScientist,
    MainBranch,
    Age,
    Employment,
    EdLevel,
    YearsCode
  ) |>
  as.data.frame()

fit <-
  causal_forest(
    X = X,
    Y = Y,
    W = W,
    num.trees = 1000,
    mtry = 4,
    sample.fraction = 0.7,
    seed = 666,
    ci.group.size = 1,
  )

average_treatment_effect(fit, target.sample = "all") |> round(3)
average_treatment_effect(fit, target.sample = "control") |> round(3) # US
average_treatment_effect(fit, target.sample = "treated") |> round(3) # Europe

xvars <- c(
  "AI_impr_comp_prod_effc",
  "AI_complex_confidence",
  "AI_ethics_gov",
  "AI_use_con_lea_sea",
  "AI_use_production",
  "AI_use_doc_test",
  "AI_use_proj_plan",
  "DataScientist",
  "MainBranch",
  "Age",
  "Employment",
  "EdLevel",
  "YearsCode"
)

var_names <- c(
  "AI_impr_comp_prod_effc" = "Thinking AI Improves Company Productivity & Efficiency",
  "AI_complex_confidence" = "Confidence in AI Product Development for Complex Tasks",
  "AI_ethics_gov" = "Attitudes in AI Ethics & Governance",
  "AI_use_con_lea_sea" = "AI Usage in Content Generation, Learning & Search",
  "AI_use_production" = "AI Usage in Production",
  "AI_use_doc_test" = "AI Usage in Documenting & Testing",
  "AI_use_proj_plan" = "AI Usage in Project Planning",
  "DataScientist" = "Data Scientist or other Developer Role",
  "MainBranch" = "Main Branch",
  "Age" = "Age",
  "Employment" = "Employment",
  "EdLevel" = "Education Level",
  "YearsCode" = "Years of Coding Experience"
)

cf_vi <-
  sort(setNames(variable_importance(fit), xvars)) |>
  as.data.frame() |>
  janitor::clean_names() |>
  rownames_to_column("variable") |>
  rename(importance = sort_set_names_variable_importance_fit_xvars) |>
  mutate(
    variable = str_replace_all(variable, var_names),
    variable = fct_reorder(variable, importance)
  ) |>
  ggplot(aes(x = variable, y = importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(
    subtitle = "Variable Importance predicting Job Satisfaction",
    x = "",
    y = "",
  ) +
  theme_bw()

pred_fun <- function(object, newdata, ...) {
  predict(object, newdata, ...)$predictions
}

library(patchwork)
library(hstats)
library(kernelshap)
library(shapviz)

pdps <- lapply(
  xvars,
  function(v) plot(partial_dep(fit, v, X = X, pred_fun = pred_fun))
)
wrap_plots(pdps, guides = "collect", ncol = 3) &
  ylab("Effect") &
  ylim(c(-0.06, 0.06)) &
  plot_annotation(
    title = "Casual Forest",
    subtitle = "Partial Dependence Plots",
    caption = "Target feature: Job Satisfaction\nResiding in the US or Europe used as the treatment"
  ) & theme_minimal()

Hstats <- hstats(fit, X = X, pred_fun = pred_fun, verbose = TRUE)
hstats_is <-
  Hstats |>
  plot() +
  labs(
    x = "",
    subtitle = "Interaction strength",
  ) + theme_minimal()

pd1 <-
  partial_dep(
    fit,
    v = "AI_impr_comp_prod_effc",
    X = X,
    BY = "AI_ethics_gov",
    pred_fun = pred_fun
  ) |>
  plot() +
  ylim(c(-.09, .08)) +
  labs(
    x = "",
    subtitle = "Stronger pairwise interactions between
    Thinking AI Improves Company Productivity & Efficiency BY
    Attitudes in AI Ethics & Governance"
  ) + theme_minimal()

pd2 <-
  partial_dep(
    fit,
    v = "AI_complex_confidence",
    X = X,
    BY = "AI_ethics_gov",
    pred_fun = pred_fun
  ) |>
  plot() +
  ylim(c(-.09, .11)) +
  labs(
    x = "",
    subtitle = "Confidence in AI Product Development for Complex Tasks BY
    Attitudes in AI Ethics & Governance"
  ) + theme_minimal()

pd3 <-
  partial_dep(
    fit,
    v = "AI_impr_comp_prod_effc",
    X = X,
    BY = "YearsCode",
    pred_fun = pred_fun
  ) |>
  plot() +
  ylim(c(-0.11, .06)) +
  labs(
    x = "",
    subtitle = "Thinking AI Improves Company Productivity & Efficiency BY
    Years of Coding Experience",
    caption = "Source: AI Survey 2024\nResiding in the US or Europe used as the treatment"
  ) + theme_minimal()


load("data/d_AI_full_imp_EGA_LPA.RData")
d$Profile <- df$Profile
set.seed(666)
d_ss <-
  d |>
  slice_sample(
    n = 150,
    by = c("Profile", "TRT_USvEU"),
    replace = FALSE
  )
d_profile <- d_ss |> select(Profile)
d_ss <- d_ss |> select(-Profile)
rm(df)
d <- d |> select(-Profile)

d_ss |>
  group_by(TRT_USvEU) |>
  tally()
d_profile |>
  group_by(Profile) |>
  tally()

W_ss <- d_ss$TRT_USvEU
Y_ss <- d_ss$JobSat
X_ss <-
  d_ss |>
  select(
    AI_impr_comp_prod_effc,
    AI_complex_confidence,
    AI_ethics_gov,
    AI_use_con_lea_sea,
    AI_use_production,
    AI_use_doc_test,
    AI_use_proj_plan,
    DataScientist,
    MainBranch,
    Age,
    Employment,
    EdLevel,
    YearsCode
  ) |>
  drop_na() |>
  as.data.frame()

fit_ss <-
  causal_forest(
    X = X_ss,
    Y = Y_ss,
    W = W_ss,
    num.trees = 1000,
    mtry = 4,
    sample.fraction = 0.7,
    seed = 666,
    ci.group.size = 1,
  )

p1_wf <-
  kernelshap(fit_ss, X = X_ss[1, ], bg_X = X_ss, pred_fun = pred_fun) |>
  shapviz() |>
  sv_waterfall() +
  labs(
    x = "",
    caption = "Profile: Optimizer"
  )
p2_wf <-
  kernelshap(fit_ss, X = X_ss[301, ], bg_X = X_ss, pred_fun = pred_fun) |>
  shapviz() |>
  sv_waterfall() +
  labs(
    x = "",
    caption = "Profile: Blindly Confident Evangelist"
  )
p3_wf <-
  kernelshap(fit_ss, X = X_ss[901, ], bg_X = X_ss, pred_fun = pred_fun) |>
  shapviz() |>
  sv_waterfall() +
  labs(
    x = "",
    caption = "Profile: Cautious, Limited, Moderate"
  )
p4_wf <-
  kernelshap(fit_ss, X = X_ss[751, ], bg_X = X_ss, pred_fun = pred_fun) |>
  shapviz() |>
  sv_waterfall() +
  labs(
    x = "",
    caption = "Profile: Holistic Adopter"
  )
p5_wf <-
  kernelshap(fit_ss, X = X_ss[151, ], bg_X = X_ss, pred_fun = pred_fun) |>
  shapviz() |>
  sv_waterfall() +
  labs(
    x = "",
    caption = "Profile: Disengaged"
  )

ks <- kernelshap(fit_ss, X = X_ss, pred_fun = pred_fun)
shap_values <- shapviz(ks)

shapP_vi <-
  sv_importance(shap_values) +
  labs(
    x = "",
    subtitle = "SHAP Variable Importance"
  ) + theme_minimal()

shapP_vi_bee <-
  sv_importance(shap_values, kind = "bee") +
  labs(
    x = "",
    subtitle = "SHAP Variable Importance | Bee swarm",
    caption = "Note that SHAP is performed on a
    smaller subset sample (1500 out of 35553) here
    due to computational burden"
  ) + theme_minimal()

sv_dependence(shap_values, v = xvars) +
  plot_layout(ncol = 3) &
  ylim(c(-0.04, 0.03)) & theme_minimal()

(cf_vi | shapP_vi) /
  shapP_vi_bee + plot_annotation(title = "Causal Forest")

hstats_is /
  (pd1 | pd2 | pd3) +
  plot_annotation(title = "Causal Forest")

(p1_wf | p2_wf | p3_wf) /
  (p4_wf | p5_wf | plot_spacer()) +
  plot_annotation(
    title = "Causal Forest",
    subtitle = "SHAP Waterfall",
    caption = "Each profile plot represents a randomly chosen row from the data frame"
  )
