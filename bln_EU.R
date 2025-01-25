load("data/d_AI_full_causality.RData")
colnames(d)

library(dplyr)

d_eu <-
  d |>
  mutate_if(is.factor, as.numeric) |>
  mutate(
    TRT_USvEU = TRT_USvEU - 1,
    DataScientist = DataScientist - 1,
    LearnCode = LearnCode - 1,
    Employment = Employment - 1
  ) |>
  filter(TRT_USvEU == 1) |>
  select(-c(TRT_USvEU, JobSat_NS)) |>
  as.data.frame()

library(parallel)
use_cores <- detectCores() - 1
cl <- makeCluster(use_cores)

library(bnlearn)

bl_tiers <-
  tiers2blacklist(
    list(
      names(d_eu[1:7]),
      names(d_eu[9:15])
    )
  ) |> as.data.frame()
bl_set <- set2blacklist(names(d_eu[1:7])) |> as.data.frame()
bl_js <- tiers2blacklist(
  list(
    names(d_eu[c(1:7, 9:15)]),
    names(d_eu[8])
  )
) |> as.data.frame()
bl <- bind_rows(bl_set, bl_tiers, bl_js) |> as.matrix()

set.seed(666)
bn_op_eu <-
  tabu(
    x = d_eu,
    blacklist = bl,
    debug = TRUE
  )
qgraph::qgraph(bn_op_eu, vsize = 9, label.cex = 2, layout = "circle")

set.seed(666)
boot_res_eu <- boot.strength(
  data = d_eu,
  R = 10000,
  algorithm = "tabu",
  algorithm.args = list(
    blacklist = bl
  ),
  cluster = cl
)
avgnet_threshold_eu <- averaged.network(boot_res_eu, threshold = .99)

qgraph::qgraph(
  avgnet_threshold_eu,
  vsize = 9, label.cex = 2,
  layout = "circle"
)

stopCluster(cl)

path_model_syntax_eu <- bnpa::gera.pa.model(avgnet_threshold_eu, d_eu)

library(lavaan)
sem_fit_eu <-
  sem(
    path_model_syntax_eu, d_eu,
    estimator = "mlr",
    mimic = "mplus",
    std.ov = TRUE,
  )
summary(sem_fit_eu, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)

path_model_labels <-
  list(
    MainBranch = "Main Branch",
    Age = "Age",
    Employment = "Employment",
    EdLevel = "Education Level",
    LearnCode = "Learned to code\nformally or not",
    YearsCode = "Years of professional coding",
    DataScientist = "Data Scientist or any other type of deveoper",
    JobSat = "Job Satisfaction",
    AI_impr_comp_prod_effc = "Thinking that\nAI Improves\nCompany\nProductivity\n&\nEfficiency",
    AI_complex_confidence = "Confidence in\nAI Product\nDevelopment\nfor\nComplex\nTasks",
    AI_ethics_gov = "Attitudes in AI \nEthics & Governance",
    AI_use_con_lea_sea = "AI Usage in \nContent Generation, \nLearning & Search",
    AI_use_production = "AI Usage in \nProduction",
    AI_use_doc_test = "AI Usage in \nDocumenting \n& Testing",
    AI_use_proj_plan = "AI Usage in \nProject Planning"
  )

lavaanPlot::lavaanPlot(
  model = sem_fit_eu,
  graph_options = list(
    label = "PathModelEurope",
    rankdir = "LR",
    labelloc = "t"
  ),
  node_options = list(shape = "box", fontname = "Roboto"),
  edge_options = list(color = "grey"), stars = c("regress"),
  stand = TRUE, coefs = TRUE,
  labels = path_model_labels
)

eu_igraph <- bnviewer::bn.to.igraph(avgnet_threshold_eu)
eu_remaining_archs_tibble <- avgnet_threshold_eu[["arcs"]] |> as_tibble()
eu_boot_res_tibble <- boot_res_eu |> as_tibble()
eu_remaining_archs <- eu_boot_res_tibble |>
  semi_join(eu_remaining_archs_tibble, by = c("from", "to"))
library(igraph)
E(eu_igraph)$weight <- eu_remaining_archs$strength
sort(degree(eu_igraph, mode = "in"), decreasing = TRUE) |>
  as.data.frame() |>
  tibble::rownames_to_column()
sort(degree(eu_igraph, mode = "out"), decreasing = TRUE) |>
  as.data.frame() |>
  tibble::rownames_to_column()
