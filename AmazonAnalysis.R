library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(discrim)

test_data <- vroom("./test.csv")
train_data <- vroom("./train.csv") |>
  mutate(ACTION = factor(ACTION))

my_recipe <- recipe(ACTION~., data = train_data) |>
  step_mutate_at(all_numeric_predictors(), fn = factor) |>
  step_other(all_nominal_predictors(), threshold = 0.001) |>
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_normalize(all_nominal_predictors())

prepped_recipe <- prep(my_recipe)
show <- bake(prepped_recipe, new_data = train_data)

#########################################################################################

# #logistic regression
# 
# logreg_mod <- logistic_reg() |>
#   set_engine("glm")
# 
# #Workflow and fit
# logreg_workflow <- workflow() |>
#   add_recipe(my_recipe) |>
#   add_model(logreg_mod) |>
#   fit(data=train_data)
# 
# #Make Predictions
# amazon_preds <- predict(logreg_workflow,
#                         new_data = test_data,
#                         type = "prob")
# 
# ## Format predictions for Kaggle
# kaggle <- amazon_preds|>
#   bind_cols(test_data) |>
#   select(id, .pred_1) |>
#   rename(Action = .pred_1) |>
#   rename(Id = id)
# 
# ##write out file
# vroom_write(x = kaggle, file = "./AmazonLogReg.csv", delim=",")
# 
# ##########################################################################################
# 
# #Penalized Logistic Regression
# 
# pen_mod <- logistic_reg(mixture= tune(), penalty = tune()) |>
#   set_engine("glmnet")
# 
# #workflow
# pen_workflow <- workflow() |>
#   add_recipe(my_recipe) |>
#   add_model(pen_mod)
# 
# #grid of values to tune
# pen_tuning_grid <- grid_regular(penalty(),
#                                 mixture(),
#                                 levels = 10)
# 
# #split data for CV
# folds <- vfold_cv(train_data, v = 5, repeats = 1)
# 
# #Run the CV
# CV_results <- pen_workflow |>
#   tune_grid(resamples = folds,
#             grid = pen_tuning_grid,
#             metrics = metric_set(roc_auc, f_meas, sens, recall, 
#                                   precision, accuracy))
# #Find best tuning parameters
# bestTune <- CV_results |>
#   select_best(metric = "roc_auc")
# 
# #Finalize the workflow
# pen_final_wf <- pen_workflow |>
#   finalize_workflow(bestTune) |>
#   fit(data = train_data)
# 
# #Predict
# pen_preds = predict(pen_final_wf, 
#                     new_data = test_data, type = "prob")
# 
# ## Format predictions for Kaggle
# kaggle <- pen_preds|>
#   bind_cols(test_data) |>
#   select(id, .pred_1) |>
#   rename(Action = .pred_1) |>
#   rename(Id = id)
# 
# ##write out file
# vroom_write(x = kaggle, file = "./AmazonPenLog.csv", delim=",")
# 
# #####################################################################################
# 
# #knn model
# knn_model <- nearest_neighbor(neighbors = 180) |>
#   set_mode('classification') |>
#   set_engine('kknn')
# 
# #set workflow
# knn_wf <- workflow() |>
#   add_recipe(my_recipe) |>
#   add_model(knn_model) |>
#   fit(data = train_data)
# 
# #Predict
# knn_preds = predict(knn_wf, 
#                     new_data = test_data, type = "prob")
# 
# ## Format predictions for Kaggle
# knn_kaggle <- knn_preds|>
#   bind_cols(test_data) |>
#   select(id, .pred_1) |>
#   rename(Action = .pred_1) |>
#   rename(Id = id)
# 
# ##write out file
# vroom_write(x = knn_kaggle, file = "./Amazonknn.csv", delim=",")

###################################################################################

# #Random Forest
# forest_mod <- rand_forest(mtry = tune(),
#                           min_n = tune(),
#                           trees = 500) |>
#   set_engine("ranger") |>
#   set_mode("classification")
# 
# ## Create a workflow with recipe
# forest_wf <- workflow() |>
#   add_recipe(my_recipe) |>
#   add_model(forest_mod)
# 
# ## Set up grid and tuning values
# forest_tuning_params <- grid_regular(mtry(range = c(1,9)),
#                                      min_n(),
#                                      levels = 5)
# 
# ##Split data for CV
# forest_folds <- vfold_cv(train_data, v = 5, repeats = 1)
# 
# ##Run the CV
# forest_CV_results <- forest_wf |>
#    tune_grid(resamples = forest_folds,
#              grid = forest_tuning_params,
#              metrics = metric_set(roc_auc, f_meas, sens, recall, 
#                                   precision, accuracy))
# #Find best tuning parameters
#  forest_best_tune <- forest_CV_results |>
#    select_best(metric = "roc_auc")
# 
# ##finalize the workflow and fit it
# forest_final <- forest_wf |>
#   finalize_workflow(forest_best_tune) |>
#   fit(data = train_data)
# 
# ##predict
# forest_preds <- forest_final |>
#   predict(new_data = test_data, type = "prob")
# 
# 
# ## Format Predictions for Kaggle
# forest_kaggle <- forest_preds|>
#   bind_cols(test_data) |>
#   select(id, .pred_1) |>
#   rename(Action = .pred_1) |>
#   rename(Id = id)
# 
# ##write out file
# vroom_write(x = forest_kaggle, file = "./Amazonforest.csv", delim=",")

#########################################################################################
#Naive Bayes Model

nb_mod <- naive_Bayes(Laplace= tune(), smoothness = tune()) |>
  set_mode("classification") |>
  set_engine("naivebayes")

nb_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(nb_mod)

## Set up grid and tuning values
nb_tuning_params <- grid_regular(Laplace(),
                                     smoothness(),
                                     levels = 5)

##Split data for CV
nb_folds <- vfold_cv(train_data, v = 5, repeats = 1)

##Run the CV
nb_CV_results <- nb_wf |>
  tune_grid(resamples = nb_folds,
            grid = nb_tuning_params,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                 precision, accuracy))
#Find best tuning parameters
nb_best_tune <- nb_CV_results |>
  select_best(metric = "roc_auc")

##finalize the workflow and fit it
nb_final <- nb_wf |>
  finalize_workflow(nb_best_tune) |>
  fit(data = train_data)

##predict
nb_preds <- nb_final |>
  predict(new_data = test_data, type = "prob")


## Format Predictions for Kaggle
nb_kaggle <- nb_preds|>
  bind_cols(test_data) |>
  select(id, .pred_1) |>
  rename(Action = .pred_1) |>
  rename(Id = id)

##write out file
vroom_write(x = nb_kaggle, file = "./AmazonNB.csv", delim=",")
