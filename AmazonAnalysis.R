library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

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

#logistic regression

logreg_mod <- logistic_reg() |>
  set_engine("glm")

#Workflow and fit
logreg_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(logreg_mod) |>
  fit(data=train_data)

#Make Predictions
amazon_preds <- predict(logreg_workflow,
                        new_data = test_data,
                        type = "prob")

## Format predictions for Kaggle
kaggle <- amazon_preds|>
  bind_cols(test_data) |>
  select(id, .pred_1) |>
  rename(Action = .pred_1) |>
  rename(Id = id)

##write out file
vroom_write(x = kaggle, file = "./AmazonLogReg.csv", delim=",")

##########################################################################################

#Penalized Logistic Regression

pen_mod <- logistic_reg(mixture= tune(), penalty = tune()) |>
  set_engine("glmnet")

#workflow
pen_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(pen_mod)

#grid of values to tune
pen_tuning_grid <- grid_regular(penalty(),
                                mixture(),
                                levels = 10)

#split data for CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)

#Run the CV
CV_results <- pen_workflow |>
  tune_grid(resamples = folds,
            grid = pen_tuning_grid,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                  precision, accuracy))
#Find best tuning parameters
bestTune <- CV_results |>
  select_best(metric = "roc_auc")

#Finalize the workflow
pen_final_wf <- pen_workflow |>
  finalize_workflow(bestTune) |>
  fit(data = train_data)

#Predict
pen_preds = predict(pen_final_wf, 
                    new_data = test_data, type = "prob")

## Format predictions for Kaggle
kaggle <- pen_preds|>
  bind_cols(test_data) |>
  select(id, .pred_1) |>
  rename(Action = .pred_1) |>
  rename(Id = id)

##write out file
vroom_write(x = kaggle, file = "./AmazonPenLog.csv", delim=",")

#####################################################################################

#knn model
knn_model <- nearest_neighbor(neighbors = 180) |>
  set_mode('classification') |>
  set_engine('kknn')

#set workflow
knn_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(knn_model) |>
  fit(data = train_data)

#Predict
knn_preds = predict(knn_wf, 
                    new_data = test_data, type = "prob")

## Format predictions for Kaggle
knn_kaggle <- knn_preds|>
  bind_cols(test_data) |>
  select(id, .pred_1) |>
  rename(Action = .pred_1) |>
  rename(Id = id)

##write out file
vroom_write(x = knn_kaggle, file = "./Amazonknn.csv", delim=",")


