library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

test_data <- vroom("test.csv")
train_data <- vroom("train.csv") |>
  mutate(ACTION = factor(ACTION))

my_recipe <- recipe(ACTION~., data = train_data) |>
  step_mutate_at(all_numeric_predictors(), fn = factor) |>
  step_other(all_nominal_predictors(), threshold = 0.001) |>
  step_dummy(all_nominal_predictors()) 

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
