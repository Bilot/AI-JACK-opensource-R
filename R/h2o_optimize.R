# BILOT AI-jack H2O-module, optimize
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.

#' Optimizing GLM parameters.
#'
#' @param h2o_data a list of H2OFrames
#' @param set config object
#' @param predictors vector of explanatory variables
#' @param targets label variable
#' @param runid run ID
#'
#' @return best model
#'
#' @export

optimize_glm_parameters <- function(h2o_data,set,
                                    predictors,targets,runid){

  # (1) Make data: ----
  #h2o_data <- make_h2o_data(df)
  x_train <- h2o_data$full_train
  x_val <- h2o_data$val

  # (2) Get parameters: ----
  glm_params <- set$model$glm

  # (3) Optimize lambda: ----
  model <- h2o.glm(
    model_id = paste(runid,set$main$model_name_part,
                     targets, 'GLM',
                     get_datetime, sep="_"),
    x = predictors,
    y = targets,
    training_frame = x_train,
    validation_frame = x_val,
    family = glm_params$family,
    lambda_search = TRUE,
    lambda_min_ratio = glm_params$lambda_min_ratio,
    seed = set$model$random_seed,
    alpha = glm_params$alpha,
    nfolds = glm_params$nfold)

  # model if returned with best lambda used;
  # model@model$lambda_best

  #h2o.removeAll(timeout_secs = 15)

  return(model)
}

#' Optimizing DRF parameters
#'
#' @param h2o_data a list of H2OFrames
#' @param set config object
#' @param estimator either \code{'decisionTree'} or \code{'randomForest'}
#' @param metric evaluation metric
#' @param predictors vector of explanatory variables
#' @param targets label variable
#' @param runid run ID
#'
#' @return best model
#'
#' @export

optimize_drf_parameters <- function(h2o_data,set,estimator,metric,
                                    predictors,targets,runid){

  # (1) Make data: ----
  x_train <- h2o_data$train
  x_train_full <- h2o_data$full_train
  x_test <- h2o_data$test
  x_val <- h2o_data$val

  # (2) Get params: ----
  general <- set$model$general
  fixed <- set$model$param_grid[[estimator]]$fixed
  hyper <- set$model$param_grid[[estimator]]$hyper_params

  # estimator = 'randomForest'
  algorithm <- ifelse(estimator == 'decisionTree',
                      'randomForest',estimator)

  # (3) Optimize: ----
  grid <- h2o.grid(
    algorithm,
    grid_id = paste0(estimator,'_grid'),
    training_frame = x_train,
    validation_frame = x_test,
    x = predictors,
    y = targets,
    stopping_tolerance = general$stopping_tolerance,
    stopping_rounds = general$stopping_rounds,
    stopping_metric = metric,
    hyper_params = hyper(length(predictors)),
    search_criteria = list(
      strategy = general$strategy,
      max_runtime_secs = fixed$max_runtime_secs,
      max_models = fixed$max_models,
      seed = general$random_seed
    )
  )

  # (4) Get best parameters: ----
  ids <- unlist(grid@model_ids)
  best_params <- get_best_params(grid,ids, metric)

  # (5) Fit final model: ----
  model = h2o.randomForest(
    x = best_params$x,
    y = best_params$y,
    model_id = paste(runid,set$main$model_name_part,
                     best_params$y,
                     estimator,
                     get_datetime, sep="_"),
    training_frame = x_train_full,
    validation_frame = x_val,
    max_depth = best_params$max_depth,
    ntrees = best_params$ntrees,
    mtries = best_params$mtries,
    nbins = best_params$nbins,
    nbins_cats = best_params$nbins_cats,
    sample_rate = best_params$sample_rate,
    stopping_tolerance = best_params$stopping_tolerance,
    stopping_rounds = best_params$stopping_rounds,
    stopping_metric = best_params$stopping_metric
  )

  return(model)
}

#' Optimizing GBM parameters.
#'
#' @param h2o_data a list of H2OFrames
#' @param set config object
#' @param algorithm either \code{'gbm'} or \code{'xgboost'}
#' @param metric evaluation metric
#' @param predictors vector of explanatory variables
#' @param targets label variable
#' @param runid run ID
#'
#' @return best model
#'
#' @description
#' Optimazation is done in two steps: (1) maximum depth is optimized,
#' (2) the rest of the parameters are optimized. This procedure makes the
#' optimization problem more manageable.
#'
#' @export

optimize_gbm_parameters <- function(h2o_data,set,algorithm,metric,
                                    predictors,targets,runid){

  # (1) Make data: ----
  x_train <- h2o_data$train
  x_train_full <- h2o_data$full_train
  x_test <- h2o_data$test
  x_val <- h2o_data$val

  # (2) Get params: ----
  general <- set$model$general
  fixed <- set$model$param_grid[[algorithm]]$fixed
  hyper1 <- set$model$param_grid[[algorithm]]$hyper_params1
  hyper2 <- set$model$param_grid[[algorithm]]$hyper_params2

  # (3) Optimize depth: ----
  grid_gbm_step1 <- h2o.grid(
    hyper_params = hyper1(length(predictors)),
    search_criteria = list(strategy = "Cartesian"),
    algorithm = algorithm,
    grid_id = "depth_grid",
    x = predictors,
    y = targets,
    training_frame = x_train,
    validation_frame = x_test,
    ntrees = 200,
    learn_rate = 0.1,
    sample_rate = 0.8,
    col_sample_rate = 0.8,
    seed = general$random_seed,
    stopping_rounds = general$stopping_rounds,
    stopping_tolerance = general$stopping_tolerance,
    stopping_metric = metric,
    score_tree_interval = fixed$score_tree_interval
  )

  ids <- unlist(grid_gbm_step1@model_ids)
  scores <- score_models(set,ids,metric)

  topDepths <- grid_gbm_step1@summary_table$max_depth[1:3]
  minDepth <- min(as.numeric(topDepths))
  maxDepth <- max(as.numeric(topDepths))

  # (4) Optimize rest: ----
  grid_gbm_step2 <- h2o.grid(
    hyper_params = hyper2(minDepth, maxDepth, x_train),
    search_criteria = list(strategy = general$strategy,
                           max_runtime_secs = fixed$max_runtime_secs,
                           max_models = fixed$max_models,
                           seed = general$random_seed),
    algorithm = algorithm,
    grid_id = paste0("rest_grid_",algorithm),
    x = predictors,
    y = targets,
    training_frame = x_train,
    validation_frame = x_test,
    stopping_rounds = general$stopping_rounds,
    stopping_tolerance = general$stopping_tolerance,
    stopping_metric = metric,
    score_tree_interval = general$score_tree_interval
  )

  # (5) Get best parameters: ----
  ids <- unlist(grid_gbm_step2@model_ids)
  best_params <- get_best_params(grid_gbm_step2,ids, metric)

  # (6) Fit final model: ----
  if(algorithm == 'gbm'){
    model <- train_gbm_model(best_params,
                             x_train_full, x_val,
                             set,runid)
  }else{
    model <- train_xgboost_model(best_params,
                                 x_train_full, x_val,
                                 set,runid)
  }

  return(model)
}

#' Optimizing DeepLearning parameters
#'
#' @param h2o_data a list of H2OFrames
#' @param set config object
#' @param metric evaluation metric
#' @param predictors vector of explanatory variables
#' @param targets label variable
#' @param runid run ID
#'
#' @return best model
#'
#' @export

optimize_deeplearning_parameters <- function(h2o_data,set,metric,
                                             predictors,targets,runid){

  # (1) Make data: ----
  x_train <- h2o_data$train
  x_test <- h2o_data$test
  x_val <- h2o_data$val

  # (2) Get params: ----
  general <- set$model$general
  fixed <- set$model$param_grid$deeplearning$fixed
  hyper <- set$model$param_grid$deeplearning$hyper_params
  if(set$main$labeliscategory){
    hyper$distribution <- NULL
    hyper$tweedie_power <- NULL
  }

  # (3) Optimize: ----
  grid <- h2o.grid(
    "deeplearning",
    grid_id = "deeplearning_grid",
    training_frame = x_train,
    validation_frame = x_test,
    x = predictors,
    y = targets,
    score_validation_samples = fixed$score_validation_samples,
    score_duty_cycle = fixed$score_duty_cycle,
    adaptive_rate = fixed$adaptive_rate,
    momentum_start = fixed$momentum_start,
    momentum_stable = fixed$momentum_stable,
    momentum_ramp = fixed$momentum_ramp,
    max_w2 = fixed$max_w2,
    stopping_tolerance = general$stopping_tolerance,
    stopping_rounds = general$stopping_rounds,
    stopping_metric = metric,
    hyper_params = hyper,
    search_criteria = list(
      strategy = general$strategy,
      max_runtime_secs = fixed$max_runtime_secs,
      max_models = fixed$max_models,
      seed = general$random_seed
    )
  )

  # (4) Get best parameters: ----
  if(length(grid@model_ids) > 0){
    ids <- unlist(grid@model_ids)
    scores <- score_models(set, ids, metric)
    best_model <- h2o.getModel(names(scores)[1])
    best_params <- best_model@allparameters

    # (5) Refit: ----
    model <- h2o.deeplearning(
      x = predictors,
      y = targets,
      model_id = paste(runid,set$main$model_name_part,
                       targets, 'DeepLearnin',
                       get_datetime, sep="_"),
      training_frame = x_train,
      validation_frame = x_val,
      hidden = best_params$hidden,
      input_dropout_ratio = best_params$input_dropout_ratio,
      hidden_dropout_ratios = best_params$hidden_dropout_ratios,
      epochs = best_params$epochs,
      score_validation_samples = fixed$score_validation_samples,
      score_duty_cycle = fixed$score_duty_cycle,
      adaptive_rate = fixed$adaptive_rate,
      momentum_start = fixed$momentum_start,
      momentum_stable = fixed$momentum_stable,
      momentum_ramp = fixed$momentum_ramp,
      max_w2 = fixed$max_w2,
      stopping_tolerance = fixed$stopping_tolerance,
      stopping_rounds = fixed$stopping_rounds,
      stopping_metric = metric,
      rate = best_params$rate,
      rate_annealing = best_params$rate_annealing,
      l1 = best_params$l1,
      l2 = best_params$l2,
      activation = best_params$activation
    )

    return(model)
  }else{
    return('All parameters failed.')
  }
}

#' Return best parameters for given models
#'
#' @param grid H2O model grid
#' @param model_ids model-ids
#' @param metric evaluation metric
#'
#' @return best parameters
#'
#' @export

get_best_params <- function(grid, model_ids, metric){

  scores <- sapply(model_ids, function(ii){
    perf = h2o.performance(h2o.getModel(ii),valid=T)
    x = names(perf@metrics)
    z = grep(metric,x,ignore.case = T,value = T)[1]
    as.numeric(perf@metrics[z])
  })

  best <- which.max(scores)
  best_model <- h2o.getModel(names(best))
  best_criterion <- scores[best]
  best_params = best_model@allparameters

  return(best_params)
}

# WRAPPERS: ----

#' Wrapper Function for optimizing H2O models
#'
#' @param df list of H2OFrames
#' @param set config object
#' @param estimator which model algorithm should be optimized?
#' @param runid run ID
#'
#' @return: optimized model and performance of the best model
#'
#' @description
#' Allowed algorithms: \code{'glm'}, \code{'gbm'}, \code{'xgboost'},
#' \code{'decisionTree'}, \code{'randomForest'}, \code{'deeplearning'}, 
#' and \code{'automl'}.
#'
#' @export

optimize_h2o_model <- function(df, set, estimator, runid){

  # (1) Get parameters: ----
  general <- set$model$general
  cols <- names(main$splitted$value$train)
  predictors <- setdiff(cols,set$model$cols_not_included)
  targets <- set$model$label
  metric <- fix_metric(set)

  # (2) Start Optimizing: ----

  # (2.1) Optimize GLM: ----
  if(grepl('glm',estimator,ignore.case = T)){
    best_model <- optimize_glm_parameters(df,set,
                                          predictors,targets,runid)
  }
  # (2.2) Optimize DeepLearning: ----
  if(grepl('deep',estimator,ignore.case = T)){
    best_model <- optimize_deeplearning_parameters(df,set,metric,
                                                   predictors,targets,
                                                   runid)
  }
  # (2.3) Optimize boosting: ----
  if(grepl('gbm|xgboost',estimator,ignore.case = T)){
    best_model <- optimize_gbm_parameters(df,set,estimator,
                                          metric,predictors,targets,
                                          runid)
  }
  # (2.4) Optimize trees: ----
  if(grepl('forest|tree',estimator,ignore.case = T)){
    best_model <- optimize_drf_parameters(df,set,estimator,
                                          metric,predictors,targets,
                                          runid)
  }

  # (3) Score model: ----
  perf <- h2o.performance(best_model,valid = T)
  best_criterion <- perf@metrics[grep(metric,
                                      names(perf@metrics),
                                      ignore.case = T)][1]

  # (4) Return best model: ----
  return(
    list(
      best_model = best_model,
      score = best_criterion
    )
  )
}
