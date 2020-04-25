# BILOT AI-jack H2O-module, train
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.

# TRAINING: -----

#' Training GLM model, using optimized lambda-value
#'
#' @param model_name model_id
#' @param set config object
#' @param h2o_data list of H2OFrames
#' @param best_lambda optimised lambda value
#'
#' @return GLM model
#'
#' @export

train_glm_model <- function(model_name, set, h2o_data, best_lambda){

  # (1) Get parameters:
  params = set$model$glm
  predictors <- setdiff(names(h2o_data$train),set$model$cols_not_included)
  targets <- set$model$label

  # (2) Fit with optimal lambda:
  model <- h2o.glm(
    model_id = model_name,
    x = predictors,
    y = targets,
    training_frame = h2o_data$full_train,
    validation_frame = h2o_data$val,
    family = params$family,
    lambda = best_lambda,
    lambda_search = FALSE,
    seed = set$model$random_seed,
    alpha = params$alpha)

  return(model)
}

#' Training a DRF model, using optimized parameters
#'
#' @param model_name model_id
#' @param h2o_data list of H2OFFrames
#' @param best_params optimised parameters
#'
#' @return DRF model
#'
#' @export

train_drf_model <- function(model_name, h2o_data, best_params){

  model = h2o.randomForest(
    x = best_params$x,
    y = best_params$y,
    model_id = model_name,
    training_frame = h2o_data$full_train,
    validation_frame = h2o_data$val,
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

#' Training GBM model, using optimized parameters.
#'
#' @param best_params optimised parameters
#' @param train_frame H2OFrame for training
#' @param val_frame H2OFrame for validation
#' @param set config object
#' @param runid execution ID
#'
#' @return GBM model
#'
#' @export

train_gbm_model <- function(best_params,train_frame,val_frame,
                            set,runid){

  model = h2o.gbm(
    x = best_params$x,
    y = best_params$y,
    model_id = paste(runid,set$main$model_name_part,
                     best_params$y, 'GBM',
                     get_datetime, sep="_"),
    training_frame = train_frame,
    validation_frame = val_frame,
    max_depth = best_params$max_depth,
    ntrees = best_params$ntrees,
    nbins = best_params$nbins,
    nbins_cats = best_params$nbins_cats,
    sample_rate = best_params$sample_rate,
    col_sample_rate = best_params$col_sample_rate,
    col_sample_rate_per_tree = best_params$col_sample_rate_per_tree,
    col_sample_rate_change_per_level = best_params$col_sample_rate_change_per_level,
    min_rows = best_params$min_rows,
    min_split_improvement = best_params$min_split_improvement,
    histogram_type = best_params$histogram_type,
    score_tree_interval = best_params$score_tree_interval,
    learn_rate = best_params$learn_rate,
    learn_rate_annealing = best_params$learn_rate_annealing,
    stopping_tolerance = best_params$stopping_tolerance,
    stopping_rounds = best_params$stopping_rounds,
    stopping_metric = best_params$stopping_metric
  )

  return(model)
}

#' Training XGBoost model, using optimized parameters
#'
#' @param best_params optimised parameters
#' @param train_frame H2OFrame for training
#' @param val_frame H2OFrame for validation
#' @param set config object
#' @param runid execution ID
#'
#' @return XGBOOST model
#'
#' @export

train_xgboost_model <- function(best_params,train_frame,val_frame,
                                set,runid){

  if(h2o.xgboost.available()){
    model = h2o.xgboost(
      x = best_params$x,
      y = best_params$y,
      model_id = paste(runid,set$main$model_name_part,
                       best_params$y, 'xgboost',
                       get_datetime, sep="_"),
      training_frame = train_frame,
      validation_frame = val_frame,
      max_depth = best_params$max_depth,
      ntrees = best_params$ntrees,
      sample_rate = best_params$sample_rate,
      col_sample_rate = best_params$col_sample_rate,
      col_sample_rate_per_tree = best_params$col_sample_rate_per_tree,
      min_rows = best_params$min_rows,
      min_split_improvement = best_params$min_split_improvement,
      score_tree_interval = best_params$score_tree_interval,
      learn_rate = best_params$learn_rate,
      stopping_tolerance = best_params$stopping_tolerance,
      stopping_rounds = best_params$stopping_rounds,
      stopping_metric = best_params$stopping_metric
    )
  }else{
    model = 'XGBoost not available'
  }

  return(model)
}

#' Training DeepLearning model, using optimized parameters
#'
#' @param model_name model_id
#' @param h2o_data list of H2OFFrames
#' @param best_params optimised parameters
#'
#' @return DeepLearning model
#'
#' @export

train_deeplearning_model <- function(model_name, h2o_data, best_params){

  general <- set$model$general
  fixed <- set$model$param_grid$deeplearning$fixed

  model = h2o.deeplearning(
    x = best_params$x,
    y = best_params$y,
    model_id = model_name,
    training_frame = h2o_data$train,
    hidden = best_params$hidden,
    input_dropout_ratio = best_params$input_dropout_ratio,
    hidden_dropout_ratios = best_params$hidden_dropout_ratios,
    epochs = best_params$epochs,
    rate = best_params$rate,
    rate_annealing = best_params$rate_annealing,
    l1 = best_params$l1,
    l2 = best_params$l2,
    activation = best_params$activation,
    distribution = best_params$distribution,
    tweedie_power = best_params$tweedie_power,
    score_validation_samples = fixed$score_validation_samples,
    score_duty_cycle = fixed$score_duty_cycle,
    adaptive_rate = fixed$adaptive_rate,
    momentum_start = fixed$momentum_start,
    momentum_stable = fixed$momentum_stable,
    momentum_ramp = fixed$momentum_ramp,
    max_w2 = fixed$max_w2,
    stopping_tolerance = general$stopping_tolerance,
    stopping_rounds = general$stopping_rounds,
    stopping_metric = general$stopping_metric
  )

  return(model)
}

#' Trainign H2O models using AutoML
#'
#' @param df list of H2OFrames
#' @param set config object
#' @param runid run ID
#'
#' @return a list of n best models (n is defines in set) and performance of the best models
#'
#' @export

train_automl_model <- function(df, set, runid){

  # (1) Get parameters: ----
  params <- set$model$general
  automl_params <- set$model$automl
  predictors <- setdiff(names(df$train),set$model$cols_not_included)
  targets <- set$model$label
  metric <- fix_metric(set)

  # (2) Start training: ----
  model_h2o_train <-
    h2o.automl(
      x = predictors,
      y = targets,
      training_frame = df$train,
      validation_frame = df$test,
      max_models = automl_params$max_num_models,
      seed = params$random_seed,
      max_runtime_secs = automl_params$max_runtime_secs,
      stopping_tolerance = params$stopping_tolerance,
      stopping_rounds = params$stopping_rounds,
      stopping_metric = metric,
      exclude_algos = automl_params$exclude
    )

  if(attributes(model_h2o_train)$leaderboard[1,1]==''){
    stop('increase max-time')
  }

  # (3) Output: ----

  # Number of models to return:
  n <- ifelse(automl_params$save_n_best_models >
                nrow(model_h2o_train@leaderboard),
              nrow(model_h2o_train@leaderboard),
              automl_params$save_n_best_models)

  # Estimators:
  algos <- sapply(1:n,function(ii){
    x = as.vector(model_h2o_train@leaderboard[ii,]$model_id)
    h2o.getModel(x)@algorithm
  })

  # Rename models:
  models <- list()
  for(ii in 1:n) {
    x = as.vector(model_h2o_train@leaderboard[ii,]$model_id)
    models[[ii]] = h2o.getModel(x)
    #models[[ii]]@model_id = paste(runid,targets,
    #                              'autoML',algos[ii],ii,
    #                              set$main$model_name_part,
    #                              get_datetime, sep="_")
    #names(models)[ii] = models[[ii]]@model_id
    names(models)[ii] = paste(runid,targets,
                              'autoML',algos[ii],ii,
                              set$main$model_name_part,
                              get_datetime, sep="_")
  }

  # Score models:
  ids <- names(models)
  scores <- sapply(ids, function(ii){
    perf = h2o.performance(models[[ii]], valid=T)
    as.numeric(perf@metrics[[metric]])
  })

  # (4) Return best models: ----
  best_models <- list()
  for(ii in names(models)){
    best_models[[ii]] <- list(
      best_model = models[[ii]],
      score = as.numeric(scores[ii])
    )
  }
  return(best_models)
}

#' Training a anomaly detection model, using isolationForest
#'
#' @param model_name model_id
#' @param h2o_data list of H2OFFrames
#' @param set config object
#'
#' @return isolationForest model
#'
#' @export

train_isoforest_model <- function(model_name, h2o_data, set){
  
  predictors <- setdiff(names(h2o_data$train),set$model$cols_not_included)
  params <- set$anomaly$isoForest
  
  model = h2o.isolationForest(
    training_frame = h2o_data$full_train,
    x = predictors,
    model_id = model_name,
    max_depth = params$max_depth,
    ntrees = params$ntrees,
    mtries = params$mtries,
    sample_rate = params$sample_rate,
    stopping_tolerance = params$stopping_tolerance,
    stopping_rounds = params$stopping_rounds,
    stopping_metric = params$stopping_metric,
    score_tree_interval = params$score_tree_interval
  )
  
  return(model)
}

#' Training a anomaly detection model, using autoencoder network
#'
#' @param model_name model_id
#' @param h2o_data list of H2OFFrames
#' @param set config object
#'
#' @return autoencoder model
#'
#' @export

train_autoencoder_model <- function(model_name, h2o_data, set){
  
  # (1) Make data: ----
  x_train <- h2o_data$full_train
  x_val <- h2o_data$val
  
  x <- setdiff(names(x_train),set$model$cols_not_included)
  params <- set$anomaly$autoencoder
  
  model <- h2o.deeplearning(
    x = x,
    model_id = model_name,
    training_frame = x_train,
    validation_frame = x_val,
    hidden = params$hidden,
    epochs = params$epochs,
    adaptive_rate = params$adaptive_rate,
    stopping_tolerance = params$stopping_tolerance,
    stopping_rounds = params$stopping_rounds,
    stopping_metric = params$stopping_metric,
    activation = params$activation,
    autoencoder = TRUE
  )
  
  return(model)
}

# SCORING: -----

#' Scoring models
#'
#' @param set config object
#' @param model_ids model identifiers
#' @param metric evaluation metric
#'
#' @return best model
#'
#' @export

score_models <- function(set, model_ids, metric){

  perf <- sapply(model_ids, function(ii){
    perf = h2o.performance(h2o.getModel(ii),valid=T)
    x = perf@metrics
    use = grep(metric,names(x),ignore.case = T, value = T)[1]
    as.numeric(perf@metrics[use])
  })

  sorted <- sort(perf,decreasing = set$main$labeliscategory)

  return(sorted)
}

# WRAPPERS: -----

#' Wrapper function for training final models.
#'
#' @param h2o_data a list of H2OFrames
#' @param set config object
#' @param models either a single model object or a list of models
#'
#' @return A list of final model objects
#'
#' @export

train_h2o_model <- function(h2o_data, set, models){

  library(foreach)

  if(!is.list(models)){
    models = list(models)
  }

  estimators <- sapply(models,function(x) x@algorithm)
  model_names <- as.character(sapply(models,function(x) x@model_id))

  time <- format(Sys.time(), "%d-%m-%Y %H:%M:%S")
  names(models) <- model_names

  time = format(Sys.time(), "%d-%m-%Y %H:%M:%S")

  best_models <- foreach(ii = model_names) %do% {
    # (1) Get algorithm from model object,
    #     parameters & time:
    best_model <- models[[ii]]
    estimator <- best_model@algorithm
    best_params <- best_model@allparameters

    # (2) Train model using full
    #     training data:
    print(paste('Training',estimator,'model...'))

    if(estimator == 'glm'){
      model <- train_glm_model(model_name = ii,
                               set = set,
                               h2o_data = h2o_data,
                               best_lambda = best_model@model$lambda_best)
    }
    if(estimator == 'drf'){
      model <- train_randomForest_model(ii,h2o_data,best_params)
    }
    if(estimator == 'deeplearning'){
      model <- train_deeplearning_model(ii,h2o_data,best_params)
    }
    if(estimator == 'gbm'){
      model <- train_gbm_model(ii,h2o_data,best_params)
    }
    if(h2o.xgboost.available() &
       estimator == 'xgboost'){
      model <- train_xgboost_model(ii,h2o_data,best_params)
    }

    return(model)
  }

  names(best_models) = model_names

  return(best_models)
}

#' Wrapper Function for creating models & outputs
#'
#' @param set config object
#' @param main main data object
#' @param prep summary object
#' @param odbc ODBC connection object (only needed when using DB connection)
#'
#' @return (1) Model fit measures, (2) Featurwe importances (for ML models),
#' (3) Model coefficients (GLM models),
#' (4) Model predictions for VALIDATION data,
#' (5) Table for model application, (6) Data factor levels, and
#' (7) Model parameters
#'
#' @export

create_models <- function(set,main,prep,odbc){

  #h2o.removeAll(timeout_secs = 15)
  start <- Sys.time()

  if(any(grepl('automl',set$model$train_models,
               ignore.case = T))){
    to_train = 'automl'
  }else{
    to_train = set$model$train_models
  }

  allowed = c('glm','gbm','xgboost',
              'decisionTree','randomForest',
              'deeplearning','automl')

  test = to_train %in% allowed
  if(!all(test)){
    wrong = to_train[!test]
    stop(paste(wrong,"is not a supported estimator."))
  }

  print(paste('Selected models:',
              paste(to_train,collapse = ', ')),
        quote = F)

  # (1) Make data: ----
  print('',quote = F)
  print('Converting data to H2O...',quote = F)
  h2o_data <- make_h2o_data(main,set)

  # (2) Create models: ----
  print('',quote = F)
  print('Start model training step:',quote = F)

  # (2.1) Train models: ----
  best_models = list()
  # Train single-estimator superviser models:
  for(estimator in set$model$train_models){
    if(estimator %in% c('automl','isoForest')){
      next()
    }
    print(paste0('   Building ',estimator,' model...'),
          quote = F)
    model <- optimize_h2o_model(df = h2o_data, set = set,
                                estimator = estimator,
                                runid = prep$runid)
    best_models[[model$best_model@model_id]] <- model
  }
  
  # Train autoML models:
  if(to_train == 'automl'){
    best_models <- train_automl_model(df = h2o_data,
                                      set = set,
                                      runid = prep$runid)
  }
  
  # Train anomaly models:
  if('isoForest' %in% to_train){
    print('   Building isolationForest model...',quote = F)
    model_id <- paste(prep$runid,
                     set$main$model_name_part,
                     set$main$label, 'isoForest',
                     get_datetime, sep="_")
    model <- list()
    model$best_model <- train_isoforest_model(
      model_name = model_id,
      h2o_data = h2o_data,
      set = set)
    score <- h2o.predict(model$best_model,
                         h2o_data$val)
    model$score = list(Mean_error = mean(score$predict))
    best_models[[model$best_model@model_id]] <- model
  }
  if('autoencoder' %in% to_train){
    print('   Building autoencoder model...',quote = F)
    model_id <-paste(runid,set$main$model_name_part,
                     set$main$label, 'autoencoder',
                     get_datetime, sep="_")
    model <- list()
    model$best_model <- train_autoencoder_model(
      model_name = model_id,
      h2o_data = h2o_data,
      set = set)
    
    model$score = list(MSE = h2o.mse(model$best_model))
    best_models[[model$best_model@model_id]] <- model
  }


  # (2.2) Score models: ----
  model_names <- names(best_models)
  scores <- unlist(lapply(best_models,'[[','score'))
  names(scores) <- model_names
  if (metric %in% c("rmse", "mae")){
    best <- which.min(scores)
  }else{
    best <- which.max(scores)
  }
  metric <- fix_metric(set = set)
  print(paste0('   Best model: ',
               unlist(strsplit(names(best),'_'))[4],' -- ',
               metric,': ',
               round(scores[best],3)),quote = F)

  # (3) Get outputs: ----
  print('',quote = F)
  print('Collecting outputs...',quote = F)
  models <- lapply(best_models,'[[','best_model')

  output <- get_model_output(models = models,
                             model_names = model_names,
                             set = set,
                             runid = prep$runid,
                             h2o_data = h2o_data)

  # (4) Write outputs: ----
  export_model_output(models = models,
                      output = output,
                      set = set,
                      prep = prep,
                      odbc = odbc)

  print_time(start)
}
