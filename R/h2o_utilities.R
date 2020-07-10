# BILOT AI-jack H2O-module, utilities
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.

#' Start H2O cluster
#'
#' @param set config object
#'
#' @export

start_h2o <- function(set){
  library(h2o)

  h2o.init(nthreads = set$main$num_cores,
           max_mem_size = set$main$max_mem_size,
           min_mem_size = set$main$min_mem_size)

  h2o.removeAll(timeout_secs = 15)
}

#' Convert data-splits to H2O frames
#'
#' @param df main data object, from which the "splitted"-object is used
#' @param set config object
#'
#' @return
#' a list containing training ("train"),
#' testing ("test"), and validation ("val")
#' sets, as well as a combined train-test
#' set ("full_train").
#'
#' @export

make_h2o_data <- function(df,set){

  # Get data splits:
  if(set$model$discretize){
    source = "recategorized"
  }else{
    source = "splitted"
  }
  
  if('timeseries' %in% set$model$train_models){
    library(timetk)
    library(tidyquant)
    X <- main$constants_deleted$value
    X[,1] <- as.Date(dmy(X[,1]))
    X <- tibble::as_tibble(X)
    X <- X %>%
      select(1:3) %>%
      tk_augment_timeseries_signature()
    
    X <- X %>%
      select_if(~ !is.Date(.)) %>%
      select_if(~ !any(is.na(.))) %>%
      mutate_if(is.ordered, ~ as.character(.) %>% as.factor)
    
    train_tbl <- X %>% top_frac(-.7, index.num)
    test_tbl  <- X %>% top_frac(.15, index.num)
    valid_tbl <- X %>% filter(!(index.num %in% train_tbl$index.num), !(index.num %in% test_tbl$index.num))

    
    x_h2o_train <- as.h2o(train_tbl)
    x_h2o_val <- as.h2o(valid_tbl)
    x_h2o_test <- as.h2o(test_tbl)
  }else{

    x_train = df[[source]]$value$train
    x_test = df[[source]]$value$test
    x_val = df[[source]]$value$val
  
    # Create h2o-Frames:
    x_h2o_train <- as.h2o(x_train)
    x_h2o_test <- as.h2o(x_test)
    x_h2o_val <- as.h2o(x_val)
  }
  full_h2o_train <- h2o.rbind(x_h2o_train, x_h2o_test)

  return(
    list(
      train = x_h2o_train,
      test = x_h2o_test,
      val = x_h2o_val,
      full_train = full_h2o_train
    )
  )
}

#' Update H2O installation
#'
#' @examples
#' update_h2o()
#'
#' @export

update_h2o <- function(){
  # The following two commands remove any previously installed H2O packages for R.
  if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
  if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

  # Next, we download packages that H2O depends on.
  pkgs <- c("RCurl","jsonlite")
  for (pkg in pkgs) {
    if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
  }

  # Now we download, install and initialize the H2O package for R.
  install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-yates/4/R")
}



#' Function for making sure evaluation metric is compatible with H2O.
#'
#' @param set config object
#'
#' @return fixed metric
#'
#' @export

fix_metric <- function(set){
  metric <- ifelse(set$main$labeliscategory,
                   set$model$acc_metric_class,
                   set$model$acc_metric_reg)
#  metric <- ifelse(metric %in%
#                     c('mae','auc','rmse','mse','rmsle'),
#                   toupper(metric),metric)
  return(metric)
}

#' Function for generating model outputs
#'
#' @param models either a single model object or a list of models
#' @param model_names a vector of model IDs
#' @param set config object
#' @param runid run ID
#' @param h2o_data a list of H2OFrames
#'
#' @return data.frame of model performance, data.frame of feature importances/model coefficients,
#' model predictions for VALIDATION set, table of model ID's and application tag, and
#' a list of factor levels for categorical predictors.
#'
#' @export

get_model_output <- function(models, model_names, set, runid, h2o_data){

  library(foreach)

  if(!is.list(models)){
    models = list(models)
  }

  # (1) Get parameters: ----
  label <- set$main$label
  metric <- fix_metric(set)
  ids <- model_names
  names(ids) <- names(models)

  time = format(Sys.time(), "%d-%m-%Y %H:%M:%S")

  # Model parameters:
  params <- list()
  for(ii in names(models)){
    params[[ids[ii]]] <- models[[ii]]@allparameters
  }

  # (2) Get feature importance: ----
  test <- grepl('glm',names(models),ignore.case = T)
  coef <- foreach(ii = which(test), .combine = rbind) %do%{
    x <- models[[ii]]
    coef <- h2o.coef(x)
    data.frame(executionid = runid,
               label = label,
               model_name = ids[ii],
               variable = names(coef),
               coef = as.vector(coef),
               row.names = NULL)
  }

  nam <- names(models)
  
  # update test:
  test <- test | grepl('isoForest',names(models),ignore.case = T)
  varimp <- foreach(ii = nam[!test], .combine = rbind)%do%{
    x = models[[ii]]
    data.frame(executionid = runid,
               label = label,
               model_name = ids[ii],
               h2o.varimp(x),
               row.names = NULL)
  }

  # (3) Predict for valid-set: ----
  val_frame <- h2o_data$val
  # (3.1) Other than anomaly detection: ----
  anomaly <- grep('isoForest',names(models),value = T)
  nam <- setdiff(names(models),anomaly)
  pred_val <- foreach(ii = nam, .combine = rbind)%do%{
    x = models[[ii]]
    preds = h2o.predict(x, newdata = val_frame)
    data.frame(
      executionid = as.numeric(runid),
      model_name = ids[ii],
      row_identifier = as.vector(h2o_data$val[, set$main$id]),
      obs = as.vector(h2o_data$val[,label]),
      pred = as.vector(preds$predict),
      proba = round(as.vector(ifelse(is.factor(h2o_data$val[,label]),
                                     preds[,3],NA)),3),
      row.names = NULL,
      stringsAsFactors = F
    )
  }
  
  # (3.2) Anomaly detection: ----
  if(!is.character0(anomaly)){
    vals <- foreach(ii = anomaly, .combine = rbind)%do%{
      x = models[[ii]]
      preds = h2o.predict(x, newdata = h2o_data$val)
      data.frame(
        executionid = as.numeric(runid),
        model_name = ids[ii],
        row_identifier = as.vector(h2o_data$val[, set$main$id]),
        obs = NA,
        pred = as.vector(preds$mean_length),
        proba = as.vector(preds$predict),
        row.names = NULL,
        stringsAsFactors = F
      )
    }
    pred_val <- rbind(pred_val,vals)
  }

  # (4) Calculate model fitting/accuracy measures: ----
  # (4.1) Other than anomaly detection: ----
  perf <- foreach(ii = nam,.combine = rbind) %do% {
    sets = c('train','test','val')
    perf <- lapply(sets, function(x){
      perf = tryCatch(
        h2o.performance(models[[ii]],
                        newdata = h2o_data[[x]])
      )
      x = perf@metrics
      use = grep(metric,names(x),ignore.case = T, value = T)[1]
      perf = round(unlist(x[use]),3)
    })
    perf <- data.frame(perf)
    names(perf) <- paste0('value_',sets)
    return(perf)
  }
  
  # (4.2) Anomaly detection: ----
  if(!is.character0(anomaly)){
    sets = c('train','test','val')
    tmp <- perf[1:length(anomaly),]
    tmp[1:length(anomaly),] <- NA
    perf <- rbind(perf,tmp)
  }

  model_fit_measures <- data.frame(
    executionid = as.numeric(runid),
    time = time,
    label = label,
    model_name = names(models),
    metric = metric,
    perf,
    notions = '',
    row.names = 1:nrow(perf)
  )

  # (5) Row to Apply-table: ----
  apply_model <- data.frame(
    executionid = as.numeric(runid),
    label = label,
    model_name = model_names,
    apply = 0)

  # (6) Output: ----
  return(
    list(
      model_fit_measures = model_fit_measures,
      feature_importance = varimp,
      coefficients = coef,
      predictions = pred_val,
      apply_model = apply_model,
      factor_levels = get_levels(as.data.frame(h2o_data$full_train)),
      parameters = params
    )
  )
}

#' Function for writing model outputs
#'
#' @param models a list of trained models
#' @param output a list of generated model outputs
#' @param set config object
#' @param prep summary object
#' @param odbc ODBC connection file (only needed when using DB connection)
#' @param h2o_data h2o data frame
#'
#' @return (1) Model fit measures, Featurwe importances (for ML models),
#' Model coefficients (GLM models),
#' Model predictions for VALIDATION data, and
#' Table for model application are written to tables (either locl or DB)
#' (2) Data factor levels and Model parameters are written to local .Rds files
#' (3) Models are exported as MOJO objects.
#'
#' @export

export_model_output <- function(models, output, set, prep, odbc, h2o_data){

  # Write warnings, fitmeasures, validation predictions and model apply information
  if(set$main$use_db) {
    # (1) Export to DB: ----

    write_db(channel = odbc$value$odbc_metadata, output$model_fit_measures, set$odbc$result$acc)
    write_db(channel = odbc$value$odbc_metadata, output$apply_model, set$odbc$result$model)
    write_db(channel = odbc$value$odbc_validation, output$predictions, set$odbc$result$val)
    write_db(channel = odbc$value$odbc_metadata, output$feature_importance, set$odbc$result$imp)
    write_db(channel = odbc$value$odbc_metadata_azuredb, output$model_fit_measures, set$odbc$result$acc)
    write_db(channel = odbc$value$odbc_metadata_azuredb, output$apply_model, set$odbc$result$model)
  } else {
    # (2) Export to flat files: ----

    # (2.1) Export performance: ----
    write_csv(set, output$model_fit_measures,
              paste(set$csv$result$prefix,
                    set$csv$result$acc,sep=set$main$path_sep), append = T)
    # (2.2) Export application table: ----
    write_csv(set, output$apply_model,
              paste(set$csv$result$prefix,
                    set$csv$result$model,sep=set$main$path_sep), append = T)
    # (2.3) Export predictions: ----
    # choose target table based on label type
    #target <- ifelse(set$main$labeliscategory,
    #                set$csv$result$val,
    #                set$csv$result$val_reg)
    #if(set$main$append_predicts == FALSE){
    #  if(set$main$labeliscategory){
    #    nam <- c('executionid','model_name',"row_identifier",'obs','pred','proba')
    #  }else{
    #    nam <- c('executionid','model_name',"row_identifier",'obs','pred','proba')
    #  }
    #}else{
    #   nam = FALSE
    #}
    write_csv(set, output$predictions,
              paste(set$csv$result$prefix,
                    set$csv$result$val,sep=set$main$path_sep),
              append = set$main$append_predicts,
              colnames = c('executionid','model_name',"row_identifier",'obs','pred','proba'))
    # (2.4) Export regression coefficients: ----
    if(length(output$coefficients) > 0){
      write_csv(set, output$coefficients,
                paste(set$csv$result$prefix,
                      set$csv$result$coef, sep=set$main$path_sep),
                append = TRUE)
      #,colnames = c('executionid','label','model_name',"variable",'coef'))
    }
    # (2.5) Export feature importances: ----
    if(length(output$feature_importance) > 0){
      write_csv(set, output$feature_importance,
                paste(set$csv$result$prefix,
                      set$csv$result$imp,sep=set$main$path_sep),
                append = TRUE)
    }
  }

  # (3) Save factor levels: ----
  loc <- paste0(set$main$project_path,set$main$path_sep,"output_model",
                set$main$path_sep,"factor_levels",set$main$path_sep,
                paste(prep$runid,set$main$model_name_part,set$main$label,
                      'factorLevels.rds',sep='_'))
  saveRDS(output$factor_levels, file = loc)

  # (4) Save model parameters: ----
  loc <- paste0(set$main$project_path,set$main$path_sep,"output_model",
                set$main$path_sep,"parameters",set$main$path_sep,
                paste(prep$runid,set$main$model_name_part,set$main$label,
                      'parameters.rds',sep='_'))
  saveRDS(output$parameters, file = loc)

  # (5) Save models to MOJO: ----
  path = paste(set$main$project_path,set$main$model_path,sep=set$main$path_sep)
  for(ii in names(models)){
    test <- file.exists(paste0(set$main$model_path,set$main$path_sep,
                               'h2o-genmodel.jar'))
    gwt_jar <- ifelse(test,FALSE,TRUE)

    h2o.download_mojo(model = models[[ii]],
                      get_genmodel_jar = gwt_jar,
                      path = path)

    #h2o.saveModel(object = models[[ii]],
    #             path = path,
    #             force = T)
  }
  
  # (6) Save visuals if there are such: ----
  if("timeseries" %in% set$model$train_models){
  ggsave("time_series.png", viz_ts(data$distance_matrix) 
         path = paste(set$main$project_path, set$main$model_path, 
                      sep = set$main$path_sep))
  }
}
