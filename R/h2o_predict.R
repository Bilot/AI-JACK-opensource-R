# BILOT AI-jack H2O-module, predict
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.

#' List models
#'
#' @description
#' This function lists available models
#' that can be found in the \code{models}
#' results table, located in the
#' \code{model_path} directory, which
#' typically is \code{"output_model/results"}
#'
#' @export

list_models = function(model_path){
  sep = ifelse(grepl('/$',model_path),"","/")
  print(
    read.csv2(paste0(model_path,sep,'models.csv'))
  )
}

#' Get model predictions.
#'
#' @param df main data object
#' @param set config file
#' @param prep summary object
#' @param apply_models row specifying applied model
#' @param odbc ODBC connection config
#' @param odbc_pred ODBC connection config for model application
#'
#' @return model predictions
#'
#' @export

create_predictions <- function(df, set, prep,
                               apply_models,
                               odbc, odbc_pred,
                               useMOJO = TRUE){

  # (1) PARAMETERIZE: ----
  runid <- prep$runid
  model_execid <- apply_models$excutionid
  id_col <- df[, set$main$id]
  label <- as.character(apply_models$label)
  features <- setdiff(names(df),c(label,set$main$id))

  # Is a H2O model used?
  use_h2o <- any(sapply(c('glm','deep','gbm','forest',
                          'tree',"AutoML",'xgboost'),
                        function(x) grepl(x,apply_models$model_name,
                                          ignore.case = T)))


  # (2) FACTOR LEVELS: ----
  # (2.1) Get levels ----
  path = paste0(set$main$project_path,set$main$path_sep,
                'output_model',set$main$path_sep,'factor_levels',
                set$main$path_sep)
  factor_levels <- readRDS(paste0(path,
                                  runid,'_',
                                  set$main$model_name_part,'_',
                                  label,"_factorLevels.rds"))
  # (2.2) Remove label ----
  factor_levels[[label]] = NULL

  # (2.3) Set others-category ----
  for(i in names(factor_levels)){
    replace_others <- sapply(df[,i], function(x) {
      !x %in% factor_levels[[i]]
    })
    if(any(replace_others)){
      df[,i] <- as.character(df[,i])
      df[replace_others,i] <- "others"
    }
    df[,i] <- factor(df[,i],
                     levels = c(factor_levels[[i]],
                                if(any(replace_others)){"others"}else{NULL}))
  }

  # (3) DISCRETIZATION: ----
  if(grepl("disc", apply_models$model_name)){

    # (3.1) Get cutpoints ----
    path <- paste0(set$main$project_path,
                   'output_model',set$main$path_sep,'discretization',
                   set$main$path_sep,
                   runid,'_',
                   set$main$model_name_part,'_',
                   label)
    cutp <- unique(readRDS(file = paste0(path,'_cutpoints.rds')))

    # (3.2) Get factor_levels_disc ----
    factor_levels_disc <- readRDS(file = paste0(path,
                                                '_factor_levels_disc.rds'))

    # (3.3) Apply discretization ----
    vars <- sapply(cutp,'[[',1)
    for(i in 1:length(cutp)){
      df[, vars[i] ] <- factor(
        as.character(
          cut(df[,vars[i]],breaks = unlist(cutp[[i]][-1]))),
        levels = unlist(unique(factor_levels_disc[vars[i]])))
    }
  }

  # (4) MAKE PREDICTIONS: ----
  if(useMOJO){
    path = paste(set$main$project_path,
                 set$main$model_path,
                 sep=set$main$path_sep)
    names <- grep('.zip',dir(path),value = T)
    use <- grep(apply_models$model_name,names,value = T)
    if(length(use)==0){
      use = names(
        which.max(
          sapply(names,function(x){
            stringdist::stringdist(apply_models$model_name,x)
          })
        )
      )
    }
    path <- paste0(set$main$project_path,set$main$path_sep,
                   set$main$model_path,set$main$path_sep,use)
    preds <- h2o.mojo_predict_df(df,mojo_zip_path = path)
  }else{
    start_h2o(set = set)

    path <- paste0(set$main$model_path,set$main$path_sep, apply_models$model_name)
    model_temp <- h2o.loadModel(path)
    mod_feat = attr(model_temp,'model')$names

    # Check features:
    if(!all(mod_feat[-length(mod_feat)] %in% features)){
      cat('MODEL has features:','\n',paste(attr(model_temp,'model')$names,collapse=', '),'\n')
      cat('\nDATA has features:','\n',paste(features,collapse=', '))
      stop('Features in data do not match those assumed by the model.')
    }

    df_hex <- as.h2o(df[,features])
    preds <- h2o.predict(model_temp, newdata = df_hex)
  }

  col = ifelse(set$main$labeliscategory,
               names(preds)[ncol(preds)],
               'predict')
  preds <- round(preds[,col],4)
  pred_temp <- data.frame(
    executionid = runid,
    id = id_col,
    type = ifelse(set$main$labeliscategory,
                  'classification','regression'),
    model = apply_models$model_name,
    pred = as.vector(preds)
  )

  # (5) COLLECT OUTPUT: ----
  # (5.1) Create Model_execution_model_applier ----
  model_exec_ma <- data.frame(
    executionid = runid,
    model_executionid = model_execid,
    label = apply_models$label,
    model_name = apply_models$model_name
  )

  # (5.2) Export ----
  if(set$main$use_db) {
    write_db(channel = odbc_pred, pred_temp, set$odbc$result$pred_app)
    write_db(channel = odbc, model_exec_ma, set$odbc$result$exec_model_app)
  }else{
    write_csv(set, pred_temp,
              paste(set$csv$result$prefix,
                    set$csv$result$pred,sep=set$main$path_sep),
              append = T)
    write_csv(set, model_exec_ma,
              paste(set$csv$result$prefix,
                    set$csv$result$exec_model,sep=set$main$path_sep),
              append = T)
  }
}

# WRAPPERS: ----

#' Wrapper Function for creating model predictions
#'
#' @param set config object
#' @param main main data object
#' @param prep summary object
#' @param odbc ODBC connection object (only needed when using DB connection)
#'
#' @description
#' This fucntion calls the \code{create_predictions()} function.
#'
#' @export

make_predictions = function(set,main,prep,odbc){

  proj = set$main$model_name_part
  dat = length(main)

  df = main[[dat]]$value
  # drop label if included:
  df = df[,setdiff(names(df),set$main$label)]

  if(set$main$use_db==T){
    query = paste0("SELECT * FROM ", set$odbc$result$model,
                   " WHERE apply=1 AND model_name LIKE '%_",proj,"_%'")
    apply_models <- sqlQuery(odbc$value$odbc_metadata,query)
    create_predictions(df, set, prep$runid, prep$runid_row,
                       odbc$value$odbc_metadata, odbc$value$odbc_pred)
  } else {
    path = paste0(set$main$project_path,set$main$path_sep,
                  set$main$model_model_path,set$main$path_sep,
                  set$csv$result$model, ".csv")
    apply_models <- read.csv2(path)
    apply_models$apply[set$main$model_row] = 1
    apply_models <- apply_models[apply_models$apply==1,]

    create_predictions(df = df, set = set,prep = prep,
                       apply_models = apply_models,
                       useMOJO = TRUE,odbc = odbc,
                       odbc_pred = NULL)
  }
}
