# BILOT AI-jack H2O-module, transformations
# (c) Bilot Oy 2020
# Any user is free to modify this software for their
# own needs, bearing in mind that it comes with no warranty.

#' Create data partitioning key
#'
#' @param main main data object
#' @param set config object
#'
#' @return main data object with data split column added
#'
#' @description
#' If the label is categorical, a stratified data split
#' (based on the label variable) is created using
#' the createDataPartition function in the caret-package,
#' based on split proportions defined in configuration. Othervise,
#' a random split is made. A new column is added to the type-set
#' data, which is returned.
#'
#' @export

create_split <- function(main,set){
  X <- main$with_types$value$Data_with_types
  y <- X[[set$main$label]]

  # Calculate proportions:
  p_tr <- set$split_data$prop_train
  p_ts <- set$split_data$prop_test/(1-p_tr)
  splits <- numeric(nrow(X))

  # Get indices:
  if (set$main$labeliscategory){
    i_tr <- caret::createDataPartition(
      y,p <- p_tr)$Resample1
    i_ts <- caret::createDataPartition(
      y[-i_tr],
      p <- p_ts)$Resample1

    # Define splits:
    splits[i_tr] <- 1
    splits[-i_tr][i_ts] <- 2
    splits[splits==0] <- 3
  } else {
    vals <- runif(nrow(X))
    i_tr <- which(vals < set$split_data$prop_train)
    i_ts <- which(vals > set$split_data$prop_train &
                  vals < (set$split_data$prop_train + set$split_data$prop_test))
    # Define splits:
    splits[i_tr] <- 1
    splits[i_ts] <- 2
    splits[splits==0] <- 3
  }

  # Add to data:
  main$with_types$value$Data_with_types$test_train_val <- as.character(splits)
  main$with_types$value$Variable_types <- rbind(main$with_types$value$Variable_types,c('test_train_val','char'))

  return(main)
}

#' Nominalise features with NA's
#'
#' @param df data.frame to transform
#' @param set config object
#'
#' @return main object with transformed data.
#'
#' @description
#' This function is used, when there is too much NA's in
#' continuous variables and you can't impute them.
#' This replaces negative values with \code{'Under0'}, zeros with \code{'0'},
#' positive values with \code{'Over0'} and NA's with \code{'Missing'} and
#' factorizes the variable.
#'
#' @export

trans_classifyNa <- function(df, set) {
  nam = setdiff(names(df),set$main$id)
  na_count = colSums(is.na(df))

  for(i in nam[na_count >= set$trans_classifyNa$limit]){
    tmp = df[, i]
    tmp = as.character(tmp)
    tmp = ifelse(tmp < 0 & !is.na(tmp),'Under0',tmp)
    tmp = ifelse(is.na(tmp) | tmp == 'Missing' | tmp == 'missing','Missing',tmp)
    tmp = ifelse(tmp != '0' & tmp != 'Missing' & tmp != "Under0", 'Over0',tmp)
    df[,i] = factor(tmp)
  }

  return(df)
}

#' Delete constant features
#'
#' @param df data.frame to transform
#' @param set config object
#'
#' @return main object with transformed data.
#'
#' @description
#' Variables that have no variation are dropped.
#'
#' @export

trans_delconstant <- function(df, set){
  nam = setdiff(names(df),c(set$main$id,set$main$test_train_val))
  fix = nam[sapply(df[,nam],is.character)]
  for(ii in fix){
    df[,ii] = factor(df[,ii])
  }

  cols <- names(df)[sapply(df, is.numeric) | sapply(df, is.factor)]
  deleted_columns <- cols[which(sapply(df[,cols],function(x) length(unique(x)))==1)]

  if(!is.character0(deleted_columns)){
    df <- df[,-which(colnames(df) %in% deleted_columns)]
  }

  return(df)
}

#' Delete equal features
#'
#' @param df data.frame to transform
#'
#' @return main object with transformed data.
#'
#' @description
#' This function deletes equal columns. Deletes columns with lower indexes.
#' Returns a list where the first object is a list of vectors that include the
#' columns that are similar from one loop.
#'
#' @export

trans_delequal <- function(df){
  j <- 1
  deleted_columns <- list()
  for(k in 0:(ncol(df)-1)){
    b <- sapply(1:ncol(df), FUN  =function(i){
      identical(df[, i], df[, ncol(df)-k])})
    if(sum(b) > 1){
      deleted_columns[[j]] <- names(df[b])
      j <- j + 1
    }
    b[ncol(df)-k] <- FALSE
    df <- df[, !b]
  }
  list(Data_del_equal = df, deleted_columns =  deleted_columns)
}

#' Replace species characters in variables
#'
#' @param df data.frame to transform
#' @param set config object
#'
#' @return main object with transformed data.
#'
#' @export


trans_replaceScandAndSpecial <- function(df, set){
  names(df) = gsub(' ','_',names(df))
  names(df) = gsub("&", "ja", names(df))
  names(df) = gsub("\u00E4", "a", names(df))
  names(df) = gsub("\u00F6", "o", names(df))

  tmp <- df[, !names(df)%in%set$main$label]

  if(!is.character0(names(tmp[,sapply(tmp, is.factor)]))){
    d <- df[, names(df[,sapply(df, is.factor)])]
    d2 <- df[, setdiff(names(df), names(d))]
    if(nrow(d) == 1) {
      d <- data.frame(lapply(d, function(x){gsub("\u00E4", "a", x)}))
      d <- data.frame(lapply(d, function(x){gsub("\u00F6", "o", x)}))
    } else {
      d <- data.frame(sapply(d,function(x) gsub("\u00E4", "a", x)))
      d <- data.frame(sapply(d,function(x) gsub("\u00F6", "o", x)))
    }
    if(!is.null(names(d2))){
      d2 <- data.frame(d2, d)
    }else{
      d2 <- data.frame(Id = d2, d)
      d2$Id <- as.character(d2$Id)
    }
    return(d2)
  }else{
    return(df)
  }
}

#' Discretisize continuous variables by entropy
#'
#' @param df1 full data
#' @param df2 combined testing & validation set
#' @param set config object
#'
#' @return
#' Returns a list of discretized data,
#' new variables and their cut-points.
#' If it is not possible to descretize,
#' returns NULL.
#'
#' @export

trans_entropy <- function(df1, df2, set){

  start <- Sys.time()
  df_disc <- df2
  df_disc_full <- df1
  cutp <- list()
  numeric_variables <- names(df2)[sapply(df2, is.numeric)]
  cutpoints <- list()
  h <- 1
  jitter_factor <- set$trans_entropy$jitter_factor

  cl <- makeCluster(set$main$num_cores)
  registerDoParallel(cl)
  clusterEvalQ(cl, .libPaths())

  cutp <- foreach(i = numeric_variables,
                  .combine='c',
                  .packages = c("data.table", "discretization")
  ) %dopar% {
    cutp_help <- NULL
    df_help_narm <- df_disc[
      complete.cases(
        df_disc[, c(i, set$main$label)]),
      c(i, set$main$label)
      ]

    if(set$trans_entropy$skip_na){
      try(cutp[i] <- list(cutp=c(i,-Inf,
                                 mdlp(df_help_narm)$cutp,
                                 Inf)),
          silent = F)
    } else {
      try(cutp_help <- list(cutp=c(i,-Inf,
                                   mdlp(df_help_narm)$cutp,
                                   Inf)),
          silent = F)
      while(is.null(cutp_help)){
        try(cutp_help <- list(cutp=c(i,-Inf, mdlp(jitter(df_help_narm[,1],
                                                         factor = jitter_factor))$cutp, Inf)), silent = F)
        jitter_factor <- jitter_factor*2
      }
      cutp[i] <- cutp_help
    }
    return(cutp)
  }

  stopCluster(cl)
  if(length(cutp) > 0){
    for(i in 1:length(cutp)){
      if(!"All" %in% cutp[[i]]) {
        df_disc_full[paste(cutp[[i]][1], "DISC", sep="_" )] <-
          as.character(cut(df_disc_full[,unlist(cutp[[i]][1])],
                           breaks=unlist(cutp[[i]][-1])))
        cutpoints[[h]] <- cutp[[i]]
        h <- h + 1
      }
    }
    diff_names <- setdiff(colnames(df_disc_full), colnames(df2))

    if(identical(diff_names, character(0))){
      print(round((Sys.time() - start), 1))
      out = NULL
    } else {
      df_disc_full <- as.data.frame(df_disc_full[, diff_names],
                                    stringsAsFactors = F)
      colnames(df_disc_full) <- diff_names
      df_disc_full[is.na(df_disc_full)] <- "missing"
      df_disc_full[] <- lapply(df_disc_full, factor)
      df_full <- df1[, !colnames(df1)%in%colnames(df_disc_full)]
      df_full <- cbind(df_full, df_disc_full)

      out = list(data_disc = df_full,
                 new_variables = df_disc_full,
                 cutpoints = cutpoints,
                 train = df_full[df_full$test_train_val =='1',],
                 test = df_full[df_full$test_train_val =='2',],
                 val = df_full[df_full$test_train_val =='3',])
    }
  } else {
    out = list()
  }
  return(out)
}

# WRAPPERS: ----

#' Wrapper Function for calling entropy transformation.
#'
#' @param main main data object
#' @param set config object
#' @param prep summary object
#'
#' @return main object with transformed data.
#'
#' @export

entropy_recategorization = function(main, set, prep) {

  start <- Sys.time()

  if (set$model$discretize) {
    label <- set$main$label
    test <- is.factor(main$with_types$value$Data_with_types[[label]])
    if (test) {
      df_full <- main$constants_deleted$value
      df_test_val <- do.call("rbind", main$splitted$value[c(2,
                                                            3)])
      main$recategorized <- handling_trycatch(trans_entropy(df1 = df_full,
                                                            df2 = df_test_val, set))

      df <- main$recategorized$value$data_disc
      df2 <- main$constants_deleted$value
      factor_levels <- get_levels(df2)
      lev <- get_levels(df)
      factor_levels_disc <- lev[!(lev %in% factor_levels)]

      # Save cutpoints:
      path = paste(set$main$project_path, "output_model/",
                   sep = "/")
      saveRDS(main$recategorized$value$cutpoints,
              file = paste0(path, "discretization/",
                            prep$runid, "_", set$main$model_name_part,
                            "_", set$main$label, "_", "cutpoints.rds"))
      # Save factor_levels_disc:
      saveRDS(factor_levels_disc, file = paste0(path,
                                                "discretization/", prep$runid, "_", set$main$model_name_part,
                                                "_", set$main$label, "_", "factor_levels_disc.rds"))

    } else {
      main$recategorized = list()
      main$recategorized$value <- "Label is not factor"
    }

    print("Data categorised (entropy).", quote = F)
    print_time(start)

  }
  return(main)
}

#' Wrapper Function for Making Data Transforms.
#'
#' @param main main data object
#' @param set config object
#' @param prep summary object
#' @param selection which transformations to apply?
#'
#' @return main data object with transformed data.
#'
#' @examples
#' do_transforms(main, set, prep,
#'               selection = c("delete_equal",
#'                             "clean_special",
#'                             "classify_NA",
#'                             "delete_constant"))
#'
#' @description
#' allowed transformations \code{c("delete_equal", "clean_special",
#' "classify_NA", "delete_constant")}. In addition, a column
#' defining a data split will also be added, if missing from
#' the data.
#'
#' @export

do_transforms = function(main, set, prep,
selection = c("delete_equal", "clean_special",
              "classify_NA", "delete_constant")) {

  start <- Sys.time()

  allowed = c("delete_equal", "clean_special", "classify_NA",
              "delete_constant")

  if (is.null(selection)) {
    stop("Need to specify at least one transformation")
  }
  if (!all(selection %in% allowed)) {
    stop("Unrecognised transformation")
  }

  # (0) Add data split column if missing: ----
  if (!set$main$test_train_val %in% colnames(main$with_types$value$Data_with_types)) {
    print("   Adding data-split...", quote = F)
    main <- create_split(main,set)
  }

  # (1) Delete equal features: ----
  if ("delete_equal" %in% selection) {
    print("   Deleting equal features...", quote = F)
    main$equal_deleted <- handling_trycatch(
      trans_delequal(main$with_types$value$Data_with_types))
  }

  # (2) Clean Scandic and special characters: ----
  if ("clean_special" %in% selection) {
    print("   Cleaning special characters...", quote = F)
    if (!("delete_equal" %in% selection)) {
      X = main$with_types$value$Data_with_types
    } else {
      X = main$equal_deleted$value$Data_del_equal
    }
    main$special_replaced <- handling_trycatch(
      trans_replaceScandAndSpecial(X,set))
  }

  # (3) Classify features with NA: ----
  if ("classify_NA" %in% selection) {
    if ("clean_special" %in% selection) {
      X = main$special_replaced$value
    } else {
      if ("delete_equal" %in% selection) {
      } else {
        X = main$with_types$value$Data_with_types
      }
    }
    # Map missing values:
    print("   Mapping missing values...", quote = F)
    # save if needed later for validating new model
    # versions
    main$NA_classified <- handling_trycatch(
      trans_classifyNa(X,set))
  }

  # (4) Delete constant features: ----
  if ("delete_constant" %in% selection) {
    print("   Deleting constant features...", quote = F)
    if ("classify_NA" %in% selection) {
      X = main$NA_classified$value
    } else {
      if ("clean_special" %in% selection) {
        X = main$special_replaced$value
      } else {
        if ("delete_equal" %in% selection) {
        } else {
          X = main$with_types$value$Data_with_types
        }
      }
    }
    main$constants_deleted <- handling_trycatch(trans_delconstant(X,set))
  }

  print("Transformations applied.", quote = F)

  # (5) Final check for missing values: ----
  check_missing(main, set, prep)

  print_time(start)

  return(main)
}
