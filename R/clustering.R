# BILOT AI-jack CLUSTERING-module
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.

#' Wrapper Function for preparing data to apply clustering methods.
#'
#' @param X data object
#' @param set config object
#'
#' @return list of data frames and distance matrix.
#'
#' @export

cluster_dataprep <- function(X, set){
  
  types <- set$cluster$col_types_used
  
  scaler <- function(x, method='standard'){
    if(method == 'standard'){
      z <- scale(x)
    }else if(method == 'minmax'){
      z <- sapply(x,function(x) (x-min(x))/(max(x)-min(x)))
    }
    return(z)
  } 
  
  # Drop unused columns: ----
  X <- X[,setdiff(names(X),set$cluster$cols_not_included)]
  
  # Down sample data: ----
  if(nrow(X) > set$cluster$n_max){
    X <- X[sample(x = 1:nrow(X),
                  size = set$cluster$n_max),]
  
  }
  
  # Feature types: ----
  feat_types <- list(
    numeric = names(which(sapply(X,is.numeric))),
    categorical = names(which(sapply(X,is.factor))),
    all = names(X)
  )
  
  # OneHot encoding: ----
  if(types %in% c('all','categorical')){
    tmp <- X[,feat_types$categorical]
    dummies <- dummyVars(~.,data=tmp)
    dummies <- predict(dummies,tmp)
  }
  
  # Calculate distance matrix: ----
  if(types == 'numeric'){
    X_tr <- scaler(X[,feat_types$numeric])
    D <- dist(X_tr)
  } else if(types == 'categorical'){
    D <- dist(dummies)
  } else if(types == 'all'){
    X_tr <- scaler(X[,feat_types$numeric],'minmax')
    if(set$cluster$use_gower){
      X_tr <- cbind(X_tr,X[,feat_types$categorical])
      D <- daisy(X_tr,metric = 'gower')  
    }else{
      X_tr <- cbind(X_tr,dummies)
      D <- dist(X_tr)
    }
  }
  
  return(
    list(
      raw_data = X,
      transformed_data = X_tr,
      distance_matrix = D
    )
  )
}

#' Function for applying clustering algorithms.
#'
#' @param data data object
#' @param set config object
#'
#' @return list of algorithms results.
#'
#' @export

optimize_clustering <- function(data,set){
  
  cluster_range <- set$cluster$cluster_range
  k_vals <- cluster_range[1]:cluster_range[2]
  X_tr <- data$transformed_data
  
  results <- list()
  clusts <- list()
  sils <- list()
  opt <- list()
  
  # Perform Expectation Maximization clustering: ----
  results[['EM']] <- Mclust(X_tr, G = k_vals)
  clusts[['EM']] <- results[['EM']]$classification
  sils[['EM']] <- silhouette(clusts[['EM']], data$distance_matrix)
  opt[['EM']] <- list()
  opt[['EM']]$clustering <- clusts[['EM']]
  opt[['EM']]$n_clusters <- max(opt[['EM']]$clustering)
  opt[['EM']]$silhouette_widths <- sils[['EM']]
  opt[['EM']]$clust_avg_widths <- summary(opt[['EM']]$silhouette_widths)$clus.avg.widths
  opt[['EM']]$avg_width <- summary(opt[['EM']]$silhouette_widths)$avg.width
  
  registerDoParallel(set$cluster$n_jobs)
  
  # Perform k-means clustering: ----
  results[['kmeans']] <- foreach(k = k_vals) %dopar% {
    kmeans(data$distance_matrix,centers = k,iter.max = set$cluster$max_iter)
  }
  clusts[['kmeans']] <- lapply(results[['kmeans']], '[[', 'cluster')
  
  # Perform Partition Around Medoids clustering: ----
  results[['pam']] <- foreach(k = k_vals) %dopar% {
    cluster::pam(data$distance_matrix,k = k)
  }
  clusts[['pam']] <- lapply(results[['pam']], '[[', 'clustering')
  
  for(i in names(results)[2:length(results)]){
    mw <- which.max(sapply(clusts[[i]], function(cl) summary(silhouette(cl,data$distance_matrix))$avg.width))
    sils[[i]] <- silhouette(clusts[[i]][[mw]],data$distance_matrix)
    opt[[i]]$clustering <- clusts[[i]][[mw]]
    opt[[i]]$n_clusters <- max(opt[[i]]$clustering)
    opt[[i]]$silhouette_widths <- sils[[i]]
    opt[[i]]$clust_avg_widths <- summary(opt[[i]]$silhouette_widths)$clus.avg.widths
    opt[[i]]$avg_width <- summary(opt[[i]]$silhouette_widths)$avg.width
  }
  
  stopImplicitCluster()
  
  return(opt)
}

#' Function for saving models results to output.
#'
#' @param data data object
#' @param set config object
#'
#' @return list of algorithms results
#'
#' @export

get_cluster_output <- function(set, runid, data){
  
  runid <- prep$runid
  
  time = format(Sys.time(), "%d-%m-%Y %H:%M:%S")
  
  pred_val <- foreach(ii = names(opt), .combine = rbind) %do% {
    data.frame(executionid = as.numeric(runid), model_name = ii, 
               row_identifier = names(opt[ii][[1]][1]$clustering), 
               obs = as.vector(opt[ii][[1]][1]$clustering))
  }
  perf <- foreach(ii = names(opt), .combine = rbind) %do% {
    data.frame(model_name = ii, 
               avg_silhouette_widths = opt[ii][[1]]$avg_width, 
               clusters_avg_silhouette_widths = as.list(opt[ii][[1]]$clust_avg_widths))
  }
    
  apply_model <- data.frame(
    executionid = as.numeric(runid),
    model_name = names(opt),
    apply = 0)

  model_fit_measures <- data.frame(executionid = as.numeric(runid), 
                                   time = time, model_name = names(opt), 
                                   perf, notions = "", row.names = 1:nrow(perf))
  apply_model <- data.frame(executionid = as.numeric(runid), 
                            model_name = names(opt), apply = 0)
                            
  return(
    list(
      model_fit_measures = model_fit_measures,
      predictions = pred_val,
      apply_model = apply_model,
      factor_levels = get_levels(as.data.frame(data$transformed_data))
    )
  )
}

#' Function for saving outputs.
#'
#' @param set config object
#' @param prep config object
#' @param odbc database
#'
#' @export

write_cluster_exec <- function (set, prep, odbc) {
  if (set$main$use_db == T) {
    write_db(channel = odbc$value$odbc_metadata, prep$execution_row, 
             set$odbc$result$exec)
    write_db(channel = odbc$value$odbc_metadata, prep$summary_table, 
             set$odbc$result$metad)
    write_db(channel = odbc$value$odbc_metadata, prep$columns, 
             set$odbc$result$cols)
  }
  if (set$main$use_db == F) {
    write_csv(set, prep$execution_row, paste(set$csv$result$prefix, 
                                             set$csv$result$exec, sep = set$main$path_sep), append = T)
    write_csv(set, prep$summary_table, paste(set$csv$result$prefix, 
                                             set$csv$result$metad, sep = set$main$path_sep), append = T)
    write_csv(set, prep$columns %>% select(-label), paste(set$csv$result$prefix, 
                                       set$csv$result$cols, sep = set$main$path_sep), append = T)
  }
  print("Execution rows written.", quote = F)
}
