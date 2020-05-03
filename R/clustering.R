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
               avg_silhouette_widths = opt[ii][[1]]$avg_width)
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

#' Function for saving models results to output.
#'
#' @param output clustering results
#' @param set config object
#' @param prep config object
#' @param odbc db connection
#'
#' @export
                           
export_clust_output <- function (output, set, prep, odbc) 
{
  if (set$main$use_db) {
    write_db(channel = odbc$value$odbc_metadata, output$model_fit_measures, 
             set$odbc$result$acc)
    write_db(channel = odbc$value$odbc_metadata, output$apply_model, 
             set$odbc$result$model)
    write_db(channel = odbc$value$odbc_validation, output$predictions, 
             set$odbc$result$val)
    write_db(channel = odbc$value$odbc_metadata, output$feature_importance, 
             set$odbc$result$imp)
    write_db(channel = odbc$value$odbc_metadata_azuredb, 
             output$model_fit_measures, set$odbc$result$acc)
    write_db(channel = odbc$value$odbc_metadata_azuredb, 
             output$apply_model, set$odbc$result$model)
  }else {
    write_csv(set, output$model_fit_measures, paste(set$csv$result$prefix, 
                                                    set$csv$result$acc, sep = set$main$path_sep), append = T)
    write_csv(set, output$apply_model, paste(set$csv$result$prefix, 
                                             set$csv$result$model, sep = set$main$path_sep), append = T)
    write_csv(set, output$predictions, paste(set$csv$result$prefix, 
                                             set$csv$result$val, sep = set$main$path_sep), append = set$main$append_predicts, 
              colnames = c("executionid", "model_name", 
                           "row_identifier", "obs"))
    if (length(output$coefficients) > 0) {
      write_csv(set, output$coefficients, paste(set$csv$result$prefix, 
                                                set$csv$result$coef, sep = set$main$path_sep), 
                append = TRUE)
    }
    if (length(output$feature_importance) > 0) {
      write_csv(set, output$feature_importance, paste(set$csv$result$prefix, 
                                                      set$csv$result$imp, sep = set$main$path_sep), 
                append = TRUE)
    }
  }
  loc <- paste0(set$main$project_path, set$main$path_sep, "output_model", 
                set$main$path_sep, "factor_levels", set$main$path_sep, 
                paste(prep$runid, set$main$model_name_part, set$main$label, 
                      "factorLevels.rds", sep = "_"))
  saveRDS(output$factor_levels, file = loc)
  loc <- paste0(set$main$project_path, set$main$path_sep, "output_model", 
                set$main$path_sep, "parameters", set$main$path_sep, 
                paste(prep$runid, set$main$model_name_part, set$main$label, 
                      "parameters.rds", sep = "_"))
  saveRDS(output$parameters, file = loc)
  path = paste(set$main$project_path, set$main$model_path, 
               sep = set$main$path_sep)
  ggsave("clusters.png", viz_clusters(data$distance_matrix,clust = opt$EM$clustering), 
         path = paste(set$main$project_path, set$main$model_path, 
               sep = set$main$path_sep))
  ggsave("silhouettes.png", viz_silhouette(opt), 
         path = paste(set$main$project_path, set$main$model_path, 
               sep = set$main$path_sep))
}
                           
#' Function for visualizing clustering result.
#'
#' @param D distance matrix
#' @param clust clustering model
#' @param colors color scale
#'
#' @export plot

viz_clusters <- function(D,clust,colors = NULL){
  
  if(is.null(colors)){
    cols <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
  } else {
    cols <- colors
  }
  
  ord <- cmdscale(D)
  
  data.frame(ord,clustering=as.factor(clust)) %>%
    ggplot2::ggplot(.,aes(X1,X2,fill=clustering)) +
    geom_point(shape=21,size=2,alpha=0.8)+
    stat_ellipse(level = 0.9,geom='polygon',alpha=0.1)+
    scale_fill_manual(values=cols,name='')+
    theme_linedraw()+
    theme(legend.position = 'top')+
    xlab('Axis-1')+ylab('Axis-2')
}

#' Function for visualizing clustering metrics for a given model.
#'
#' @param opt clustering model
#' @param colors color scale
#'
#' @export plot
                           
viz_silhouette <- function(opt, colors = NULL){
  
  if(is.null(colors)){
    cols <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
  } else {
    cols <- colors
  }
  
  df <- do.call(rbind,
                lapply(names(opt),function(x){
                  df <- as.data.frame(
                    unclass(opt[[x]]$silhouette_widths)
                  )
                  df$method <- x
                  return(df)
                })
  ) %>%
    group_by(method,cluster) %>%
    arrange(-sil_width,.by_group=T) %>%
    group_by(method) %>%
    mutate(x = 1:n())
  
  ggplot(df,aes(x,sil_width,
                fill=as.factor(cluster),
                col=as.factor(cluster)))+
    geom_col(position = 'dodge')+
    facet_wrap(~method,ncol = 1)+
    theme_linedraw()+
    scale_color_manual(values=cols,name='')+
    scale_fill_manual(values=cols,name='')+
    xlab('')+ylab(expression(paste('Silhouette width, ',S[i])))
  
}
