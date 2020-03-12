# BILOT AI-jack WRAP-UP-module
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.


#' Clean up session
#'
#' @examples
#' clean_up()
#'
#' @export

clean_up = function(){

  warnings()
  sink(type="message")
  sink()
  #if(!is.null(odbc)){
  #  odbcClose(odbc$odbc_metadata)
  #}
  #try({h2o.shutdown(prompt = FALSE)},silent = TRUE)
  try(closeAllConnections(),silent = T)
}

#' Clear results tables
#'
#' @param result_path where to find results tables?
#'
#' @examples
#' clear_model_results(result_path)
#'
#' @description
#' This function clears result tables
#' by removing all rows, except for the
#' column names. As input, give the path
#' where the \code{result} folder is located.
#'
#' @export

clear_model_results = function(result_path){

  if(!grepl('results',result_path)){
    path = paste0(result_path,'/results')
  }else{
    path = result_path
  }


  # Loop through tables:
  tables = setdiff(dir(path),'README.txt')
  for(ii in tables){
    nam = paste0(path,'/',ii)
    x = read.csv2(nam,nrow = 1)
    write.table(x[-1,],file = nam,row.names = FALSE,
                col.names = TRUE, sep = ';')
  }
  print('Result tables cleared.')
}

#' Collect desired results from local files.
#'
#' @param result_path path to \code{result} directory
#' @param executionid for which execution to get the results
#' @param tables which result tables to return?
#'
#' @export

collect_results = function(result_path,executionid,
                           tables = c('execution','accuracy','column_importance',
                                      'coefficients','metadata','validation')){

  options(readr.num_columns = 0)
  files = dir(result_path)

  if(!all(sapply(tables,function(x) any(grepl(x,files))))){
    stop('All tables not found in result_path.')
  }

  nam = sapply(tables,function(x) grep(x,files,value = T))
  eid = executionid

  results = lapply(nam,function(x){
    df <- suppressWarnings(
    suppressMessages(
    readr::read_csv2(paste0(result_path,'/',x),col_types = cols())
    )
    )
    return(df[df$executionid %in% eid,])
  })
  names(results) = gsub('.csv','',nam)
  return(results)
}


#' Write main data object to file.
#'
#' @param df data.frame
#' @param set config object
#' @param prep summary object
#' @param get_datetime timestamp
#'
#' @export

save_data = function(df,set,prep,get_datetime){

  loc <- paste(set$main$project_path,'/output_model',"/data_objects/",
               prep$runid, "_",
               set$main$model_name_part,'_',
               set$main$label, "_data_",
               get_datetime, ".rds", sep="")
  saveRDS(df, loc)
  print("Data saved to file.", quote = F)
}

#' Clear result tables
#'
#' @param result_path
#'
#' @description
#' This function clears information from result tables
#'
#' @export

clear_model_results = function(result_path){

  #if(!('result' %in% dir())) stop('result-folder not in path')

  if(!grepl('results',result_path)){
    path = paste0(result_path,'/results')
  }else{
    path = result_path
  }

  # Loop through tables:
  tables = setdiff(dir(path),'README.txt')
  for(ii in tables){
    nam = paste0(path,'/',ii)
    x = read.csv2(nam,nrow = 1)
    write.table(x[-1,],file = nam,row.names = FALSE,
                col.names = TRUE, sep = ';')
  }
  print('Result tables cleared.')
}

#' Delete a project from given path.
#'
#' @param path system path from where to delete a project
#'
#' @export

delete_project = function(project_path){
  unlink(project_path,recursive = T)
}
