# BILOT AI-jack PLUMBER-module
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.


#' Get model predictions through plumber API
#' 
#' @param df data.frame returned by \code{create_df()}
#' @param set plumber config object
#' @param param API parameter of data values
#' @param param2 API parameter of column names
#' @param param3 API parameter of data types
#' @param odbc optional ODBC connection object
#' 
#' @return Writes the output to file or to DB, depending on the settings.
#' 
#' @export
plumber_predict <- function(df, set, param, param2, param3, odbc){
  
  # Select model for application:
  if(set$main$use_db==T) {
    apply_models<-sqlQuery(odbc$odbc_metadata_azuredb, set$odbc$query_r)	
  }
  if(set$main$use_db==F) {
    apply_models <- read.csv2(paste(set$main$model_model_path,'/', 
                                   set$main$model_model_file, 
                                   ".csv", sep=""))
    apply_models$apply[set$main$model_row] = 1
    apply_models <- apply_models[apply_models$apply==1,]
  }
  
  # Create prediction:
  pred <- data.frame(
    ID = as.vector(df[, set$main$id]), 
    pred = h2o.mojo_predict_df(
      frame = df, 
      mojo_zip_path = paste0(set$main$model_path,'/',
                             apply_models$model_name,'.zip')), 
    model_name = apply_models$model_name, 
    predtime = "default", 
    notions='', 
    parameters = paste(param, param2, param3, sep="|")
    )
  
  # Write result:
  if(set$main$use_db) {
    insert_db_plumber(
      odbc$odbc_metadata_azuredb, 
      set$odbc$model_table_pred, 
      pred_temp, apply_models)
  } else {
    pred$predtime <- Sys.time()
    write.table(x = pred, file = set$write_csv$file_name,
                append = T, quote = F, row.names = F,
                sep = ";", col.names=F)
  }
}

#' Create ODBC connection
#' 
#' @param set plumber config object
#' 
#' @export
plumber_odbc <- function(set){
    if(set$main$use_db==T) {
    odbc_metadata_azuredb <- odbcDriverConnect(
        connection = paste("Driver={ODBC Driver 13 for SQL Server};server=", 
        set$odbc$server_r, ";database=", 
        set$odbc$database_r, ";Uid=", 
        set$odbc$user_r, ";Pwd=", 
        set$odbc$user_pw_r, ";Encrypt=yes", 
        sep=""))
    return(list(odbc_metadata_azuredb = odbc_metadata_azuredb))
    }
}

#' Parse data.frame from input parameters
#' 
#' @param param API parameter of data values
#' @param param2 API parameter of column names
#' @param param3 API parameter of data types
#' 
#' @return data.frame object
#' 
#' @export
create_df <- function(param, param2, param3){
    # Parse data.frame from input parameters:
    df <- data.frame(t(unlist(strsplit(param, split="#"))))
    colnames(df) <- unlist(strsplit(param2, split="#"))
    types <- unlist(strsplit(param3, split="#"))
    
    # Set col-types:
    df[types=="c"] <- lapply((df[types=="c"]),as.character)
    df[types=="f"] <- lapply(df[types=="f"], factor)
    df[types=="n"] <- apply(df[types=="n"],1, FUN=as.numeric)
    df[types=="i"] <- apply(df[types=="i"],1, FUN=as.numeric)

    return(df)
}

#' Parse input for plumber API
#' 
#' @param file_path path to file that contains row(s) for prediction
#' @param row which row to use (default = 1)?
#' @param set plumber config object
#' 
#' @return parameter string
#'
#' @export
parse_params <- function(file_path,row = 1,set){
  line <- suppressWarnings(
    read.table('test_line.csv',header = T,sep = set$main$file_sep,
               stringsAsFactors = F)
  )[row,]
  param <- paste0("param=",paste(line,collapse = '#'))
  param2 <- paste0("param2=",paste(colnames(line),collapse = '#'))
  
  types <- read.table(set$read_variable_types$file_path,
                      header = T,sep=set$main$file_sep,
                      stringsAsFactors = F,strip.white = T)
  x <- types[,set$read_variable_types$type_column]
  y <- types[,set$read_variable_types$name_column]
  
  new <- as.data.frame(matrix(substring(sapply(line,class),1,1),1,length(line)),
                       stringsAsFactors = F)
  colnames(new) <- colnames(line)
  new[colnames(new) %in% y] <- dplyr::case_when(
      x == 'real' ~ 'n', x == 'int' ~ 'n', x == 'float' ~ 'n', x == 'numeric' ~ 'n',
      x == 'bigint' ~ 'c', x == 'identity' ~ 'c', x == 'char' ~ 'c',
      x == 'bit' ~ 'f', x == 'varchar' ~ 'f', x == 'nvarchar' ~ 'f',
      x == 'datetime' ~ 'c', x == 'date' ~ 'c', x == 'time' ~ 'c',
      TRUE ~ 'c')
  
  param3 <- paste0("param3=",paste(new,collapse = '#'))
  
  return(paste(param,param2,param3,sep='&'))
}

#' Insert predictions to DB table
#' 
#' @param channel odbc connection
#' @param table_name table name in DB
#' @param prediction data.frame with model prediction 
#' @param apply_models selected row from models-table
#' 
#' @return Writes predictions to DB table.
#' 
#' @export
insert_db_plumber <- function(channel, table_name, 
                              prediction, apply_models){
	query <- paste0("INSERT INTO ", table_name, 
                   " VALUES( '", prediction$ID, 
                   "',", prediction$pred, 
                   ",", "default", 
                   ",'", apply_models$model_name, 
                   "',", "''", ")")
	sqlQuery(channel, query)
}