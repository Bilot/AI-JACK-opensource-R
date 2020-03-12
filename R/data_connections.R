# BILOT AI-jack DATA CONNECTION-module
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.

# DB-functions: ----

#' Establishing ODBC connection
#'
#' @param set config object
#'
#' @return
#' A list of odbc read and write connections.
#'
#' @export

odbc_connections <- function(set) {

    odbc_source <- odbcConnect("odbcmodellingsource")
    odbc_metadata <- odbcConnect("odbcmodellingmetadata")
    odbc_prediction <- odbcConnect("odbcmodellingprediction")
    odbc_validation <- odbcConnect("odbcmodellingvalidation")
    odbc_metadata_azuredb <- odbcDriverConnect(connection = set$odbc$con_string)
    return(list(odbc_source = odbc_source, odbc_metadata = odbc_metadata,
        odbc_prediction = odbc_prediction, odbc_validation = odbc_validation,
        odbc_metadata_azuredb = odbc_metadata_azuredb))
}

#' Establishing ODBC connection
#'
#' @param set config object
#' @param variable_types variable types table
#' @param df data object
#' @param odbc_metadata
#'
#' @examples
#' db_preparations(set, variable_types, df, odbc_metadata)
#'
#' @return
#' A list of odbc read and write connections.
#'
#' @export

db_preparations <- function(set, variable_types, df, odbc_metadata) {
    # (1) Get (next) execution id: ----
    runid <- sqlQuery(odbc_metadata, paste("SELECT CASE WHEN MAX(executionid) IS NULL THEN 0 ELSE max(executionid) END as executionid FROM ",
        set$odbc$result$exec, sep = "")) + 1

    # (2) Get tables: ----
    query <- set$odbc$query_r
    tables <- make_tables(runid, df, variable_types, query)

    # (3) Return output: ----
    return(list(runid = runid, execution_row = tables$execution_row,
        summary_table = tables$summary_table, columns = tables$columns))
}

# CSV-functions: ----

#' Generate execution tables
#'
#' @description
#' Function for getting execution tables, when using
#' local file connection.
#'
#' @param main main data object
#' @param set config object
#'
#' @return
#' list with runid,execution_row, summary_table, and columns
#'
#' @export

csv_preparations <- function(set, main) {

    variable_types <- main$with_types$value$Variable_types
    df <- main$with_types$value$Data_with_types
    sep = ifelse(grepl("/$", set$csv$result$prefix), "",
        "/")
    path <- paste0(set$csv$result$prefix, sep, set$csv$result$exec,
        ".csv")

    # (1) Get (next) execution id: ----
    if (nrow(read.csv(path, sep = set$csv$result$sep, nrows = 2)) ==
        0) {
        runid <- 1
    } else {
        tmp = read.csv(path, header = T, sep = set$csv$result$sep)
        runid <- as.numeric(max(tmp$executionid)) + 1
    }

    # (2) Get tables: ----
    path <- paste(set$main$project_path, set$read_csv$file_path,
        sep = "/")
    # If several files, map to 'model_name_part'
    path <- path[grep(set$main$model_name_part,path,ignore.case = T)]

    query <- paste("csv-file:", path)
    tables <- make_tables(runid, df, variable_types, query)

    # (3) Return output: ----
    return(list(runid = runid, execution_row = tables$execution_row,
        summary_table = tables$summary_table, columns = tables$columns))
}

# WRAPPER: ----

#' Wrapper Function for Establishing Data Source
#' Connections.
#'
#' @param set = config object
#' @return
#' ODBC connection object (NULL if using file
#' connection)
#'
#' @export

open_connections = function(set) {

    if (set$main$use_db) {
        odbc <- handling_trycatch(odbc_connections(set))
        print("ODBC connection established.", quote = F)
    } else {
        print("File connection established.", quote = F)
        odbc <- NULL
    }
    return(odbc)
}
