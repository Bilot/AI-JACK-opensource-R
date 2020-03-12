# BILOT AI-jack DATA WRITE-module
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.


#' Write data to DB
#'
#' @param channel from \code{odbc} object (\code{odbc_metadata})
#' @param x data to be written
#' @param name name of database table
#'
#' @export

write_db <- function(channel, x, name) {
    sqlSave(channel, x, tablename = paste(name), append = TRUE,
        rownames = FALSE, colnames = FALSE, verbose = FALSE,
        safer = TRUE, addPK = FALSE, fast = TRUE, test = FALSE,
        nastring = NULL)
}

#' Write data to file
#'
#' @param set config object
#' @param df data to be written
#' @param name name of target file
#' @param append wether to extend a file (default = TRUE)
#' @param colnames whether to write col.names (default = FALSE)
#'
#' @export

write_csv <- function(set, df, name, append = TRUE,
                      colnames = FALSE) {

    if (is.null(set$main$file_fwrite))
        set$main$file_fwrite = F

    if (!set$main$file_fwrite) {
        write.table(x = df, file = paste0(name, ".csv"),
            append = append, quote = F, row.names = F,
            sep = ";", col.names = colnames)
    } else {
        fwrite(x = df, file = paste0(set$write_csv$model_path_results,
            "/", name, ".csv"), append = append)
    }
}

#' Write Execution Rows to table
#'
#' @param set config object
#' @param prep summary object
#' @param odbc odbc connection object
#'
#' @description
#' The following info is saved:
#' (1) execution time & query
#' (2) data summary
#' (3) variables used in modeling
#'
#' @export

write_exec = function(set, prep, odbc) {

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
            set$csv$result$exec, sep = "/"), append = T)
        write_csv(set, prep$summary_table, paste(set$csv$result$prefix,
            set$csv$result$metad, sep = "/"), append = T)
        write_csv(set, prep$columns, paste(set$csv$result$prefix,
            set$csv$result$cols, sep = "/"), append = T)
    }
    print("Execution rows written.", quote = F)
}
