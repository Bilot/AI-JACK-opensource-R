# BILOT AI-jack DATA READ-module
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.


#' Read data from DB
#'
#' @param query query statement from config object
#' @param odbc_read ODBC-source from config object
#'
#' @return data.frame object.
#'
#' @description
#' Uses either \code{read.table} (\code{base}) or
#' \code{fread} (\code{data.table}), depending on settings.
#'
#' @export

read_db <- function(query, odbc_read) {
    df <- sqlQuery(odbc_read, query)
    return(df)
}

#' Read data from CSV files
#'
#' @param set config object
#'
#' @return data.frame object.
#'
#' @description
#' Uses either \code{read.table} (\code{base}) or
#' \code{fread} (\code{data.table}), depending on settings. The data is read
#' from the directory \code{'source_model'/'source_apply'} within the
#' project folder. If there are several files, the one correct one
#' is identified by mapping the \code{'model_name_part'}-parameter to the 
#' file names to find a match.
#'
#' @export

read_csv <- function(set) {

    # Get data path:
    path <- paste(set$main$project_path, set$read_csv$file_path,
        sep = "/")

    # If several files, map to 'model_name_part'
    path <- path[grep(set$main$model_name_part,path,ignore.case = T)]

    if (!set$read_csv$file_fread) {
        df <- read.table(file = path, header = T, sep = set$read_csv$file_sep,
            dec = set$read_csv$file_dec, na.strings = set$read_csv$file_na,
            stringsAsFactors = F)
    } else {
        df <- fread(input = path, sep = set$read_csv$file_sep,
            header = T, na.strings = set$read_csv$file_na,
            stringsAsFactors = FALSE, data.table = F)
    }

    # Check if ID-column exists:
    if(!(set$main$id %in% colnames(df))){
        # Create dummy ID:
        df[[set$main$id]] <- as.character(1:nrow(df))
    }

    return(df)
}

.set_type <- function(types, column_name, df_i, set) {
    if (types[types[, set$read_variable_types$name_column] ==
        column_name, ][, set$read_variable_types$type_column] %in%
        c("bigint identity", "char") || column_name %in%
        c(set$main$id, set$main$test_train_val)) {
        df_i <- as.character(df_i)
    } else if (types[types[, set$read_variable_types$name_column] ==
        column_name, ][, set$read_variable_types$type_column] %in%
        c("bit", "varchar", "nvarchar")) {
        df_i <- as.factor(df_i)
    } else if (types[types[, set$read_variable_types$name_column] ==
        column_name, ][, set$read_variable_types$type_column] %in%
        c("int", "float", "numeric", "real")) {
        df_i <- as.numeric(df_i)
    } else if (types[types[, set$read_variable_types$name_column] ==
        column_name, ][, set$read_variable_types$type_column] %in%
        c("datetime", "date", "time")) {
        df_i <- as.character(df_i)
    } else {
        df_i <- df_i
    }
    return(df_i)
}

#' Set variable types in data
#'
#' @param set config object
#' @param df data.frame
#' @param odbc_read ODBC-source from config object
#'
#' @return data.frame object.
#'
#' @export

read_variabletypes <- function(set, df, odbc_read) {

    if (set$read_variable_types$types_from_database) {
        variable_types <- sqlColumns(odbc_read, set$odbc$table_r)
        for (i in variable_types[, set$read_variable_types$name_column]) {
            df[, i] <- .set_type(variable_types, i, df[,
                i], set)
        }
    } else {
        # Get data path:
        path <- set$read_variable_types$file_path

        # If several files, map to 'model_name_part'
        if (length(path) > 1){
            path <- path[grep(set$main$model_name_part,path,ignore.case = T)]
        }

        # Read types:
        variable_types <- read.table(path,
            header = T, stringsAsFactors = F, sep = set$main$file_sep)

        # Check if names match:
        cols <- setdiff(colnames(df),variable_types[[set$read_variable_types$name_column]])
        if (length(cols)>0) {
            # Add missing:
            types <- sapply(cols, function(x) class(df[[x]]))
            types <- gsub('factor','varchar',
                          gsub('numeric','real',
                               gsub('character','char',types)))
            df_typ <- data.frame(x1=names(types),x2=as.vector(types))
            names(df_typ) <- c(set$read_variable_types$name_column,
                               set$read_variable_types$type_column)
            variable_types <- rbind(variable_types,df_typ)            
        }

        for (i in variable_types[, set$read_variable_types$name_column]) {
            df[, i] <- .set_type(variable_types, i, df[,
                i], set)
        }
    }
    return(list(Data_with_types = df, Variable_types = variable_types))
}


# WRAPPER: ----

#' Wrapper Function for Reading Data from Source.
#'
#' @param set config object
#' @param odbc ODBC connection object
#'
#' @return main data object
#'
#' @export

data_read = function(set, odbc) {

    start <- Sys.time()
    main <- list()

    # (1) Get raw data: ----
    if (set$main$use_db) {
        main$raw <- handling_trycatch(read_db(set$odbc$query_r,
            odbc$value$odbc_source))
    } else {
        main$raw <- handling_trycatch(read_csv(set))
    }

    if (file.exists(set$read_csv$file_path)) {
        tmp = strsplit(set$read_csv$file_path, "/")[[1]]
        print(paste0("Source data ", "'", tmp[length(tmp)],
            "'", " loaded."), quote = F)

        # (2) Add var-types: ----
        if (set$main$use_db) {
            main$with_types <- handling_trycatch(
                read_variabletypes(set,
                main$raw$value, odbc$value$odbc_source))
        } else {
            main$with_types <- handling_trycatch(
                read_variabletypes(set,
                main$raw$value, NULL))
        }
        print("Variable types defined.", quote = F)

        print_time(start)

        return(main)

    } else {
        stop("ERROR: file not found")
    }
}
