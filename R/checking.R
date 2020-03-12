# BILOT AI-jack DATA CHECK-module
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.


#' Check label type
#'
#' @param main main data object
#' @param set config object
#'
#' @return set
#'
#' @description
#' This function makes sure the set$main$labeliscategory
#' parameter is correctly set, according to the data type
#' of the label column. 
#'
#' @export

check_label = function(main, set) {
    x = main$with_types$value$Data_with_types
    label = set$main$label
    test = is.factor(x[[label]]) | is.character(x[[label]])
    if (test) {
        set$main$labeliscategory = TRUE
    } else {
        if (is.numeric(x[[label]])){
            set$main$labeliscategory = FALSE
        }
    }
    return(set)
}

#' Check for Missing Data.
#'
#' @param main main data object
#' @param set config object
#' @param prep summary object
#'
#' @description
#' This function can be used to halt the workflow, if the data contains
#' missing values.
#'
#' @export

check_missing = function(main, set, prep) {

    tmp = which(colSums(is.na(main[[length(main)]]$value)) >
        0)
    test = ifelse(length(tmp) == 0, "NULL", names(tmp))

    if (test != "NULL") {
        df = data.frame(executionid = prep$runid, model_name = "Missing values in data",
            label = set$main$label, phase = "check_missing",
            description = paste(test, collapse = ", "))
        dest = ifelse(set$main$use_db, set$odbc$result$war,
            set$csv$result$war)
        write_csv(set, df = df, name = dest, append = T)
        stop("Check your data, no missing values allowed!")
    }
}

#' Check if object is character(0).
#'
#' @param a object
#'
#' @return NULL if true, else returns a
#'
#' @export
char0 <- function(a) { if(identical(a, character(0))==TRUE) {NULL} else {a} }

#' Check if object is character(0).
#'
#' @param x object
#'
#' @return logical
#'
#' @export
is.character0 <- function(x) { is.character(x) && length(x) == 0L }

#' Check if object is integer(0).
#'
#' @param x object
#'
#' @return logical
#'
#' @export
is.integer0 <- function(x) { is.integer(x) && length(x) == 0L }

#' Check if object is factor(0).
#'
#' @param x object
#'
#' @return logical
#'
#' @export
is.factor0 <- function(x) { is.factor(x) && length(x) == 0L }
