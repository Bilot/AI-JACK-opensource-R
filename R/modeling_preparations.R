# BILOT AI-jack PREPARATION-module
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.

#' Function for generating tables for execution,
#' metadata and data columns
#'
#' @param runid execution ID
#' @param df data.frame
#' @param types variable types
#' @param query data load call
#'
#' @return list with execution_row, summary_table, and columns
#'
#' @export

make_tables <- function(runid, df, types, query){

  # (1) Create execution row: ----
  execution_row <- data.frame(
    executionid = runid,
    preddate = format(Sys.time(), "%d.%m.%Y"),
    predtime = format(Sys.time(), "%H:%M:%S"),
    query = query
  )

  # (2) Save columnset metadata: ----
  summary_table <- character(8)
  for(ii in names(df)) {
    summary_table <- rbind_diff(summary_table,
                                c(ii,summary(df[ii])))
  }
  summary_table <- summary_table[-1,]
  colnames(summary_table) <- character(8)

  # Replace NA-values
  summary_table[is.na(summary_table)] <- ""

  # Create datasummary table
  summary_table <- data.frame(
    executionid = as.numeric(runid),
    column = summary_table[,1],
    stat1 = summary_table[,2],
    stat2 = summary_table[,3],
    stat3 = summary_table[,4],
    stat4 = summary_table[,5],
    stat5 = summary_table[,6],
    stat6 = summary_table[,7],
    stat7 = summary_table[,8]
  )

  # (3) Create columnname dataframe: ----
  columns <- data.frame(
    executionid = as.numeric(runid),
    column = types$COLUMN_NAME,
    label = as.numeric(types$COLUMN_NAME %in%
                         set$main$label),
    used_in_model = as.numeric(
      types$COLUMN_NAME %in% colnames(df) &
        types$COLUMN_NAME %in%
        colnames(df)[which(sapply(df, class)!="character")])
  )

  # (4) Return output: ----
  return(
    list(
      execution_row = execution_row,
      summary_table = summary_table,
      columns = columns
    )
  )
}

#' Function for extracting and factor levels
#'
#' @param df data.frame
#'
#' @return list of factor levels per feature
#'
#' @export

get_levels <- function(df) {
  loop_variables <- colnames(df[sapply(df, is.factor)])
  a <- list()
  if(!is.character0(loop_variables)) {
    for(i in loop_variables){
      a <- c(a, list(i = c(as.character(levels(df[,i])))))
    }
    names(a) <- loop_variables
  } else {
    a <- NULL
  }
  return(a)
}

#' Function for Data Splitting
#'
#' @description
#' Split data based on the \code{"test_train_val"}
#' column in the data table.
#'
#' @param main main data object
#' @param set config object
#'
#' @return main object with splitted data.
#'
#' @export

split_data <- function(main,set){

  start <- Sys.time()

  X = main$constants_deleted$value
  main$splitted <- handling_trycatch(
    split(X, X[,set$main$test_train_val])
  )

  names(main$splitted$value) <- c('train','test','val')

  main$splitted$value <- lapply(
    main$splitted$value,function(x) x[,setdiff(names(x),'test_train_val')])

  print("Data splitted.", quote = F)
  print_time(start)

  return(main)
}
