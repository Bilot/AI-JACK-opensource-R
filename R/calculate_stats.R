# BILOT AI-jack STATISTICS-module
# (c) Bilot Oy 2020
# Any user is free to modify this software for their 
# own needs, bearing in mind that it comes with no warranty.

#' Wrapper Function for Calculating Data Statistics.
#'
#' @param set config object
#' @param main main data object
#' @param methods correlation coefficients to be used
#'
#' @return list of statistics objects.
#' @examples
#' calculate_stats(set, main, methods = "pearson")
#'
#' @details
#' Only correlation has been implemented.
#'
#' @export

calculate_stats <- function(set, main, methods = "pearson") {

    start <- Sys.time()

    # (1) Correlations: ----
    corrs <- handling_trycatch(
        stat_correlation(main$splitted$value,
        set, cors = methods))

    print("Statistics calculated.", quote = F)
    print_time(start)

    return(corrs)
}

#' Calculate correlation coefficients.
#'
#' @param df data.frame object
#' @param set config object
#' @param cors correlation methods to be used (\code{'pearson'},\code{'spearman'}, or \code{'kendall'})
#'
#' @return list of pair-wise correlations per method.
#' @examples
#' stat_correlation(df, set, cors = "pearson")
#'
#' @details
#' Automatically handels feature tyopes, such that only
#' numerical variables are used in operations.
#'
#' @export

stat_correlation <- function(df, set, cors){
    corrs <- list()
    use = set$stat_correlation$filter_type
    #vars =
    for(i in names(df)){
        corr<-list()
        if(any(grepl("kendall", cors))) {
            corr$kendall <- cor(df[[i]][, sapply(df[[i]], is.numeric)],
                                method="kendall", use = use)
        }
        if(any(grepl("pearson", cors))) {
            corr$pearson <- cor(df[[i]][, sapply(df[[i]], is.numeric)],
                                method="pearson", use = use)
        }
        if(any(grepl("spearman", cors))) {
            corr$spearman <- cor(df[[i]][, sapply(df[[i]], is.numeric)],
                                 method="spearman", use = use)
        }
        corrs[[i]]<-corr
    }
    return(corrs)
}
