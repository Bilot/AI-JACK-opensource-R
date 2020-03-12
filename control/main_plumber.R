
# BILOT AI-jack, main-file for WEB SERVICE
#
# Can be executed either line-by-line, or called
# directly from console/terminal, given that
# necessary configurations have been made.
#
# To run within R: source("main_plumber.R")
# To run from CL: Rscript main_plumber.R
#
# (c) Bilot Oy 2020
# Any user is free to modify this software for their own needs,
# bearing in mind that it comes with no warranty.

# (1) Clear session and set working directory ----
rm(list=ls())

# (2) Source main settings and functions ----
source("config_plumber.r")
setwd(set$main$project_path)

##########################
# To make odbc connection to work:
# SQL Server Configuration manager
# -> Sql Server Network Configuration
# -> Protocols for MSSQLSERVER
# -> Right click to enable TCP/IP

# (3) Source connections ----
odbc <- odbc_connections(set)

# (4) Run plumber ----
# Parameters for the API can be 
# created from a data file, using the 
# parse_params() function.

# feature values:
param <- "param=val1#val2#val3#val4"
# feature names:
param2 <- "param2=nam1#nam2#nam3#nam4"
# feature data types (f = factor, n = numeric)
param3 <- "param3=f#n#n#f"

# Run on command line:
# curl --data "param=val1#val2#val3#val4&param2=nam1#nam2#nam3#nam4&param3=f#n#n#f" "http://localhost:8000/predict"

r <- plumber::plumb("control/plumber_core.R")

# Expose API:
r$run(host="0.0.0.0",port=12345,swagger=TRUE)
    