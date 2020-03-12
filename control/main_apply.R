
# BILOT AI-jack, main-file for APPLYING models
#
# Can be executed either line-by-line, or called
# directly from console/terminal, given that
# necessary configurations have been made.
#
# To run within R: source("main_apply.R")
# To run from CL: Rscript main_apply.R
#
# (c) Bilot Oy 2020
# Any user is free to modify this software for their own needs,
# bearing in mind that it comes with no warranty.

rm(list=ls())

# (1) Clear session ----
print("Bilot AI-core",quote = F)
print("=========================",quote = F)
print("Starting new session...",quote = F)

# (2) Get datetime for logs and model names ----
get_datetime <- format(Sys.time(),"%Y_%m_%d_%H_%M")

# (3) Source main settings and functions ----
source("control/config_apply.R")
setwd(set$main$project_path)

# (4) Start logging ----
logging_control(set = set)

# (5) Source connections ----
odbc = open_connections(set = set)

##########################
# To make odbc connection to work:
# SQL Server Configuration manager
# -> Sql Server Network Configuration
# -> Protocols for MSSQLSERVER
# -> Right click to enable TCP/IP

print_message("Starting data prep...")
# (6) Read source data ----
main = data_read(set = set, odbc = odbc)

# (7) Prepare results sets ----
prep = prep_results(set = set,main = main,odbc = odbc)
print_message("Execution row:")
print(prep$execution_row)

# (8) Transform source data ----
main = do_transforms(main = main,
                     set = set,
                     selection = c("clean_special","classify_NA"))

print_message("Starting applying...")

# (9) CREATE PREDICTIONS ----
# Reads applier models based on settings (set):
make_predictions(set = set,main = main,
                 prep = prep,odbc = odbc,
                 useMOJO = T)

print_message("Exiting...")
# (10) Write execution rows ----
write_exec(set = set,prep = prep,odbc = odbc)

# (11) Stop logging, print warnings & close connections ----
clean_up()

    