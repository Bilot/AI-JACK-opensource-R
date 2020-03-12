
# BILOT AI-jack, main-file for MODELLING
#
# Can be executed either line-by-line, or called
# directly from console/terminal, given that
# necessary configurations have been made.
#
# To run within R: source("main_model.R")
# To run from CL: Rscript main_model.R
#
# (c) Bilot Oy 2020
# Any user is free to modify this software for their own needs,
# bearing in mind that it comes with no warranty.

# (1) Clear session ----
print("Bilot AI-core",quote = F)
print("=========================",quote = F)
print("Starting new session...",quote = F)
rm(list=ls())

# (2) Get datetime for logs and model names ----
get_datetime <- format(Sys.time(),"%Y_%m_%d_%H_%M")

# (3) Source main settings and functions ----
source("control/config_model.R")
setwd(set$main$project_path)

# (4) Start logging ----
logging_control(set)

# (5) Source connections ----
odbc = open_connections(set)

# To make odbc connection to work:
# SQL Server Configuration manager
# -> Sql Server Network Configuration
# -> Protocols for MSSQLSERVER
# -> Right click to enable TCP/IP

print_message("Starting data prep...")
# (6) Read source data ----
main = data_read(set = set, odbc = odbc)
set = check_label(main, set)

# (7) Prepare results sets ----
prep = prep_results(set = set,main = main,odbc = odbc)

# (8) Transform source data ----
main = do_transforms(main = main,set = set,prep = prep)

# (9) Split data into train, test, val ----
main = split_data(main = main,set = set)

# (10) Special transforms ----
main = entropy_recategorization(main = main,
                                set = set, prep = prep)
# (11) Calculate statistics ----
main$stats <- calculate_stats(set = set,
                              main = main,
                              methods = "spearman")


print_message("Starting modelling...")
# (12) Start H2O cluster ----
start_h2o(set = set)

# (13) Create prediction models ----
create_models(set = set,main = main,
              prep = prep,odbc = odbc)

print_message("Exiting...")
# (14) Write execution rows ----
write_exec(set = set,prep = prep,odbc = odbc)

# (15) Save data object ----
save_data(main,set,prep,get_datetime)

# (16) Stop logging, print warnings & close connections ----
clean_up()
    