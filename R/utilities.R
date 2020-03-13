# BILOT AI-Jack UTILITIES-module
# (c) Bilot Oy 2020
# Any user is free to modify this software for their
# own needs, bearing in mind that it comes with no warranty.


#' Creates a project template to given
#' path
#'
#' @param path system path where to create a project
#'
#' @description
#' The function creates a new directory (if it
#' does not exist) and within that directory, four sub-
#' directories: \code{output_model}, \code{output_apply},
#' \code{source_model}, and \code{source_apply}. Directories
#' \code{output_model} and \code{output_apply} will be further
#' populated by \code{.scv} files that contain output tables.
#' In addition, a directory with name \code{control} is created,
#' which is populated with .R scripts for controlling execution.
#' To delete a project, call \code{delete_project(path)}
#' function.
#'
#' @export

init_aijack <- function(path) {

    if (!dir.exists(path)) {
        dir.create(path)
    }

    sep = ifelse(grepl("/$", path), "", "/")
    directories = list(
        source_model = paste(path, "source_model",sep = sep),
        source_apply = paste(path, "source_apply",sep = sep),
        output_model = paste(path, "output_model",sep = sep),
        output_apply = paste(path, "output_apply",sep = sep),
        output_plumber = paste(path, "output_plumber",sep = sep),
        logs = paste(path, "logs", sep = sep),
        control = paste(path, "control", sep = sep)
        )

    output_folders = c("discretization", "factor_levels",
        "data_objects", "models", "parameters", "results")
    output_model_dirs = paste(directories$output_model,
        output_folders, sep = "/")

    # Create 1st directory level: ----
    for (d in names(directories)) {
        if (!dir.exists(directories[[d]])) {
            dir.create(directories[[d]], recursive = T)
        }
    }

    # Create 2nd directory level: ----
    for (d in output_model_dirs) {
        if (!dir.exists(d)) {
            dir.create(d)
        }
    }

    # Fill directories: ----
    add = list("cut-points and factor levels for discretised features",
        "factor levels in the training data", "data objects (main-list)",
        "trained models", "parameter objects", "tables of modelling results",
        "error and warning logs")
    names(add) = output_folders
    ending = "for each execution ID\n"

    readmes = lapply(as.list(output_folders), function(x) {
        paste0("This directory will be populated with ",
            add[[x]], ", ", ending)
    })
    names(readmes) = output_folders
    for (d in output_folders) {
        where = paste(directories$output_model, d, "README.txt",
            sep = "/")
        cat(readmes[[d]], file = where)
    }

    # Fill output_model/results: ----
    tables = c("accuracy", "column_importance", "models",
        "validation", "execution", "coefficients", "metadata",
        "columns", "warning_error")
    table_list = paste0(path, "/output_model/results/",
        tables, ".csv")
    table_list = as.list(table_list)
    names(table_list) = tables

    if (!file.exists(table_list$accuracy)) {
        tab = data.frame(executionid = numeric(0), time = character(0),
            label = character(0), model_name = character(0),
            metric = character(0), value_train = numeric(0),
            value_test = numeric(0), value_val = numeric(0))
        write.csv2(tab, table_list$accuracy, row.names = F)
    }
    if (!file.exists(table_list$column_importance)) {
        tab = data.frame(executionid = numeric(0), label = character(0),
            model_name = character(0), variable = character(0),
            relative_importance = numeric(0), scaled_importance = numeric(0),
            percentage = numeric(0))
        write.csv2(tab, table_list$column_importance, row.names = F)
    }
    if (!file.exists(table_list$models)) {
        tab = data.frame(executionid = numeric(0), label = character(0),
            model_name = character(0), apply = numeric(0))
        write.csv2(tab, table_list$models, row.names = F)
    }
    if (!file.exists(table_list$validation)) {
        tab = data.frame(executionid = numeric(0), model_name = character(0),
            row_identifier = character(0), obs = numeric(0),
            pred = numeric(0), prob = numeric(0))
        write.csv2(tab, table_list$validation, row.names = F)
    }
    if (!file.exists(table_list$execution)) {
        tab = data.frame(executionid = numeric(0), preddate = character(0),
            predtime = character(0), query = character(0))
        write.csv2(tab, table_list$execution, row.names = F)
    }
    if (!file.exists(table_list$coefficients)) {
        tab = data.frame(executionid = numeric(0), label = character(0),
            variable = character(0), coef = character(0),
            model_name = character(0))
        write.csv2(tab, table_list$coefficients, row.names = F)
    }
    if (!file.exists(table_list$metadata)) {
        tab = data.frame(executionid = numeric(0), column = character(0),
            stat1 = character(0), stat2 = character(0),
            stat3 = character(0), stat4 = character(0),
            stat5 = character(0), stat6 = character(0),
            stat7 = character(0))
        write.csv2(tab, table_list$metadata, row.names = F)
    }
    if (!file.exists(table_list$columns)) {
        tab = data.frame(executionid = numeric(0), column = character(0),
            label = character(0), used_in_model = character(0))
        write.csv2(tab, table_list$columns, row.names = F)
    }
    if (!file.exists(table_list$warning_error)) {
        tab = data.frame(executionid = numeric(0), model_name = character(0),
            label = character(0), phase = character(0),
            description = character(0))
        write.csv2(tab, table_list$warning_error, row.names = F)
    }

    # Fill output_apply/results: ----
    cat("This directory will be populated with tables of model application execution",
        file = paste0(path, "/output_apply/README.txt"))
    tables = c("execution", "execution_model", "predictions")
    table_list = paste0(path, "/output_apply/", tables,
        ".csv")
    table_list = as.list(table_list)
    names(table_list) = tables

    if (!file.exists(table_list$execution)) {
        tab = data.frame(executionid = numeric(0), preddate = character(0),
            predtime = character(0), query = character(0))
        write.csv2(tab, table_list$execution, row.names = F)
    }
    if (!file.exists(table_list$execution_model)) {
        tab = data.frame(executionid = numeric(0), label = character(0),
            model_name = character(0), rowid = character(0))
        write.csv2(tab, table_list$execution_model, row.names = F)
    }
    if (!file.exists(table_list$predictions)) {
        tab = data.frame(rowid = numeric(0), id = character(0),
            type = character(0), pred = character(0), model = character(0))
        write.csv2(tab, table_list$predictions, row.names = F)
    }

    # Fill output_plumber: ----
    cat("This directory will be populated with a table of API predictions",
        file = paste0(path, "/output_plumber/README.txt"))
    tables = "predictions"
    table_list = paste0(path, "/output_plumber/", tables,
        ".csv")
    table_list = as.list(table_list)
    names(table_list) = tables
    if (!file.exists(table_list$predictions)) {
        tab = data.frame(ID = character(0),
            predict = numeric(0),
            model_name = character(0),
            predtime = character(0),
            notions = character(0),
            parameters = character(0))
        write.csv2(tab, table_list$predictions, row.names = F)
    }

    print("Directory structure created.",quote = F)

    # Create config_model.R: ----
    sink(
      file=paste0(directories$control,"/config_model.R")
    )
    cat(
      '
# CONFIG_MODEL
#
# This config-file is used to
# parameterize AI-jack modules
# when performing model training.
#
# (c) Bilot Oy 2020
# Any user is free to modify this software for their own needs,
# bearing in mind that it comes with no warranty.

# (1) MAIN SETTINGS: ----
set <- list()
project_path <- "<PATH TO PROJECT>"

# (1.1) Input/Output: ----
set$main <- list(
  project_path = project_path,

	# DATA:
	use_db = FALSE,
	label = "<LABEL NAME>",
	model_name_part = "<PROJECT NAME>",
	id = "id",
	test_train_val = "test_train_val",
	labeliscategory = F,

	# OUTPUT:
	model_path = "output_model/models",
	result_path = "output_model/result",
	append_predicts = FALSE
)

# (1.2) Technical: ----
set$main$write_to_log = FALSE # write logs to file?
set$main$log_path = "logs"
set$main$num_cores = parallel::detectCores()-1
set$main$seed = 1234
set$main$min_mem_size ="5g"
set$main$max_mem_size ="7g"

# (2) FILE CONNECTION PARAMETERS: ----
# (2.1) Files: ----
files = dir(paste(set$main$project_path,"source_model",sep="/"))
set$main$data_path = paste0("source_model/",grep("csv",files,value = T))
set$main$type_path = paste0("source_model/",grep("types",files,value = T))
set$main$file_sep = ";"
set$main$file_dec = "."
set$main$file_fread = FALSE
set$main$file_fwrite = FALSE

# (2.2) Variable types: ----
set$read_variable_types	<-	list(
  # CSV-source column types (path)
  file_path	=	paste(set$main$project_path,
                    set$main$type_path,sep="/"),
  # Database specific column names (SQL SERVER)
  name_column =	"COLUMN_NAME",
  type_column =	"TYPE_NAME",
  # Read column types from db?
  types_from_database	=	set$main$use_db
)

# (3) ODBC CONNECTION PARAMETERS: ----
set$odbc = list(
  # Source server name
  server_r = "localhost",
  # Source database name
  database_r = "modellingsource",
  # Source table/view name
  table_r = "",
  # Source user name
  user_r = "",
  # Source user password
  user_pw_r = "",

  # Result server name (models)
  server_m = "localhost",
  database_m = "modellingmetadata",
  server_p = "localhost",
  database_p = "modellingprediction",
  server_v = "localhost",
  database_v = "modellingvalidation"
)

set$odbc$con_string <- "Driver={SQL Server Native Client 11.0};server=XXXXXXX;database=modellingvalidation-dev;Uid=XXXXXXX;Pwd=XXXXXXX;Encrypt=yes"

set$odbc$result = list(
  prefix = set$main$result_path,
  exec = "execution",
  coef = "coefficient",
  acc = "accuracy",
  cols = "columns",
  metad = "metadata",
  war = "warning_error",
  model = "models",
  imp = "column_importance",
  val = "validation",
  val_reg = "validation"
)
#Source queries
set$odbc$query_r <- paste("SELECT * FROM",
                          set$odbc$table_r, sep=" ")

# (4) DATA PREP PARAMETERS: ----
set$trans_entropy <- list(
  jitter_factor = 0.0001,
  skip_na = FALSE
)
set$stat_correlation	<-	list(
  #Correlation pairs
  filter_type	=	"pairwise.complete.obs"
)
set$trans_classifyNa	<-	list(
  limit	=	10
)
set$split_data <- list(
  # Set parameters such that their
  # sum is < 1; rest of the data
  # goes to validation
  prop_train = 0.7,
  prop_test = 0.2
)

# (5) MODELING PARAMETERS: ----
set$model <- list(
  label = set$main$label,
  discretize = FALSE,
  cols_not_included = c(set$main$label,
                        set$main$id,
                        set$main$test_train_val),
  # Models available for training (one can specify only those that
  # they wish to train):
  train_models = c("glm","randomForest","decisionTree",
                   "deeplearning","automl","gbm","xgboost"),
  acc_metric_reg = "mae",
  acc_metric_class = "auc",
  probth = 0.5,
  random_seed = 1234
)

# (5.1) General Model Training Parameters: ----
set$model$general <- list(
  stopping_metric = ifelse(set$main$labeliscategory,
                           set$model$acc_metric_class,
                           set$model$acc_metric_reg),
  stopping_rounds = 5,
  stopping_tolerance = 0.001,
  strategy = "RandomDiscrete",
  random_seed = set$main$seed,
  score_tree_interval = 5
)

# (5.2) Model Optimization Parameters: ----
# GLM model: ----
set$model$glm <- list(
  alpha = 0.5,
  lambda_min_ratio = -1,
  nfold = 5,
  family = ifelse(
    set$main$labeliscategory,
    "binomial",
    "gaussian"
  )
)

set$model$param_grid <- list()

# decisionTree model: ----
set$model$param_grid$decisionTree <- list(
  fixed = list(
    max_runtime_secs = 3000,
    max_models = 10
  ),
  hyper_params = function(x){
    list(
      ntrees = 1,
      max_depth = seq(3,x,by = 2)
    )
  }
)
# randomForest model: ----
set$model$param_grid$randomForest <- list(
  fixed = list(
    max_runtime_secs = 6000,
    max_models = 10
  ),
  hyper_params = function(x){
    # this is a function for compatibility
    list(
      ntrees = c(10, 30, 50, 100),
      nbins = c(7, 5, 3),
      sample_rate = c(0.75, 0.4)
    )
  }
)
# DeepLearning model: ----
set$model$param_grid$deeplearning <- list(
  fixed = list(
    max_runtime_secs = 10000,
    max_models = 30,
    score_validation_samples = 50,
    score_duty_cycle = 0.05,
    adaptive_rate = F,
    momentum_start = 0.5,
    momentum_stable = 0.9,
    momentum_ramp = 1e7,
    max_w2 = 1
  ),
  hyper_params = list(
    hidden = list(c(20), c(20,20)),
    input_dropout_ratio = c(0.05, 0.10, 0.20),
    rate = c(0.0003),
    rate_annealing = c(1e-3, 1e-4, 1e-2),
    l1 = c(1e-4, 1e-3),
    l2 = c(1e-4, 1e-3),
    epochs = c(20, 50),
    activation = c("Rectifier","Tanh","Maxout",
                   "RectifierWithDropout","TanhWithDropout",
                   "MaxoutWithDropout"),
    distribution = "tweedie",
    tweedie_power = c(1.2, 1.4, 1.6)
  )
)
# Gradient Boosting model: ----
set$model$param_grid$gbm <- list(
  fixed = list(
    max_runtime_secs = 8000,
    max_models = 10,
    score_tree_interval= 5
  ),
  hyper_params1 = function(x){
    list(
      max_depth = seq(3,x,by = 2)
    )
  },
  hyper_params2 = function(minDepth, maxDepth, train){
    list(
      max_depth = seq(minDepth,maxDepth,1),
      sample_rate = seq(0.7,0.9,0.01),
      col_sample_rate = seq(0.6,1,0.01),
      col_sample_rate_per_tree = seq(0.6,1,0.01),
      col_sample_rate_change_per_level = seq(0.9,1.1,0.01),
      nbins = 2^seq(2,7,1),
      nbins_cats = 2^seq(2,10,1),
      min_rows =  2^seq(2,log2(nrow(train))-3,1),
      min_split_improvement = c(1e-6,1e-5),
      histogram_type = c("QuantilesGlobal", "UniformAdaptive", "RoundRobin"),
      ntrees = 200,
      learn_rate = 0.05,
      learn_rate_annealing = 0.99
    )
  }
)
# Extreme Gradient Boosting model: ----
set$model$param_grid$xgboost <- list(
  fixed = list(
    max_runtime_secs = 8000,
    max_models = 10,
    score_tree_interval= 0
  ),
  hyper_params1 = function(x){
    list(
      max_depth = seq(3,x,by = 2)
    )
  },
  hyper_params2 = function(minDepth, maxDepth, train){
    list(
      max_depth = seq(minDepth,maxDepth,1),
      sample_rate = seq(0.7,0.9,0.01),
      col_sample_rate = seq(0.6,1,0.01),
      col_sample_rate_per_tree = seq(0.6,1,0.01),
      ntrees = 50,
      learn_rate = c(0.1,0.2,0.3)
    )
  }
)
# AutoML model: ----
set$model$automl <- list(
  max_num_models = 2,
  max_runtime_secs = 100,
  save_n_best_models = 2,
  exclude = c("GLM","GBM","DRF","StackedEnsemble")
)

# (6) CSV Connection Parameters: ----
# (6.1) Configure data read: ----
set$read_csv 	<-	list(
  # CSV-source path and file name
  file_path	=	set$main$data_path,
  # File separator
  file_sep	=	set$main$file_sep,
  # NA-coding in source
  file_na		=	c("NA", "", "NULL"),
  # Read with using fread-package?
  file_fread	=	set$main$file_fread,
  # File decimal separator
  file_dec	=	set$main$file_dec
)
# (6.2) Configure result output: ----
set$csv$result <- list(
  prefix = paste(set$main$project_path,"output_model/results",sep="/"),
  exec = "execution",
  coef = "coefficients",
  acc = "accuracy",
  cols = "columns",
  metad = "metadata",
  war = "warning_error",
  model = "models",
  imp = "column_importance",
  val = "validation",
  val_reg = "validation",
  sep=";"
)

print("Settings created.", quote = F)

')
  sink()

  # Create config_apply.R: ----
  sink(
    file=paste0(directories$control,"/config_apply.R")
  )
  cat(
    '
# CONFIG_APPLY
#
# This confiq-file is used to
# parameterize AI-jack modules
# when performing model application.
#
# (c) Bilot Oy 2020
# Any user is free to modify this software for their own needs,
# bearing in mind that it comes with no warranty.

# (1) MAIN SETTINGS: ----
set <- list()
project_path = "<PATH TO PROJECT>"

# (1.1) Input/Output: ----
set$main	 	<-	list(
  project_path = project_path,

  # DATA:
  use_db = FALSE,
  label = "<LABEL NAME>",
  labeliscategory = T,
  model_name_part = "<PROJECT NAME>",
  id = "id",

  # PATHS:
  model_path = "output_model/models",
  model_model_path = "output_model/results",

  # PREDICTION:
  # which row to use from Model_model-table?
  model_row = 2
)

# (1.2) Technical: ----
set$main$write_to_log = FALSE
set$main$log_path = "logs"
set$main$num_cores = 1
set$main$seed = 1234
set$main$min_mem_size ="3g"
set$main$max_mem_size ="3g"

# (2) FILE CONNECTION PARAMETERS: ----
# (2.1) Files: ----
files = dir(paste(set$main$project_path,"source_model",sep="/"))
set$main$data_path = paste0("source_apply/",grep("csv",files,value = T))
set$main$type_path = paste0("source_apply/",grep("types",files,value = T))
set$main$file_sep = ";"
set$main$file_dec = "."
set$main$result_path = "results"
set$main$file_fread = FALSE
set$main$file_fwrite = FALSE

# (2.2) Variable types: ----
set$read_variable_types	<-	list(
  # CSV-source column types (path)
  file_path	=	paste(set$main$project_path,
                    set$main$type_path,sep="/"),
  # Database specific column names (SQL SERVER)
  name_column =	"COLUMN_NAME",
  type_column =	"TYPE_NAME",
  # Read column types from db?
  types_from_database	=	set$main$use_db
)

# (3) ODBC CONNECTION PARAMETERS: ----
set$odbc = list( )

# (4) CSV Connection Parameters: ----
# (4.1) Configure data read: ----
set$read_csv 	<-	list(
  # CSV-source path and file name
  file_path	=	set$main$data_path,
  # File separator
  file_sep	=	set$main$file_sep,
  # NA-coding in source
  file_na		=	c("NA", "", "NULL"),
  # Read with using fread-package?
  file_fread	=	set$main$file_fread,
  # File decimal separator
  file_dec	=	set$main$file_dec
)
# (4.2) Configure result output: ----
set$csv$result = list(
  prefix = paste(set$main$project_path,"output_apply",sep="/"),
  exec = "execution",
  pred = "predictions",
  exec_model = "execution_model",
  model = "models",
  sep = ";"
)

print("Settings created", quote = F)
    ')
  sink()

  # Create config_plumber.R: ----
  sink(
    file=paste0(directories$control,"/config_plumber.R")
  )
  cat(
    '
# CONFIG_PLUMBER
#
# This confiq-file is used to
# parameterize AI-jack modules
# when exposing model via API.
#
# (c) Bilot Oy 2020
# Any user is free to modify this software for their own needs,
# bearing in mind that it comes with no warranty.

set <- list()
project_path = "<PATH TO PROJECT>"

set$main <-	list(
  # FILE connection:
  use_db = F,
  id = "id",
  file_sep = ";",

  # PREDICTION:
  model_row = 2,

  # PATHS:
  type_path = paste(project_path,"source_model/boston_types.txt",sep="/"),
  model_path = paste(project_path,"output_model/models",sep="/"),
  model_name_part = "<PROJECT NAME>",
  model_model_path = paste(project_path,"output_model/results",sep="/"),
  model_model_file = "models",
	log_path = paste(project_path,"logs",sep="/"),

	# COMPUTATION:
	num_cores = 1,
	min_mem_size = 3,
	max_mem_size = 3
)

# OUTPUT:
set$write_csv$file_name <- paste(project_path,"output_plumber/predictions.csv",sep="/")

set$read_variable_types	<-	list(
  # CSV-source column types (path)
  file_path	=	set$main$type_path,
  # Database specific column names (SQL SERVER)
  name_column =	"COLUMN_NAME",
  type_column =	"TYPE_NAME",
  # Read column types from db?
  types_from_database	=	set$main$use_db
)

set$odbc <- list(
  model_table = "models",
  server_r = "",
  database_r = "modellingvalidation-dev",
  user_r = "",
  user_pw_r = "",
  model_table_pred = "predictions"
)

# Source queries
set$odbc$query_r <- paste("SELECT * FROM",
                          set$odbc$model_table, " WHERE apply=1", sep=" ")
    '
  )
  sink()

  # Create main_model.R: ----
  sink(
    file=paste0(directories$control,"/main_model.R")
  )
  cat(
    '
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
library(AIjack)

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
    '
  )
  sink()

  # Create main_apply.R: ----
  sink(
    file=paste0(directories$control,"/main_apply.R")
  )
  cat(
    '
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
library(AIjack)

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

    '
  )
  sink()

  # Create main_plumber.R: ----
  sink(
    file=paste0(directories$control,"/main_plumber.R")
  )
  cat(
    '
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
library(AIjack)

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
    '
  )
  sink()

  # Create plumber_core.R: ----
  sink(
    file=paste0(directories$control,"/plumber_core.R")
  )
  cat(
    '
#* @param param parameter string values
#* @param param parameter string colnames
#* @post /predict
function(param, param2, param3){
    # Parse data.frame:
    df <- create_df(param, param2, param3)
    # Predict:
    plumber_predict(df, set, param, param2, param3, "")
}
    '
  )
  sink()

  print("Control files created.", quote = F)
  print("AI-jack project initiation complete.", quote = F)
}


#' Generate prep-object (summary)
#'
#' @param main main data object
#' @param set config object
#' @param odbc DB connection object
#'
#' @return prep object
#'
#' @export

prep_results <- function(set, main, odbc) {

    start <- Sys.time()
    types <- main$with_types$value$Variable_types
    data <- main$with_types$value$Data_with_types

    if (set$main$use_db) {
        prep <- db_preparations(set, types, data, odbc$value$odbc_metadata)
    } else {
        prep <- csv_preparations(set, main)
    }

    print("Result tables pre-created.", quote = F)
    print_time(start)

    return(prep)
}

#' Handling excution logs
#'
#' @param set config object
#'
#' @description
#' Depending on settings, either prints logs to
#' console or writes them files.
#'
#' @export

logging_control <- function(set) {

    if (set$main$write_to_log) {
        log_file <- file(paste(set$main$project_path, "/",
            set$main$log_path, "/log_modelling_", get_datetime,
            ".txt", sep = ""), open = "wt")
        sink(log_file, type = "message")
        target <- ifelse(set$main$use_db, "DB", "file")
        print(paste0("Logs will be written to ", target),
            quote = F)
    } else {
        print("Logs will be printed to console", quote = F)
    }
}

#' Function for printing messages
#'
#' @param text message to print
#'
#' @export

print_message <- function(text) {
    print("", quote = FALSE)
    print(paste0(rep("-", nchar(text)), collapse = ""),
        quote = FALSE)
    print(text, quote = FALSE)
    print(paste0(rep("-", nchar(text)), collapse = ""),
        quote = FALSE)
}

#' Function for printing execution time
#'
#' @param start time of execution start
#'
#' @export

print_time <- function(start) {
    x = round((Sys.time() - start), 1)
    print(paste("   Took:", as.numeric(x), attr(x, "units")),
          quote = F)
}

#' Catching up warnings-messages
#'
#' @param exp expression to evaluate
#'
#' @return a list with the evaluation result and warnings
#'
#' @export


withWarnings <- function(expr) {
    myWarnings <- NULL
    wHandler <- function(w) {
        myWarnings <<- c(myWarnings, list(w))
        invokeRestart("muffleWarning")
    }
    val <- withCallingHandlers(expr, warning = wHandler)
    return(list(value = val, warnings = myWarnings))
}


#' Catch errors, warnings, execution time
#'
#' @param expr function call
#'
#' @description
#' This module is used in front of all functions and
#' commands
#'
#' @examples
#' handling_trycatch(data_read(set, odbc))
#'
#' @export

handling_trycatch <- function(expr) {
    from <- deparse(sys.calls()[[sys.nframe()]])
    start <- Sys.time()
    warnings <- NULL
    f.warning <- function(war) {
        warnings <<- list(paste(war, collapse = " | "))
        invokeRestart("muffleWarning")
    }
    temp <- list(value = withCallingHandlers(tryCatch(expr,
                                                      error = function(err) list(from = from, error = paste(paste(err),
                                                                                                            collapse = " | "))), warning = f.warning),
                 warning = warnings)
    return(list(value = temp$value, warning = temp$warning[[1]],
                execution_time = round((Sys.time() - start), 1)))
}


#' Combining tables of different size
#'
#' @description
#' Copied from internal function rbind.na in package qpcR.
#'
#' @export
rbind_diff <- function(..., deparse.level = 1) {
    na <- nargs() - (!missing(deparse.level))
    deparse.level <- as.integer(deparse.level)
    stopifnot(0 <= deparse.level, deparse.level <= 2)
    argl <- list(...)
    while (na > 0 && is.null(argl[[na]])) {
        argl <- argl[-na]
        na <- na - 1
    }
    if (na == 0)
        return(NULL)
    if (na == 1) {
        if (isS4(..1))
            return(rbind2(..1)) else return(matrix(..., nrow = 1))
    }
    if (deparse.level) {
        symarg <- as.list(sys.call()[-1L])[1L:na]
        Nms <- function(i) {
            if (is.null(r <- names(symarg[i])) || r ==
                "") {
                if (is.symbol(r <- symarg[[i]]) || deparse.level ==
                  2)
                  deparse(r)
            } else r
        }
    }
    if (na == 0) {
        r <- argl[[2]]
        fix.na <- FALSE
    } else {
        nrs <- unname(lapply(argl, ncol))
        iV <- sapply(nrs, is.null)
        fix.na <- identical(nrs[(na - 1):na], list(NULL,
            NULL))
        if (deparse.level) {
            if (fix.na)
                fix.na <- !is.null(Nna <- Nms(na))
            if (!is.null(nmi <- names(argl)))
                iV <- iV & (nmi == "")
            ii <- if (fix.na)
                2:(na - 1) else 2:na
            if (any(iV[ii])) {
                for (i in ii[iV[ii]]) if (!is.null(nmi <- Nms(i)))
                  names(argl)[i] <- nmi
            }
        }
        nCol <- as.numeric(sapply(argl, function(x) if (is.null(ncol(x))) length(x) else ncol(x)))
        maxCol <- max(nCol, na.rm = TRUE)
        argl <- lapply(argl, function(x) if (is.null(ncol(x)))
            c(x, rep(NA, maxCol - length(x))) else cbind(x, matrix(, nrow(x), maxCol - ncol(x))))
        namesVEC <- rep(NA, maxCol)
        for (i in 1:length(argl)) {
            CN <- colnames(argl[[i]])
            m <- !(CN %in% namesVEC)
            namesVEC[m] <- CN[m]
        }
        for (j in 1:length(argl)) {
            if (!is.null(ncol(argl[[j]])))
                colnames(argl[[j]]) <- namesVEC
        }
        r <- do.call(rbind, c(argl[-1L], list(deparse.level = deparse.level)))
    }
    d2 <- dim(r)
    colnames(r) <- colnames(argl[[1]])
    r <- rbind2(argl[[1]], r)
    if (deparse.level == 0)
        return(r)
    ism1 <- !is.null(d1 <- dim(..1)) && length(d1) == 2L
    ism2 <- !is.null(d2) && length(d2) == 2L && !fix.na
    if (ism1 && ism2)
        return(r)
    Nrow <- function(x) {
        d <- dim(x)
        if (length(d) == 2L)
            d[1L] else as.integer(length(x) > 0L)
    }
    nn1 <- !is.null(N1 <- if ((l1 <- Nrow(..1)) && !ism1) Nms(1))
    nn2 <- !is.null(N2 <- if (na == 2 && Nrow(..2) && !ism2) Nms(2))
    if (nn1 || nn2 || fix.na) {
        if (is.null(rownames(r)))
            rownames(r) <- rep.int("", nrow(r))
        setN <- function(i, nams) rownames(r)[i] <<- if (is.null(nams))
            "" else nams
        if (nn1)
            setN(1, N1)
        if (nn2)
            setN(1 + l1, N2)
        if (fix.na)
            setN(nrow(r), Nna)
    }
    r
}

#' Clearing result tables
#'
#' @param result_path path to results directory
#'
#' @description
#' This function clears result tables
#' by removing all rows, except for the
#' column names. As input, give the path
#' where the \code{result} folder is located
#' (e.g., \code{"project_path/output_model"}).
#'
#' @export
clear_model_results = function(result_path = NULL){
	if(!('result' %in% dir())) stop('result-folder not in path')

	path = paste0(result_path,'/result')

	# Loop through tables:
	for(ii in dir(path)){
		nam = paste0(path,'/',ii)
		x = read.csv2(nam,nrow = 1)
		write.csv2(x[-1,],file = nam,row.names = FALSE)
	}
	print('Result tables cleared.')
}

#' Collecting results back to R from local files
#'
#' @param result_path path to result tables
#' @param executionid which execution-IDs to return?
#' @param tables which tables to retrieve? By default, all table are returned-
#'
#' @export
collect_results = function(result_path,executionid,
	tables = c('execution','accuracy','column_importance',
	'coefficient','metadata','validation')){

	files = dir(result_path)

	if(!all(sapply(tables,function(x) any(grepl(x,files))))){
		stop('All tables not found in result_path.')
	}

	nam = sapply(tables,function(x) grep(x,files,value = T))
  eid = executionid

	results = lapply(nam,function(x){
    	df <- readr::read_csv2(paste0(result_path,'/',nam[2]))
    	return(df[df$executionid %in% eid,])
  	})
  	return(results)
}
