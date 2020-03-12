
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

