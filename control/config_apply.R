
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
    