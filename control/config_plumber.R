
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
    