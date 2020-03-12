# AI-jack open source for R

<img src="https://github.com/Bilot/AI-jack-opensource-R/blob/master/AI-JACK-logo.png"/>

<br>

## *What is AI-jack?*

<p style='text-align: justify;'>
We wanted to do our own AI-projects faster and with fewer errors. Also, coding same things over and over again is quite stupid and boring. We also felt that the maintenance and development of multiple AI/ML-environments needs a coherent solution. We also wanted to create a solution which bends into several different business problems. These factors led us to develop a "framework" that we call <b>AI-jack</b>.
</p>

<p style='text-align: justify;'>
The <b>AI-jack</b> is basically a collection of code, which facilitates robust development of Machine Learning solutions. It integrates data handling, preprocessing, error handling and logging, model training and versioning, model application, and deployment. All of this is handled with just a few lines of code. The modeling is done using the <a href="http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html">H2O API</a>.
</p>

<p style='text-align: justify;'>
This is the R-version of the <b>AI-jack</b> (we also have a Python version, which is likely to be released later). As we have developped this framework using open source code, we have chosen to provide it back to the community. However, this is not the only reason; we also hope the community could help us develop the framework further and make it even better. 
</p>

<img src="https://github.com/bilotAIcore/Bilot-AI-core/raw/master/workflow.png"/>

<br>

## *Features*
<p style='text-align: justify;'>
<b>AI-jack</b> provides capabilities for end-to-end development of machine leartning projects. The functionality is built into <i>modules</i> (collections of functions) that are used to: 
</p>
<ul>
  <li>take care of data connections (e.g., from local files or remote SQL server)</li>
  <li>retrieve data from source and make prepocessing</li>
  <li>train (and optimise) user-specified models</li>
  <li>write execution logs</li>
  <li>version trained models</li>
  <li>deployment, e.g., via predictive API service</li> 
</ul>

<br>

## *How to use it?*

### Installation & setup
<p style='text-align: justify;'>
In order to work, <b>AI-jack</b> needs a working installation
of Java Runtime Environment. To check whether Java exists, type
the following command to the system terminal:
</p>

```bash
java -version
```

<p style='text-align: justify;'>
To install the <b>AI-jack</b> package, all you need is to run the following command in <b>R</b> (making sure that the `devtools` package has been installed): 
</p>

```r
devtools::install_github(repo = "Bilot/AI-jack-opensource-R")
```

Next, one is able to initiate a project as follows:

```r
library(AIjack)

project_path = "/full/path/to/my/project"

init_aijack(project_path)
```

<p style='text-align: justify;'>
here the <code>project_path</code> should contain also the final directory where the project content will be included. There is no need to create this directory manually, as the <code>init_aijack</code> function will do this automatically.
</p>

<p style='text-align: justify;'>
The <code>init_aijack</code> function also automatically creates a directory structure within <code>project_path</code> as well as <code>.csv</code> files for output tables. A project can be deleted with the <code>delete_project()</code> function.
</p>

### Handling
<p style='text-align: justify;'>
The <code>control</code> folder is intended to contain configuration files that are used for parameterising (<code>config</code> files) <b>AI-jack</b> and handling workflow (<code>main</code> files). For example, the <code>config_model.R</code> file is used to make several specifications regarding data handling, model fitting, and file management. However, to make more detailed adjustments, e.g., to model fitting behaviour, one needs to make changes to the source code.
</p>

<p style='text-align: justify;'>
In contrast, there is typically no need to modify the <code>main_model.R</code> and <code>main_apply.R</code> files, as these only execute either model training or model application workflows, respectively.  
</p>

<p style='text-align: justify;'>
The minimum requirement for adjusting the <code>config_model.R</code> 
file for model training is to:
</p>

- set the `project_path` variable as the path to the directory used in `init_aijack()` function   
- in `set$main`, set `label` as the name of the target column in the data  
- in `set$main`, set `model_name_part`to a name appearing in outputs  
- in `set$main`, set `id` as the name of an ID-column in the data (a columns with this name will be created, if missing)  
- in `set$main`, set `test_train_val` as the name of a column indicting to which data split (either 1 = 'train', 2 = 'test', 3 = 'validation') each row belongs to (if missing, a column with this name will be created automatically, containing a data split)  
- in `set$main`, set `labeliscategory` to either `TRUE`/`FALSE` according to the type of the label column (this is checked in the workflow)  
- in `set$model`, give a vector in `train_models` to indicate which models should be trained  

<p style='text-align: justify;'>
When the parameterisation has been done approprietly, the modeling workflow can be automised by scheduling the execution of the <code>main_model.R</code> script. Similarly, scheduling the execution of the <code>main_apply.R</code> script, it is possible to automate batch application of a specified model on new data. 
</p>

<p style='text-align: justify;'>
One also needs to make sure that the control <code>.R</code>-files are located in the <code>control</code>-folder in the project directory and that the working directory is set to the project directory (this can be set automatically in the workflow, given that the correct path is specified in the settings). 
</p>

To execute a workflow from command line, simply run the following command in your project path: 

```r
Rscript main_model.R 
```

### Data
<p style='text-align: justify;'>
The <b>AI-jack</b> is primarily intended to be used for ML-project management in production. This means that while there are some pre-processing steps taking place, there is no functionality for data engineering, which is typically needed <em>before</em> modelling. That is, the intention is that the initial data analysis, investigation and engineering (including feature extraction/engineering) has been done prior to using <b>AI-jack</b>. One clear reason for this is that data engineering is not easily generalised; it depends on the data what manipulations are needed / are most usefull. 
</p>

<p style='text-align: justify;'>
If the <b>AI-jack</b> is run using local files, the <code>source_model</code> directory should contain the source data file in <code>.csv</code> format (by default <code>;</code> separation is assumed). Two columns are also assumed by default: each row needs to have an ID, specified by <code>id</code> column (this can be changed in the settings), and a column <code>test_train_val</code>, which indicates whether a row is assigned to model training, testing, or validation. If these are missing, they will be added automatically (a dummy ID is created and a random data split is added).
</p>

<p style='text-align: justify;'>
Additionally, a text file (typically either <code>.txt</code> or <code>.csv</code>) containing a two-column table of variable names (<code>COLUMN_NAME</code>) and their data types (<code>TYPE_NAME</code>) (the column names can be specified in config). The idea here is to make sure the data types will be formated correctly in R.
</p>

<p style='text-align: justify;'>
If the types-file is written in csv-format, one needs to make sure that the <code>model_name_part</code> parameter string can be fully matched with the file name of the data-file (e.g., <code>model_name_part = "Churn"</code> and file name <code>churn_2020.csv</code>). This is because the relevant files are automatically searched from the "source" directories. The same applies if there are several types-files in the same directory; the correct one is found by matching the <code>model_name_part</code> to the file name. The types-file should have the same column separator as 
the data file (given by the <code>file_sep</code> parameter in config).
</p>

<p style='text-align: justify;'>
Importantly, the data types should follow SQL convention. If data is taken from an SQL database, datatypes are read automatically from the source. Types among "bigint", identity" and "char"
will be casted to <code>character</code>, those among "bit", "varchar", and "nvarchar" will be casted to <code>factor</code>, those among 
"int", "float", "numeric", and "real" will be casted to <code>numeric</code>, and those among "datetime", "date", "time" will be casted to 
<code>character</code>.
</p>

<br>

## *Examples*

<p style='text-align: justify;'>
There are two exampe data sets provided for testing purposes: <code>churn.csv</code> and <code>boston.csv</code>. The churn example is a classical data from the 90's (The Orange Telecom's Churn Dataset), with 5000 rows of customer records. This data is also available e.g. from the <code>C50</code>-package as well as on <a href="https://www.kaggle.com/becksddf/churn-in-telecoms-dataset">Kaggle</a>. 
The Boston house price data, consisting of ca. 500 rows of indicators of median house price, is availbale on <a href="https://www.kaggle.com/vikrishnan/boston-house-prices">Kaggle</a>. Each record in the data describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970.
</p>

<p style='text-align: justify;'>
The churn example is a classical data from the 90's (The Orange Telecom's Churn Dataset), with 5000 rows of customer records. This data is also available e.g. from the <code>C50</code>-package as well as on <a href="https://www.kaggle.com/becksddf/churn-in-telecoms-dataset">Kaggle</a>. Here the aim is to predict whether a customer is at risk to churn, based on the recorded history given by the <code>churn</code> column (classification problem).
</p>

<p style='text-align: justify;'>
The Boston house price data, consisting of ca. 500 rows of indicators of median house price, is availbale on <a href="https://www.kaggle.com/vikrishnan/boston-house-prices">Kaggle</a>, as well as in the <code>mlbench</code> package. Each record in the data describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. In this case the objective is to predict the level of house prises within different areas in Boston (regression problem).
</p>

<p style='text-align: justify;'>
Foe each of the datasets, there is also a data types-file available, as well as an unlabelled samples for testing model application.
</p>

## *Web service*

<p style='text-align: justify;'>
Given that there are trained models to apply, one can easily expose such a model as an API, using <code>plumber</code>. This requires:
</p>

- a script file `plumber_core.R` that defines the API logic  
- configuration file `config_plumber.R`  
- a parameter string for calling the API  

<p style='text-align: justify;'>
The parameter string consists of three parts:
</p>

- Feature values: `param <- "param=val1#val2#val3#val4"`  
- Feature names: `param2 <- "param2=nam1#nam2#nam3#nam4"` 
- Feature data types: `"param3=f#n#n#f"` (f = factor, n = numeric, etc.)  

<p style='text-align: justify;'>
If the data to query exists in a file, the parameter string can be generasted using the <code>parse_params()</code> function:
</p>

```r
parse_params(file_path = "path_to_file",
             row = 1,set = set)
```

<p style='text-align: justify;'>
The API can be exposed with the following commands:
</p>

```r
# Create Plumber router:
r <- plumber::plumb('control/plumber_core.R')

# Expose endpoint:
r$run(host='0.0.0.0', port=8000, swagger=TRUE)
```

<p style='text-align: justify;'>
which will open the API in <code>localhost:8000</code>.
</p>

<p style='text-align: justify;'>
When the endpoint is set up and running, it can be queried from the command line as follows:
</p>

```bash
curl --data "param=val1#val2#val3#val4&param2=nam1#nam2#nam3#nam4&param3=f#n#n#f" "http://localhost:8000/predict"
```

<p style='text-align: justify;'>
The result will be written either to a results table ("output_plumber/predictions.csv") or to SQL database table, depending on the settings.
</p>