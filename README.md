<p align="center">
<img src="https://github.com/Bilot/AI-JACK-opensource-R/blob/master/logo/logo.png" width="700"/>
</p>

<br>

# AI-JACK open source for R

## *What is AI-JACK?*

<p style='text-align: justify;'>
We wanted to do our own AI-projects faster and with fewer errors. Also, coding same things over and over again is quite stupid and boring. We also felt that the maintenance and development of multiple AI/ML-environments needs a coherent solution. We also wanted to create a solution which bends into several different business problems. These factors led us to develop a "framework" that we call <b>AI-JACK</b>.
</p>

<p style='text-align: justify;'>
The <b>AI-JACK</b> is basically a collection of code, which facilitates robust development of Machine Learning solutions. It integrates data handling, preprocessing, error handling and logging, model training and versioning, model application, and deployment. All of this is handled with just a few lines of code. The modeling is done using the <a href="http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html">H2O API</a>.
</p>

<p style='text-align: justify;'>
This is the R-version of the <b>AI-JACK</b> (we also have a Python version, which is likely to be released later). As we have developped this framework using open source code, we have chosen to provide it back to the community. However, this is not the only reason; we also hope the community could help us develop the framework further and make it even better. 
</p>

<img src="https://github.com/bilotAIcore/Bilot-AI-core/raw/master/workflow.png"/>

<br>

## *Features*
<p style='text-align: justify;'>
<b>AI-JACK</b> provides capabilities for end-to-end development of machine leartning projects. The functionality is built into <i>modules</i> (collections of functions) that are used to: 
</p>
<ul>
  <li>take care of data connections (e.g., from local files or remote SQL server),</li>
  <li>retrieve data from source and make prepocessing,</li>
  <li>train (and optimise) user-specified models,</li>
  <li>write execution logs,</li>
  <li>version trained models,</li>
  <li>deployment, e.g., via predictive API service.</li> 
</ul>

<br>

## *How to use it?*
We organize __webinars__! There are two upcoming events where we show briefly how to use AI-JACK and our past experiences in implementing the tool in production. If you want to see real use cases - you're welcome to join us!

- [18.06.2020 - in Polish](https://bilot.group/pl/event/webinar-ai-jack-silnik-ktory-napedzi-uczenie-maszynowe-w-twoim-biznesie-2/)
- [24.06.2020 - in English](https://bilot.group/event/webinar-ai-jack-the-engine-that-boosts-machine-learning-in-your-business/)

### Installation & setup
<p style='text-align: justify;'>
In order to work, <b>AI-JACK</b> needs a working installation
of Java Runtime Environment. To check whether Java exists, type
the following command to the system terminal:
</p>

```bash
java -version
```

If there is no Java installation on your machine, this command should prompt installation. If you have an old version, you probably neet to update it. Java installations can be found <a href="https://www.oracle.com/java/technologies/javase-downloads.html">here</a>.

<p style='text-align: justify;'>
To install the <b>AI-JACK</b> package, all you need is to run the following command in <b>R</b> (making sure that the `devtools` package has been installed): 
</p>

```r
devtools::install_github(repo = "Bilot/AI-JACK-opensource-R")
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
The <code>control</code> folder is intended to contain configuration files that are used for parameterising (<code>config</code> files) <b>AI-JACK</b> and handling workflow (<code>main</code> files). For example, the <code>config_model.R</code> file is used to make several specifications regarding data handling, model fitting, and file management. However, to make more detailed adjustments, e.g., to model fitting behaviour, one needs to make changes to the source code.
</p>

<p style='text-align: justify;'>
In contrast, there is typically no need to modify the <code>main_model.R</code> and <code>main_apply.R</code> files, as these only execute either model training or model application workflows, respectively.  
</p>

<p style='text-align: justify;'>
The minimum requirement for adjusting the <code>config_model.R</code> 
file for model training is to:
</p>

- set the `project_path` variable as the path to the directory used in `init_aijack()` function,   
- in `set$main`, set `label` as the name of the target column in the data,  
- in `set$main`, set `model_name_part`to a name appearing in outputs,  
- in `set$main`, set `id` as the name of an ID-column in the data (a columns with this name will be created, if missing),  
- in `set$main`, set `test_train_val` as the name of a column indicting to which data split (either 1 = 'train', 2 = 'test', 3 = 'validation') each row belongs to (if missing, a column with this name will be created automatically, containing a data split),  
- in `set$main`, set `labeliscategory` to either `TRUE`/`FALSE` according to the type of the label column (this is checked in the workflow),  
- in `set$model`, give a vector in `train_models` to indicate which models should be trained.  

<p style='text-align: justify;'>
When the parameterisation has been done approprietly, the modeling workflow can be automised by scheduling the execution of the <code>main_model.R</code> script. Similarly, scheduling the execution of the <code>main_apply.R</code> script, it is possible to automate batch application of a specified model on new data. 
</p>

<p style='text-align: justify;'>
One also needs to make sure that the control <code>.R</code>-files are located in the <code>control</code>-folder in the project directory and that the working directory is set to the project directory (this can be set automatically in the workflow, given that the correct path is specified in the settings). 
</p>

#### Handling and running when using clustering algorithm
<p style='text-align: justify;'>
As clustering algorithms are treated slightly differently than supervised ML techniques, there are separate <code>config</code> files designed to work with these methods. There is no need to configure standard <code>config_model.R</code> and <code>main_model.R</code> files - you have to use <code>config_clust_model.R</code> and <code>main_clust_model.R</code> files instead.
</p>

<p style='text-align: justify;'>
The adjustment of <code>config_clust_model.R</code> 
file for model training goes almost the same as <code>config_model.R</code>:
</p>

- set the `project_path` variable as the path to the directory used in `init_aijack()` function,   
- in `set$main`, set `model_name_part`to a name appearing in outputs,  
- in `set$main`, set `id` as the name of an ID-column in the data (a columns with this name will be created, if missing).  

### Running

After the necessary configurations have been made, a workflow can be executed from command line as follows. First, make sure that you're located in the project directory (`cd /path/to/project`). Then simply run the following command in your project path: 

**Mac/Linux**
```r
Rscript control/main_model.R 
```

Again, given that the `config_model.R` has been modified correctly, this should run the model training workflow.

**Windows**
If you're running Windows, you may need to tell where the `Rscript` program is located:

**From R**
To run the workflow from withion `R`, first set the working directory to the project path:

```r
setwd('/path/to/project')
```

Then, just source the workflow script:

```r
source('control/main_model.R')
```

### Data
<p style='text-align: justify;'>
The <b>AI-JACK</b> is primarily intended to be used for ML-project management in production. This means that while there are some pre-processing steps taking place, there is no functionality for data engineering, which is typically needed <em>before</em> modelling. That is, the intention is that the initial data analysis, investigation and engineering (including feature extraction/engineering) has been done prior to using <b>AI-JACK</b>. One clear reason for this is that data engineering is not easily generalised; it depends on the data what manipulations are needed / are most useful. 
</p>

<p style='text-align: justify;'>
If the <b>AI-JACK</b> is run using local files, the <code>source_model</code> directory should contain the source data file in <code>.csv</code> format (by default <code>;</code> separation is assumed). Two columns are also assumed by default: each row needs to have an ID, specified by <code>id</code> column (this can be changed in the settings), and a column <code>test_train_val</code>, which indicates whether a row is assigned to model training, testing, or validation. If these are missing, they will be added automatically (a dummy ID is created and a random data split is added).
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
For each of the datasets, there is also a data types-file available, as well as an unlabelled samples for testing model application.
</p>

<br>

## *Modelling*

The modelling functionality of **AI-JACK** rests upon package [`h2o`](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html), enabling running H2O from within R. H2O is an open source, in-memory, distributed, fast, and scalable machine learning and predictive analytics platform that allows you to build machine learning models on big data and provides easy productionalization of those models in an enterprise environment.

At present, <b>AI-JACK</b> has capabilities for training either classification or regression models. For classification, the logic has been built binary problems in mind, but this can be fairly easily modified. 

<br>

## *Web service*

<p style='text-align: justify;'>
Given that there are trained models to apply, one can easily expose such a model as an API, using <code>plumber</code>. This requires:
</p>

- a script file `plumber_core.R` that defines the API logic,  
- configuration file `config_plumber.R`,  
- a parameter string for calling the API.  

<p style='text-align: justify;'>
The parameter string consists of three parts:
</p>

- feature values: `param <- "param=val1#val2#val3#val4"`,  
- feature names: `param2 <- "param2=nam1#nam2#nam3#nam4"`, 
- feature data types: `"param3=f#n#n#f"` (f = factor, n = numeric, etc.).  

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

<br>

## *Technical details*

### Data
<p style='text-align: justify;'>
The <code>data_read()</code> function handles the retrieval of raw data from the specified source. When writing output, either <code>write_db()</code> or <code>write_csv()</code> will be used, depending on the data connection. 
</p>

### Statistics
<p style='text-align: justify;'>
In the workflow, the <code>prep_results()</code> function (among other operations) generates a standard statistical summary of the data, which will be outputted to a <code>metadata</code> table. In turn, the <code>calculate_stats()</code> function calculates other statistics on the data (only correlation implememted).
</p>

### Transformations
<p style='text-align: justify;'>
The following transformation routines are available:
</p>

- Classify numeric features with missing values (`trans_classifyNa`)  
- Drop constant features (`trans_delconstant`)  
- Drop equal (redundant) features (`trans_delequal`)  
- Replace special characters in nominal features and feature names (`trans_replaceScandAndSpecial`)  
- Discretise continuous features, based on entropy (`trans_entropy`)

<p style='text-align: justify;'>
The transformation step is handled by the <code>do_transforms()</code> function, except for <code>trans_entropy</code>, which is call by the <code>entropy_recategorization()</code> function. Recategorised data will be constructed and used in models, if the parameter <code>set$model$discretize</code> is set <code>TRUE</code>. Also, function <code>create_split()</code> will be called in the workflow, if the raw data does not contain a column specifying data split. 
</p>

### Model algorithms
<p style='text-align: justify;'>
Currently, the following supervised modelling methods are available:
</p>

- linear models (`glm`) with `h2o.glm`,  
- decision tree (`decisionTree`) with `h2o.randomForest` (`n_trees = 1`),  
- random forest (`randomForest`) with `h2o.randomForest`,  
- gradient boosting (`gbm`) with `h2o.gbm`,  
- extreme gradient boosting (`xgboost`) with `h2o.xgboost` (not available on Windows),  
- deep learning (`deeplearning`) with `h2o.deeplearning`,  
- autoML (`automl`) with `h2o.automl`.  

<p style='text-align: justify;'>
In addition, deep learning is also possible to run in unsupervised form, by using it in <code>autoencoder</code> form.
Also, three clustering methods are currently available:
</p>

- k-means with `kmeans` function from `stats` package,
- expectation-maximization (EM) with `Mclust` package,
- k-medoids (PAM) with `pam` function from `cluster` package.

<p style='text-align: justify;'>
In clustering case, the user doesn't have to choose a technique as all three are done in parallel and compared in terms of average silhouette width. There are also functions available to visualize the clustering results.
</p>

<p style='text-align: justify;'>
The <code>create_models()</code> function handles hyperparameter optimisation (training with <code>train</code>-split and validating with <code>test</code>-split) as well as re-fitting the best model (on both the <code>train</code>- and <code>test</code>-split, except for deep learning).
</p>
