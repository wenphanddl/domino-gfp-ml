thi new line deleted

done s is changed from workspace
latest update
new line First  s line new done line adjusted
no ok line
number line
 ignore this line added for conflicts dont ignore
This is a simple project that trains a logistic regression model.
some line added

# Requirements
* python3 
* pandas
* sklearn

# Run
To run the entire pipeline:

1. Update the paths: `vi set-local-env.sh`
2. Go into the `code` folder: `cd code`
3. Run the pipeline: `bash run-pipeline.sh`

# Code Files
* `partition-data.py`: Partitions data. Take the `adult.csv` dataset and creates a training and test set of features (X) and target (y).
* `train-column-transformer.py`: Trains a column transformer that standardizes `age` and `hours-per-week` and one-hot encodes `workclass`, `education`, `gender`, and `occupation`.  Saves transformer as an artifact (`coltransformer.pkl`).
* `train-model-logreg.py`: Trains a logistic regression model.  Saves model as an artifact (`model_logreg.pkl`)

