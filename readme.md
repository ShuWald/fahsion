Install Requirements–
`pip install -r requirements.txt`

Running–
Preferred: Run `main.ipynb` notebook. Approx. Total runtime (three simple models) ~ 6min
`python main.py` or `python -m main` produces the same results (but matplotlib will pause running code for its interactive graphs)


# What to Do
Go to `models` directory to find/add your model, and configure how they work there
Go to `main.ipynb` and look for "Model Initialization Logic Here" to change how you want to initialize your the model(s)
Use `train_model()` helper function to train your model
Use `evaluate_model()` helper function to evaluate your model
You can also refer to the later cells as a guide to training your models
If you're not using this helper function then manually plug your model into an `Evaluator` and run `eval()`

## Major Components

Dataset–
Looks like the mnist_reader provided reads data from the .gz files directly, so no need to extract them
`dataevaluator.py` contains the DataEvaluator class, defining how we preprocess the data. Currently it only runs a simple split and plots a graph

Models–
`models` directory contains basic models for kNN, Logistic, and MLP
Models MUST be plugged into Evaluator class to be evaluated properly
`models/logs`directory contains logs of model runs and results when plugged in to Evaluator

Evaluator–
`modelevaluator.py` contains the Evaluator class. Requires only the model and a name to initialize. `eval()` is the best function to run all evaluations. 

Helper Functions–
Stored in `helper.py` to simplify code. Refer to notebooks/code for implementation examples
`train_model` will train your model using model.fit(). Logs into `models/logs` and returns training time
`evaluate_model` will evaluate your model. logs into `models/logs`. Returns an `Evaluator`, useful for evaluating the same model later. Also returns evaluation outputs and time

### Other

Seed-
`seed.py` contains the fixed seed

Backup Main-
`backup_main.ipynb` is a copy main.ipynb for convenience
