# HMM Experiments

This is the Hawkes experiments section.

To run the whole script with 5 folds, run:

```shell script
bash ./main.sh --processes 5
```

To get the experiments results on one fold only, run:

```shell script
python main.py
```

The results are saved on the csv file ``results.csv``. 

To reset the results file, run:

```shell script
python reset.py
```


## Usage

```
usage: experiments/hawkes/main.py [-h] [--explainers] [--device] [--seed] [--deterministic]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["deep_lift", "gradient_shap", "integrated_gradients", "augmented_occlusion", "occlusion", "temporal_integrated_gradients"]
  --device              Which device to use. Default to 'cpu'
  --fold                Fold of the cross-validation. Default to 0
  --seed                Which seed to use to generate the data. Default to 42
  --deterministic       Whether to make training deterministic or not. Default to False
```

```
usage : experiemnts/hawkes/main.sh [--processes] [--device] [--seed]

optional arguments:
  --processes           Number of runners in parallel. Default to 5
  --device              Which device to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
```

```
usage: experiments/hawkes/reset.py [-h]

optional arguments:
  -h, --help            Show this help message and exit.
```
