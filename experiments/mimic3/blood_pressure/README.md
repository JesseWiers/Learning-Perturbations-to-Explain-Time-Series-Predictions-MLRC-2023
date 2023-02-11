# Mimic III Blood Pressure Experiments

This is the Mimic III blood pressure experiments section.

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
usage: experiments/mimci3/blood_pressure/main.py [-h] [--explainers] [--device] [--seed] [--deterministic]

optional arguments:
  -h, --help            Show this help message and exit.
  --explainers          List of the explainers to use. Default to ["deep_lift", "gradient_shap", "integrated_gradients", "augmented_occlusion", "occlusion", "temporal_integrated_gradients"]
  --areas               List of areas to use. Default to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  --device              Which device to use. Default to 'cpu'
  --fold                Fold of the cross-validation. Default to 0
  --seed                Which seed to use to generate the data. Default to 42
  --deterministic       Whether to make training deterministic or not. Default to False
```

```
usage : experiemnts/mimci3/blood_pressure/main.sh [--processes] [--device] [--seed]

optional arguments:
  --processes           Number of runners in parallel. Default to 5
  --device              Which device to use. Default to 'cpu'
  --seed                Which seed to use to generate the data. Default to 42
```

```
usage: experiments/mimci3/blood_pressure/reset.py [-h] [-e]

optional arguments:
  -h, --help            Show this help message and exit.
```
