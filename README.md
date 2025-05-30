# UDR: Uncertainty-Aware Depth Refinement
UDR is a framework that can be used on top of an existing MDE model. It works by computing the variance of each pixel in a prediction with the help of a deep ensemble and refining it with each iteration.

## Setting up a Development Environment
When using VSCode, creating a new dev environment is as easy as pressing `Ctrl+P`, searching for
```
>Python: Select Interpreter
```
and choosing the option
```
Create Virtual Environment...
```
Choosing `Venv` is most viable option since PyTorch doesn't support conda installs anymore. Create a new virtual environment using Python version `3.10.12`.

> **_HINT:_** If you don't want to use VSCode, go ahead and create a venv the normal way using Python 3.10.12

## Installing depencencies
> **_WARNING:_** Please make sure to use a Linux or Mac environment since Natten has no pre-built binaries for Windows. If you still want to use Windows, be ready to debug the local compilation process.

First download PyTorch 2.0.1 with CUDA 11.8:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Then go ahead and run the following command:
```bash
pip install -r requirements.txt
```

## Running Experiments
The majority of experiments can be run in this repository, if you want to re-run the evaluation for [MiDaS](https://www.kaggle.com/code/kailik34/inference-midas/output), [Marigold](https://www.kaggle.com/code/kailik34/inference-marigold) and [DepthAnythingV2](https://www.kaggle.com/code/kailik34/inference-depthanythingv2), run and download the outputs of the kaggle notebooks to a directory of choice and run the following command from `src`:
```bash
cd src
```
```bash
python evalute.py --predictions_dir <path_to_outputs> --gt_dir <path_to_ground_truth> --train_list <path_to_train_list.txt>
```

For experiments using UDR, download the pretrained weights from [here](https://drive.google.com/file/d/1LV_-7g9tV5sRa5z8UnYCKKblXxQlX_rd/view?usp=sharing) and unzip the folder to `cil-mde/src/checkpoints`.

> **_NOTE:_** While we did show the original SDR metrics in our report, to re-run that evaluation you have to replace the MSPN implementation

For the remaining experiments we need the following folder structure for the training and test data:

```
cil-mde
├── data
│   ├── output
│   │   ├── predictions
│   │   └── results
│   ├── test/test
│   ├── train/train
│   ├── test_list.txt
│   └── train_list.txt
├── src
│   ├── checkpoints/
│   ├── Depth_Anything_V2/
│   ├── inference_notebooks/
│   ├── MSPN_SDR/
│   ├── create_prediction_csv.py
│   ├── evaluate.py
│   ├── experiments.py
│   ├── udr_training.ipynb
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Experiment 1: Evaluation of Mean Ensemble and UDR
The mean ensemble has the means to output a variance map that can be fed into UDR. The experiment calculates metrics for both and shows the difference that UDR can make (spoiler alert: it's not that much with a model trained on ~512 images out of a potential 20k). As an added bonus, the original SDR architecture is also evaluated as an added comparison
```bash
python experiments.py --data_dir <path_to_data> --experiment_number 1
```
> **_WARNING:_** You need a lot of free RAM to run this on a PC with 16GB (8GB won't do). Close all other applications during this experiment and if you are using WSL, increase the memory size in the WSL settings to 12000 MB temporarily.

### Experiment 2: AUSE & AURG of output Variance
This experiment calculates the "Area Under the Sparsification Error" and "Area Under the Random Gain" metrics for each iteration of UDR and prints out the mean values.
```bash
python experiments.py --data_dir <path_to_data> --experiment_number 2
```

### Experiment 3: Analysis of the Kalman Gain
The Kalman gain is an integral part of UDR, which depends on a correct modeling of the uncertainty propagation. This experiment determines if our uncertainty propation actually does its job. It outputs the minimum, mean and maximum values of the Kalman gain computed each iteration and show the first 6 sample images.
```bash
python experiments.py --data_dir <path_to_data> --experiment_number 3
```

### Experiment 4: Test Predictions and Submission
Our submission was generated with this experiment. It runs the whole UDR pipeline on the whole test set and saves it to `cil-mde/src/data/output/predictions/`. Run the `create_prediction_csv.py` to generate the submission csv.
```bash
python experiments.py --data_dir <path_to_data> --experiment_number 4
python create_prediction_csv.py
```

The submission csv can then be found at `cil-mde/src/data/predictions.csv`