# CIL Project: Monocular Depth Estimation
**Group Name:** Easy Baseline\
**Deadline:** 2025-05-30\
**Presentation Date:** 2025-05-28\
**Report:** [N/A]

## Project Structure Proposal
In the interest of keeping a unified structure we can start out with the following proposal, any changes should be communicated in advance:

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
│   ├── create_prediction_csv.py
│   ├── monocular-depth-example-notebook.ipynb
│   └── lib
├── .gitignore
├── README.md
└── requirements.txt
```

Where `lib` is a stand-in for more specific folders that hold different parts of the model as well as algorithms and other helper functions that are needed (subject to discussion).

When downloading the testing and training data from the Kaggle competitions [data page](https://www.kaggle.com/competitions/ethz-cil-monocular-depth-estimation-2025/data), unzip all files except for `create_prediction_csv.py` into the `data` folder such that it looks like the structure detailed above. Then move `create_prediction_csv.py` into `src`.

## Setting up a Development Environment
When using VSCode, creating a new dev environment is as easy as pressing `Ctrl+P`, searching for
```
>Python: Select Interpreter
```
and choosing the option
```
Create Virtual Environment...
```
Choosing `Venv` is most viable option since PyTorch doesn't support conda installs anymore. Create a new virtual environment using Python version `3.10.11` to ensure stability (subject to discussion).

### Installing depencencies
Run the following command
```
pip install -r requirements.txt
```
Please don't forget to add new depencencies to `requirements.txt`

### Installing PyTorch with CUDA
If team members are in possesion of a graphics card with performance rivaling or even exceeding a GTX 1080 Ti, then it would be for the better to train models locally as opposed to the [student cluster](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster).\
The following command will install PyTorch with CUDA functionalities available:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```