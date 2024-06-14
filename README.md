# NC_regression
This is the codebase for Neural Regression Collapse on MuJoCo datasets. The code is focusing on the non-UFM case now, so modifications are needed to run UFM experiments.

## Dataset
Please find the MuJoCo datasets via [this link](https://drive.google.com/file/d/1XScUTYrXsfMCgEQQtURCmUSS0IGxatZX/view?usp=drive_link). The reacher and swimmer datasets come from [this project](https://huggingface.co/datasets/jat-project/jat-dataset), and the hopper dataset is from D4RL. Each dataset is a dictionary with 3 keys: `observations`, `actions`, and `rewards`. The corresponding values are numpy arrays. The data ratio between the training and test split is $9:1$. After downloading the datasets, please put them under `/NC_regression/dataset/mujoco`.

## Run Experiments
For your reference, all codes for experiments with L2 regularization are contained in `/NC_regression/main/BC.py`. To run the codes, one might as well take a look at the launcher files under `/NC_regression/main/exp/case1`, where `.py` files set up experimental hyperparameters, and `.sh` files submit jobs onto HPC. This is an exemplary workflow. One may use the code at their discretion.
