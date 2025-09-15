# Linear Regression/Classification
Training a linear classifier on the MNIST dataset, by fitting a line using least squares. Since regression works best when there are two classes only, we will apply regression to each pair of classes. MNIST has 10 classes (10 digits), so the total number of regressors you train will be (10 choose 2) = 45.

## How to run

These steps assume Windows with PowerShell and Python set up. The dataset files are expected under `./datasets/MNIST/raw` as in this repo.

1) Create/activate a Python environment (optional but recommended)

Pick one of the two options below.

- Using Conda (recommended if you already use Anaconda/Miniconda):

```powershell
conda create -n mnist-linreg python=3.10 -y
conda activate mnist-linreg
pip install -r requirements.txt
```

- Using venv (built-in Python virtual environment):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Verify the dataset path

`linear_classification.py` expects the MNIST files at `./datasets/MNIST/raw`:

```
datasets/
	MNIST/
		raw/
			train-images-idx3-ubyte
			train-labels-idx1-ubyte
			t10k-images-idx3-ubyte
			t10k-labels-idx1-ubyte
```

If your files are elsewhere, edit the path in `linear_classification.py`:

```python
mndata = MNIST('./datasets/MNIST/raw')
```

3) Run the script

From the project root:

```powershell
python .\linear_classification.py
```

You should see console output showing the number of samples, labels, and then training/testing progress for each digit pair. At the end of each pair, the script prints a test accuracy.

## Troubleshooting

- ModuleNotFoundError: No module named 'mnist'
	- Ensure requirements are installed: `pip install -r requirements.txt`

- FileNotFoundError: MNIST files not found
	- Check the dataset is at `./datasets/MNIST/raw` or update the path in the script.

- Singular matrix or numerical warnings
	- The code computes a pseudo-inverse via eigenvalue thresholding. If needed, adjust the tolerance `tol` inside `train()` (e.g., `1e-12`).


## Notes

- This repository focuses on linear regression for classification of two classes on MNIST data.
