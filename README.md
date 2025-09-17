# Linear Regression/Classification

Training a linear classifier on the MNIST dataset, by fitting a line using least squares. Since regression works best when there are two classes only, we will apply regression to each pair of classes. MNIST has 10 classes (10 digits), so the total number of regressors you train will be (10 choose 2) = 45.

## How to run

These steps assume Windows with PowerShell and Python set up. The dataset files are expected under `./datasets/MNIST/raw`.

1. Create/activate a Python environment (optional but recommended)

   Pick one of the two options below.

    - Using Conda (recommended if you already use Anaconda/Miniconda):

   ```powershell
   conda create -n mnist-linreg python=3.13 -y
   conda activate mnist-linreg
   pip install -r requirements.txt
   ```

    - Using venv (built-in Python virtual environment):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   ### Prerequisites

    - Python 3.13 (tested with 3.13.7)
    - Packages listed in `requirements.txt`

2. Verify the dataset path

   `linear_classification.py` expects the MNIST files in `./datasets/MNIST/raw`:

   ```text
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

3. Run the script

   From the project root:

   ```powershell
   python linear_classification.py
   ```

   You should see console output showing the number of samples, labels, and then training/testing progress for each class pair. At the end of each pair, the script prints a test accuracy.

### Plotting (optional)

There’s a helper `plot_image(...)` and `plot_avg_digit` you can call to visualize a raw MNIST and the reshaped image.

## Troubleshooting

- ModuleNotFoundError: No module named 'mnist'
  - Ensure requirements are installed: `pip install -r requirements.txt`

- FileNotFoundError: MNIST files not found
  - Check the dataset is at `./datasets/MNIST/raw` or update the path in the script.

- Singular matrix
  - The matrix is not invertible, the rows or columns are linearly dependent becuase of duplicate rows or columns.
  - Crop boarders to remove all zero row and columns.

- Plots not showing
  - If using a non-GUI environment, use `plt.savefig('figure.png', bbox_inches='tight')` to show image.

## Notes

- This repository focuses on linear regression for classification of two classes on MNIST data.
