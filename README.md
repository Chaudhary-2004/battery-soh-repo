# Battery SoH Project — Reproducible Environment & Notebooks

## 1. Create the Conda environment
```bash
conda env create -f environment.yml
conda activate battery_soh_env
```

## 2. (Optional) Generate `requirements.txt` as a pip fallback
Run this **inside** the activated environment:
```bash
pip freeze > requirements.txt
```

## 3. Register the Jupyter kernel
Inside the environment:
```bash
pip install ipykernel
python -m ipykernel install --user --name=battery_soh_env --display-name "battery_soh_env"
```

Then in JupyterLab/Notebook, select the **battery_soh_env** kernel for the notebooks in the `notebooks/` folder.

## 4. Repo hygiene
This repo includes a `.gitignore` that excludes large or generated artifacts:
- `data/raw/`, `data/interim/`, `data/processed/`
- `artifacts/`, `models/`
- `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`

Initialize and make the first commit (don’t add raw data):
```bash
git init
git add .
git commit -m "Initial project skeleton"
# create a remote on GitHub, then:
# git branch -M main
# git remote add origin <your-remote-url>
# git push -u origin main
```

## 5. Reproducibility & Random Seeds
In your training scripts, set seeds:
```python
import os, random, numpy as np
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
```

## 6. Data layout
Place your processed CSV at:
```
data/processed/all_cells_cycle_features.csv
```
The sample notebooks expect that path.

## 7. Notebooks
- `notebooks/00_env_test.ipynb`: prints package versions, tries to read 10 rows from the processed CSV, and makes a tiny sanity plot.
- `notebooks/01_EDA.ipynb`: basic exploratory data analysis scaffold.
- `notebooks/02_model_baseline.ipynb`: quick Random Forest regression baseline that saves `models/rf_baseline.pkl`.

> If `data/processed/all_cells_cycle_features.csv` is not present, the test/EDA notebooks will error on read; add the file first.

## 8. Quick smoke test
1) Open `notebooks/00_env_test.ipynb`, run all cells — you should see version prints and a simple plot.  
2) Open `notebooks/02_model_baseline.ipynb` and run all cells — you should see MAE/MSE printed and a model saved to `models/rf_baseline.pkl`.

---

**Tip:** keep `environment.yml` and (optionally) `requirements.txt` committed so others can reproduce your setup exactly.
