# Assignment-5-MLOps

## Dataset (Online via DVC)

This project uses the public `penguins.csv` dataset from:

https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv

Track it with DVC:

```bash
mkdir -p data
dvc import-url --no-download https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv data/penguins.csv
git add data/.gitignore data/penguins.csv.dvc
```

Fetch locally before training:

```bash
dvc pull
python train.py
```