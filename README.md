# Customer Segmentation for Targeted Marketing

Group customers with similar attributes to support exploratory marketing analysis.

## Why this project

This repository is part of my practical machine-learning portfolio. It focuses on a complete, understandable workflow rather than claiming production readiness.

## Dataset

The repository includes `Mall_Customers.xls`. Preserve the dataset license and document the source when publishing results.

## Approach

Data cleaning, feature preparation, scaling, and K-Means clustering with 2D and 3D visualizations.

### Features

Customer demographic and purchasing-behavior fields available in the Mall Customers dataset.

## Evaluation and current result

Cluster visualizations are included. The next documentation improvement is to report the selected k and a silhouette score, then summarize each segment with its defining characteristics.

## Run locally

```bash
git clone https://github.com/MeehdiF/Customer-Segmentation-for-Targeted-Marketing.git
cd Customer-Segmentation-for-Targeted-Marketing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python untitled.py
```

For notebook exploration, open the `.ipynb` file with Jupyter after installing the same dependencies.

## Limitations and next steps

Clusters are exploratory rather than validated customer personas. Future work should compare k values with silhouette or stability analysis and explain how each segment would change a marketing action.

## Repository structure

- `README.md` — project context and reproducibility notes
- `requirements.txt` — Python dependencies used by the scripts
- `.ipynb` / `.py` files — analysis and model experiments

## License

See [`LICENSE`](LICENSE). Check the dataset's own terms separately; repository code licensing does not automatically license bundled data.
