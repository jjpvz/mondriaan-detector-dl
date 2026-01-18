# mondriaan-detector-dl

## Getting Started

Follow these steps to set up and run the project.

1\. Create a virtual environment

```
python -m venv venv
```

2\. Activate the virtual environment (from project root)

Windows (PowerShell):

```
venv\Scripts\Activate.ps1
```

3\. Install dependencies

```
pip install -r requirements.txt
```

4\. Create the configuration file

Create a config.ini file in the project root:

```
# config.ini
[General]
fullset_evd3_path = C:\workspace\evml\EVD3\mondriaan-detector-dl\data\fullset
subset_project_path = C:\workspace\evml\EVD3\mondriaan-detector-dl\data\dataset- EVML project
subset_project_extra_path = C:\workspace\evml\EVD3\mondriaan-detector-dl\data\dataset- EVML project extra
fullset_project_path = C:\workspace\evml\EVD3\mondriaan-detector-dl\data\dataset - EVML project fullset
model_path = C:\workspace\evml\EVD3\mondriaan-detector-dl\models
```

- `fullset_evd3_path` points to the full dataset of images with green background.
- `subset_project_path` points to a small subset of images with white background for testing.
- `subset_project_extra_path` points to an additional subset of images with white background.
- `fullset_project_path` points to the full project dataset with images with a white background.
- `model_path` is where the trained models will be saved.

6\. Run the main script (from project root)

```
python src/main.py
```
