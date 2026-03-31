# ML Service v2

This directory implements version 2 of the livestock disease prediction service.

## Setup

1. **Create a virtual environment** (optional but recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

## Training the model

The API relies on a trained decision tree model stored in `decision_tree_model.pkl`.
The training script will read the dataset and write the artifact to the same
folder.

```powershell
python train_decision_tree.py
```

Running the script again will back up the previous model as
`decision_tree_model.pkl.bak` before writing the new file.

If the model file is missing when the service starts, you will see an error
like:

```
FileNotFoundError: Model artifact not found at C:\...\ml_service_v2\decision_tree_model.pkl. Have you run `train_decision_tree.py`?
```

So make sure to train the model before launching the server.

## Starting the server

From the `ml_service_v2` directory run:

```powershell
uvicorn ml_service:app --host 0.0.0.0 --port 8001 --reload
```

The `--reload` flag watches the current directory so changes to the code will
trigger a reload.

## Tests

There are a few unit tests under this directory; run them after training to
ensure the model loads correctly:

```powershell
python -m pytest
```