# PAC-Private Responses with Adversarial Composition

This repository contains code and results for the paper "PAC-Private Responses with Adversarial Composition".

## Repository Structure
- `generate_random_splits.py`: Code to generate random subsets of the universe to be used as the secret datasets.
- `train_one_model_tabular.py`: Code to train a single model on a given dataset split of given tabular dataset.
- `train_one_model_vision.py`: Code to train a single model on a given dataset split of given vision dataset.
- `train_one_model_nlp.py`: Code to train a single model on a given dataset split of given NLP dataset.
- `utils.py`: Utility functions for loading datasets, training models, and evaluating models.
- `pac_privacy_utils.py`: Functions for calibrating and sampling noise for PAC privacy, as well as other PAC privacy related utilities.
- `private_response.py`: The main code for generating PAC-private responses.
- `run_priv_responses.py`: Code to run PAC-private responses to reproduce the experiments in the paper.
- `run_mia.py`: Code to run membership inference attacks to reproduce the experiments in the paper.
- `run_distillation.py`: Code to run model distillation experiments to reproduce the experiments in the paper.
- `final_results/`: Directory containing the final results JSON files for the experiments.
- `plot.ipynb`: Jupyter notebook for generating plots and tables from the results.
- `scripts/`: Directory containing bash scripts to run the python scripts above for all datasets and settings.
- `fig/`: Directory to save generated figures and tables.
- `saved_models/`: Directory to save trained models (not included in the repository because of size, we will provide a link to download them).

## Environment
- We run our experiments with Python 3.12.3
- Required packages can be installed via `pip install -r requirements.txt`

## Reproducing Results
- To reproduce the experiments, first generate random splits:
    ```
    bash scripts/generate_splits.sh
    ```
  This will run the `generate_random_splits.py` script for all datasets and save the splits in `saved_models`
- Next, train models on the generated splits:
    ```
    bash scripts/train_models.sh
    ```
  This will run the `train_one_model_*.py` scripts for all datasets and save the trained models in `saved_models`. Note that this script is purely sequential and you should consider parallelizing it according to your available compute resources.
- Next, run PAC-private responses:
    ```
    bash scripts/run_priv_responses.sh
    ```
  This will run the `run_priv_responses.py` script for all datasets and settings, and save the results in `final_results/`
- Next, run membership inference attacks:
    ```
    bash scripts/run_mia.sh
    ```
  This will run the `run_mia.py` script for all datasets and settings, and save the results in `final_results/`
- Next, run model distillation experiments:
    ```
    bash scripts/run_distillation.sh
    ```
  This will run the `run_distillation.py` script for all datasets and settings, and save the results in `final_results/`
- Finally, generate plots and tables by running the `plot.ipynb` notebook.

## Download our Results and Model Checkpoints
Training models and generating all results can take a significant amount of time and computatinoal resources. To download our models and results directly, you can run the following command:
```bash
git clone https://huggingface.co/p9wtkg8/2gq3nt5 saved_models
```
Note that these results will override any results you have in the `saved_models/` directory, so it's a good idea to first rename your existing `saved_models/` directory if you have one.
