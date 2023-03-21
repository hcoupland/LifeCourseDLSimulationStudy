#pylint: disable=invalid-name
"""Main script."""
# Standard libraries
import random

# Third-party libraries
import torch

from lightning.fabric.utilities.seed import seed_everything
from sklearn.metrics import f1_score

# Custom imports
from data import load_yaml_config, load_data, train_test_split
from hyperopt import run_hyperopt
from train import run_final_train


def run(cfg_path):
    """Run the main script.

    Parameters
    ----------
    cfg_path : str or path-like
        Path to a .yml config file
    """
    # Load and setup configuration
    print('Loading config...')
    cfg = load_yaml_config(cfg_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')

    if cfg.seed is None:
        cfg.seed = random.randint(0, cfg.max_seed)

    seed_everything(cfg.seed)

    # Load and split data
    X, y = load_data(cfg)

    X_train, X_test, y_train, y_test = train_test_split(cfg, X, y)

    # Hyperparameter optimisation
    study = run_hyperopt(
        cfg,
        X_train, X_test,
        y_train, y_test,
        device
    )

    # Final train
    learner = run_final_train(
        study,
        X_train, X_test,
        y_train, y_test,
        device
    )

    # Sanity check: get predictions and check the F1 score with skleanr
    # Because I don't like fastai
    preds = []

    batch_size = 64

    for i in range(0, X_test.shape[0], batch_size):
        pred = learner.model(
            torch.tensor(X_test[i:i+batch_size]).float().cuda()
        ).cpu().detach().numpy()

        preds.extend(pred.argmax(-1))

    print(f'F1 score (sklearn): {f1_score(y_test, preds):4f}')


if __name__ == '__main__':
    run('../cfg/config.yml')
