"""Fit preprocessing within each CV fold and preserve original target units."""
import copy
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer


class FoldModel:
    def __init__(self, estimator):
        self.estimator = copy.deepcopy(estimator)
        if hasattr(self.estimator, 'get_params') and self.estimator.get_params().get('random_state', 'absent') is None:
            self.estimator.set_params(random_state=0)

    def fit(self, x, y, eval_set=None, **kwargs):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float).reshape(-1, 1)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Training values must be finite')
        self.x_transform = StandardScaler().fit(x)
        self.y_transform = (FunctionTransformer().fit(y) if np.ptp(y) == 0 else
                            PowerTransformer(method='yeo-johnson', standardize=False).fit(y))
        fit_args = dict(kwargs)
        if eval_set is not None:
            fit_args['eval_set'] = [(self.x_transform.transform(np.asarray(a, dtype=float)),
                                     self.y_transform.transform(np.asarray(b, dtype=float).reshape(-1, 1)).ravel())
                                    for a, b in eval_set]
        # Verbosity is cosmetic; modern LightGBM/XGBoost moved it out of fit().
        parameters = inspect.signature(self.estimator.fit).parameters
        if 'verbose' not in parameters and not any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            fit_args.pop('verbose', None)
        if 'eval_metric' in fit_args and 'eval_metric' not in parameters:
            self.estimator.set_params(eval_metric=fit_args.pop('eval_metric'))
        self.estimator.fit(self.x_transform.transform(x), self.y_transform.transform(y).ravel(), **fit_args)
        return self

    def predict(self, x):
        prediction = self.estimator.predict(self.x_transform.transform(np.asarray(x, dtype=float)))
        return self.y_transform.inverse_transform(np.asarray(prediction).reshape(-1, 1)).ravel()


def original_unit_mape(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float).ravel(), np.asarray(prediction, dtype=float).ravel()
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Evaluation requires matching finite arrays')
    observed = truth != 0
    if not np.any(observed):
        raise ValueError('MAPE is undefined for all-zero targets')
    return float(np.mean(np.abs((truth[observed] - prediction[observed]) / truth[observed])))


def fit_supported(estimator, *args, **kwargs):
    parameters = inspect.signature(estimator.fit).parameters
    if 'verbose' not in parameters and not any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        kwargs.pop('verbose', None)
    return estimator.fit(*args, **kwargs)


def evaluate_trial(trial, estimator, x, y, date_num, fit_kwargs=None):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float).ravel()
    fit_kwargs = dict(fit_kwargs or {})
    use_eval_set = fit_kwargs.pop('use_eval_set', False)
    scores, coverage = [], []
    for fold_id, (train, valid) in enumerate(KFold(5, shuffle=True, random_state=0).split(x)):
        model = FoldModel(estimator)
        kwargs = dict(fit_kwargs)
        if use_eval_set:
            kwargs['eval_set'] = [(x[valid], y[valid])]
        model.fit(x[train], y[train], **kwargs)
        prediction = model.predict(x[valid])
        scores.append(original_unit_mape(y[valid], prediction))
        coverage.append({'fold': fold_id, 'train_count': len(train), 'validation_count': len(valid),
                         'mape_count': int(np.count_nonzero(y[valid])), 'mape': scores[-1]})
    # The published model is a fresh full-training fit, with its own fitted transforms.
    final = FoldModel(estimator).fit(x, y)
    score = float(np.mean(scores))
    directory = Path('optuna_' + date_num)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'sr_{trial.number}-{score}.pickle').open('wb') as stream:
        pickle.dump(scores, stream)
    with (directory / f'ty_{trial.number}-{score}.pickle').open('wb') as stream:
        pickle.dump(final, stream)
    (directory / f'coverage_{trial.number}.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    return score
