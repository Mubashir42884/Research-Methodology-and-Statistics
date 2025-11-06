# src/utils_timer.py
import time

def timed_run(label, model, train_func, runtimes, *args, **kwargs):
    """
    Utility to measure runtime of model training or evaluation.

    Parameters
    ----------
    label : str
        Label/name of the model (e.g., "[1/7] Decision Tree").
    model : object
        ML model instance.
    train_func : function
        Function that accepts (model, *args, **kwargs) and returns (model, result).
    runtimes : dict
        Dictionary to store runtime results.
    *args, **kwargs :
        Arguments passed to train_func.

    Returns
    -------
    model : object
        Trained model.
    result : object
        Returned result from train_func (e.g., importances, scores).
    """
    print(f"\n{label}")
    start = time.time()
    model, result = train_func(model, *args, **kwargs)
    elapsed = time.time() - start
    runtimes[label] = round(elapsed, 3)
    print(f"Model Runtime: {elapsed:.3f} sec")
    return model, result
