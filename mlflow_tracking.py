import os
from random import random, randint

from mlflow import log_metric, log_param, log_artifacts, set_experiment

if __name__ == "__main__":
    print("Running mlflow_tracking.py")
    set_experiment(f"exp_time_evolution")

    for i in range(0,10):
        log_metric("foo", random(), step=i)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

    log_artifacts("outputs")
