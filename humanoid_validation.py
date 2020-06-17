import random

from mlflow import log_metric
from flask import Flask


class validation():
    def __init__(self):
        self.no_validation = 0 # Value('i', 0)
        self.no_validation_pass = 0 # Value('i', 0)

    def increment_total_count(self):
        self.no_validation = self.no_validation + 1

    def increment_success_count(self):
        self.no_validation_pass = self.no_validation_pass + 1

    def get_total_count(self):
        return self.no_validation

    def get_success_count(self):
        return self.no_validation_pass

    def get_accuracy(self):
        return self.no_validation_pass/self.no_validation

my_validation = validation()

# total_no_validation = 0
# total_no_validation_pass = 0

app = Flask(__name__)

@app.route('/validate/<input>')
def validate(input):
    # Attention! in real time, a real human should be doing the validation rather than a pseudo random number generator
    human_validation_result = random.randint(0,1000)

    my_validation.increment_total_count()
    if human_validation_result >= 400:
        my_validation.increment_success_count()
    
    accuracy = my_validation.get_accuracy()

    log_metric(key="accuracy", value=accuracy, step=my_validation.get_total_count())
    return f"current input = {input}, current evaluation = {human_validation_result}, \n  total count = {my_validation.get_total_count()}, pass_count = {my_validation.get_success_count()}, \n rolling average accuracy = {accuracy}"