from nervaluate import Evaluator
from sklearn.metrics import classification_report

def evaluate_model_cr(preds, labels):
    return classification_report(preds, labels)

def evaluate_model_metrics(preds_list, label_list, tags = ["THE"]):
    evaluator = Evaluator(label_list, preds_list, tags = tags , loader= "list")
    results, results_per_tag = evaluator.evaluate()
    return results
