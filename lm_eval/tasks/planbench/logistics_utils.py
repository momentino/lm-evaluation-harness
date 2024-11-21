from typing import Dict
from pathlib import Path
from lm_eval.tasks.planbench.planbench_eval_utils.response_evaluation import ResponseEvaluator


def process_results_plan_generation(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file = current_folder_abs / "planbench_eval_utils" / "config_files" / "logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result  =response_evaluator.evaluate_plan("plan_generation", doc, response)
    return {"acc" : result}

def process_results_plan_optimality(doc, response) -> Dict:
    current_folder = Path(__file__).parent
    current_folder_abs = current_folder.resolve()
    config_file =  current_folder_abs / "planbench_eval_utils"/"config_files"/"logistics.json"
    response_evaluator = ResponseEvaluator(config_file)
    response = response[0]
    result = response_evaluator.evaluate_plan("plan_optimality", doc, response)
    return {"acc" : result}