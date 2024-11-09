import ast

def doc_to_choice(doc) -> list:
    choices = doc["randomized_option_order"]
    choices = [str(i) for i in ast.literal_eval(choices)]
    return choices