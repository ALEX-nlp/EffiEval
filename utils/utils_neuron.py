"""
This file is adapted from project MUI-Eval (https://github.com/ALEX-nlp/MUI-Eval)
"""
import csv
import math
import re

from thefuzz import process


def count_lines(file_path):
    with open(file_path, 'rb') as f:
        return sum(1 for _ in f)

def write_csv_row(values, filename):
    try:
        with open(filename, "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(values)
    except Exception as e:
        print(f"Failed to open or write to file: {e}")

def extract_choice(gen, choice_list, choices=["A", "B", "C", "D"]):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )
    # 
    if res is None:
        res = re.search(r"\\boxed{\\text{(A|B|C|D)}}", gen)
    
    if res is None:
        res = re.search(r"\\boxed{(A|B|C|D)}", gen)

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)
        # res = re.search(r"^(?:[ABCD]|\\boxed\{[ABCD]\}|\\boxed\{\\text\{[ABCD]\}\})(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)

def extract_choice_hellaswag(gen, choice_list, choices=["1", "2", "3", "4"]):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^1234]{0,20}?(?:n't|not))[^1234]{0,10}?\b(?:|is|:|be))\b)[^1234]{0,20}?\b(1|2|3|4)\b",
        gen,
    )
    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(1|2|3|4)\b(?![^1234]{0,8}?(?:n't|not)[^1234]{0,5}?(?:correct|right))[^1234]{0,10}?\b(?:correct|right)\b",
            gen,
        )
    
    if res is None:
        res = re.search(r"\\boxed{\\text{(1|2|3|4)}}", gen)
    
    if res is None:
        res = re.search(r"\\boxed{(1|2|3|4)}", gen)

    # straight answer: A
    if res is None:
        res = re.search(r"^(1|2|3|4)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(1|2|3|4)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)

def extract_answer(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        # print(f"The last digit in {s} is {last_digit}")
    else:
        last_digit = None
    return last_digit



def is_correct(completion, answer):
    gold = extract_answer(answer)
    assert gold is not None, "No ground truth answer found in the document."

    def number_equal(answer, pred):
        if pred is None:
            return False
        try:
            return math.isclose(eval(answer), eval(pred), rel_tol=0, abs_tol=1e-4)
        except:
            # print(
            #     f"cannot compare two numbers: answer={answer}, pred={pred}", flush=True
            # )
            return False

    return number_equal(gold, extract_answer(completion))