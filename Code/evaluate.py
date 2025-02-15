import json
import numpy as np

file_name = "./cross_domain/mcq_test_sciq_cl_infonce_005.json"

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

results = load_data(file_name)

# evaluating
print("Evaluating...")
avg_eval = {
    "P@1": 0.0, "R@1": 0.0, "F1@1": 0.0, "P@3": 0.0, "R@3": 0.0, "F1@3": 0.0,
    "P@5": 0.0, "R@5": 0.0, "F1@5": 0.0, "P@10": 0.0, "R@10": 0.0, "F1@10": 0.0,
    "MRR@5": 0.0, "MAP@5": 0.0, "NDCG@1": 0.0, "NDCG@3": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0
}

def evaluate(result):
    eval = {
        "P@1": 0.0, "R@1": 0.0, "F1@1": 0.0, "P@3": 0.0, "R@3": 0.0, "F1@3": 0.0,
        "P@5": 0.0, "R@5": 0.0, "F1@5": 0.0, "P@10": 0.0, "R@10": 0.0, "F1@10": 0.0,
        "MRR@5": 0.0, "MAP@5": 0.0, "NDCG@1": 0.0, "NDCG@3": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0
    }
    distractors = [d.lower() for d in result["distractors"]]
    generations = [d.lower() for d in result["generated_distractors"]]
    #generations = [word.strip() for item in generations for word in item.split(',')]



    relevants = [int(generation in distractors) for generation in generations]

    # P@1
    eval["P@1"] = relevants[0] if len(relevants) > 0 else 0

    # R@1
    eval["R@1"] = relevants[:1].count(1) / len(distractors)

    # F1@1
    try:
        eval["F1@1"] = (2 * eval["P@1"] * eval["R@1"]) / (eval["P@1"] + eval["R@1"])
    except ZeroDivisionError:
        eval["F1@1"] = 0

    # P@3
    eval["P@3"] = relevants[:3].count(1) / 3 if len(relevants) >= 3 else 0

    # R@3
    eval["R@3"] = relevants[:3].count(1) / len(distractors)

    # F1@3
    try:
        eval["F1@3"] = (2 * eval["P@3"] * eval["R@3"]) / (eval["P@3"] + eval["R@3"])
    except ZeroDivisionError:
        eval["F1@3"] = 0

    # P@5
    eval["P@5"] = relevants[:5].count(1) / 5 if len(relevants) >= 5 else 0

    # R@5
    eval["R@5"] = relevants[:5].count(1) / len(distractors)

    # F1@5
    try:
        eval["F1@5"] = (2 * eval["P@5"] * eval["R@5"]) / (eval["P@5"] + eval["R@5"])
    except ZeroDivisionError:
        eval["F1@5"] = 0

    # P@10
    eval["P@10"] = relevants[:10].count(1) / 10 if len(relevants) >= 10 else 0

    # R@10
    eval["R@10"] = relevants[:10].count(1) / len(distractors)

    # F1@10
    try:
        eval["F1@10"] = (2 * eval["P@10"] * eval["R@10"]) / (eval["P@10"] + eval["R@10"])
    except ZeroDivisionError:
        eval["F1@10"] = 0

    # MRR@5
    eval["MRR@5"] = next((1 / (i+1) for i in range(5) if i < len(relevants) and relevants[i] == 1), 0)

    # MAP@5
    rel_num = 0
    for i in range(min(5, len(relevants))):
        if relevants[i] == 1:
            rel_num += 1
            eval["MAP@5"] += rel_num / (i+1)
    eval["MAP@5"] = eval["MAP@5"] / len(distractors) if distractors else 0

    # NDCG@1
    eval["NDCG@1"] = ndcg_at_k(relevants, 1)

    # NDCG@3
    eval["NDCG@3"] = ndcg_at_k(relevants, 3)

    # NDCG@5
    eval["NDCG@5"] = ndcg_at_k(relevants, 5)

    # NDCG@10
    eval["NDCG@10"] = ndcg_at_k(relevants, 10)

    return eval

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

for result in results:
    eval = evaluate(result)
    for k in avg_eval.keys():
        avg_eval[k] += eval[k]

# calculate average
for k in avg_eval.keys():
    avg_eval[k] /= len(results)

# show evaluation
for k in avg_eval.keys():
    print(f"{k}: {avg_eval[k]*100}%")

print("Done!")
