from datasets import load_dataset
from check_equiv import is_equivalent

# Load the dataset
equiv_dataset = load_dataset("nbalepur/QG_vs_QA_AE")

# print size

test_set = equiv_dataset['test']


results = {}

for i in range(len(test_set)):
    entry = test_set[i]
    gold_label = entry['label']
    model = entry['model']

    entity = entry['entity']
    answer = entry['answer']
    category = entry['category']

    if model not in results:
        results[model] = {
            'gold_labels': [],
            'pred_labels': [],
        }

    results[model]['gold_labels'].append(gold_label)
    equiv = is_equivalent(entity, answer, category)

    if equiv:
        results[model]['pred_labels'].append(1)
    else:
        results[model]['pred_labels'].append(0)

for model, labels in results.items():
    gold_labels = labels['gold_labels']
    pred_labels = labels['pred_labels']

    correct = sum([1 for i in range(len(gold_labels)) if gold_labels[i] == pred_labels[i]])
    accuracy = correct / len(gold_labels)

    print(f"Model: {model}")
    print(f"Accuracy: {accuracy}")
    print(f"Correct: {correct}")
    print(f"Total: {len(gold_labels)}")
    print("\n")
