import numpy as np


def confusion_matrix(targets, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix.tolist()


def classwise_accuracy(matrix, class_names):
    rows = np.array(matrix, dtype=np.float64)
    results = []
    for class_index, class_name in enumerate(class_names):
        total = float(rows[class_index].sum())
        correct = float(rows[class_index, class_index])
        accuracy = correct / total if total > 0 else None
        results.append(
            {
                "class_id": class_index,
                "class_name": class_name,
                "samples": int(total),
                "correct": int(correct),
                "accuracy": accuracy,
            }
        )
    return results


def top_misclassified_samples(records, class_names, top_k):
    misclassified = [record for record in records if record["target"] != record["prediction"]]
    misclassified.sort(key=lambda item: item["confidence"], reverse=True)
    results = []
    for record in misclassified[:top_k]:
        results.append(
            {
                "image_id": record["image_id"],
                "index": record["index"],
                "target_id": record["target"],
                "target_name": class_names[record["target"]],
                "prediction_id": record["prediction"],
                "prediction_name": class_names[record["prediction"]],
                "confidence": record["confidence"],
            }
        )
    return results
