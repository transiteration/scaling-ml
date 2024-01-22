import ray

from scripts import predict


def get_label(text, predictor):
    sample_ds = ray.data.from_items([{"headline": text, "keyword": "", "category": ""}])
    results = predict.predict_proba(ds=sample_ds, predictor=predictor)
    return results[0]["prediction"]


def decode(indices, index_to_class):
    return [index_to_class[index] for index in indices]
