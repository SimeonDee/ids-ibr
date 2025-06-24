import joblib


def load_encoding():
    return joblib.load("models/encodings_dict.joblib")


def load_scaler():
    return joblib.load("models/scaler.joblib")


def load_model(model_name: str):
    if model_name.lower().startswith("random"):
        return joblib.load("models/Random_Forest_model.pkl")
    elif model_name.lower().startswith("svm"):
        return joblib.load("models/SVM_model.pkl")
    elif model_name.lower().startswith("knn"):
        return joblib.load("models/KNN_model.pkl")
    elif model_name.lower().startswith("decision"):
        return joblib.load("models/Decision_Tree_model.pkl")
    else:
        raise ValueError("Invalid value selected.")


def get_encodings_for_column(encoding_dict: dict, col_name: str):
    return encoding_dict[col_name]


def get_encoding_labels_for_col(encoding_dict: dict, col_name: str):
    return list(encoding_dict[col_name].values())


def get_encoding_values_for_col(encoding_dict: dict, col_name: str):
    return list(encoding_dict[col_name].keys())


def get_encoding_value_for_label(
    encoding_dict: dict,
    col_name: str,
    label: int
):
    return list(encoding_dict[col_name].keys())[label]


def get_encoding_label_for_value(
    encoding_dict: dict,
    col_name: str,
    value: str
):
    return encoding_dict[col_name][value]


def get_reverse_encoding_dict(encoding_dict: dict):
    reverse_encoding = {
        col: {
            lbl: val for (val, lbl) in encoding_dict[col].items()
        } for col in encoding_dict
    }
    return reverse_encoding