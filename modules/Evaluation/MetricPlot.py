


def evaluate_metrics(self, model: ModelWrap) -> dict:
    y_pred = model.predict(self.X_test)
    metrics = {"accuracy": accuracy_score(self.y_test, y_pred),
               "f1_score": f1_score(self.y_test, y_pred, average="macro"),
               "precision_score": precision_score(self.y_test, y_pred, average="macro"),
               "recall_score": recall_score(self.y_test, y_pred, average="macro")}
    return metrics