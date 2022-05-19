
def computeMetrics(eval_pred):
    #experiment = comet_ml.get_global_experiment()
    metric0 = load_metric("accuracy")
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("f1")


    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric0.compute(predictions=predictions, references=labels)["accuracy"]
    precision = metric1.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="macro")["f1"]

    experiment.log_confusion_matrix(predictions, labels)
    experiment.log_metric("accuracy",accuracy,epoch=N_EPOCHS)
    experiment.log_metric("precision",precision,epoch=N_EPOCHS)
    experiment.log_metric("recall",recall,epoch=N_EPOCHS)
    experiment.log_metric("f1",f1,epoch=N_EPOCHS)
    print(predictions,labels)

    return {"accuracy":accuracy,"precision": precision, "recall": recall, "f1":f1}
