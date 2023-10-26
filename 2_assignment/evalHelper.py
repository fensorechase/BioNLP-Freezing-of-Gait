import json
import sklearn.metrics as skm


def read_json(infile):
    """
    Load a json file
    """
    with open(infile, 'r') as ifile:
        return json.load(ifile)


def evaluate_results(model, test_x, test_y):
    """
    Requited eval metrics:
        - Accuracy
        - micro-averaged F1 score
        - Macro-averaged F1 score
    """
    # evaluate on test
    #y_hat = model.predict_proba(test_x)[:, 1]
    #auc = skm.roc_auc_score(test_y, y_hat)
    y_hat = model.predict(test_x) #[:, 1]
    accuracy = skm.accuracy_score(test_y, y_hat)
    microF1 = skm.f1_score(y_true=test_y, y_pred=y_hat, average='micro')
    macroF1 = skm.f1_score(y_true=test_y, y_pred=y_hat, average='macro')
    #aps = skm.average_precision_score(test_y, y_hat)
    return accuracy, microF1, macroF1


def get_train_test(df, i, label="fog_q_class"):
    test_mask = df[label+"_folds"] == i
    train_df = df[~test_mask]
    test_df = df[test_mask]
    # setup y
    train_y = train_df[label]
    test_y = test_df[label]
    return train_df, test_df, train_y, test_y
