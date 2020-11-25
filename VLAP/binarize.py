from sklearn.preprocessing import MultiLabelBinarizer

def binarize(y_train, y_val):
    
    # Fit the multi-label binarizer on the training set
    print("Labels:")
    mlb = MultiLabelBinarizer()
    mlb.fit(y_train)

    # Loop over all labels and show them
    N_LABELS = len(mlb.classes_)
    for (i, label) in enumerate(mlb.classes_):
        print("{}. {}".format(i, label))

    # transform the targets of the training and test sets
    y_train_bin = mlb.transform(y_train)
    y_val_bin = mlb.transform(y_val)

    # Print example of movie posters and their binary targets
    print("some examples of binarized response")
    for i in range(3):
        print(X_train[i], y_train_bin[i])

    return(y_train_bin, y_val_bin)
