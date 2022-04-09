import myDatasets

def loadMyData(FLAGS, demon=None):
    raw_datasets = myDatasets.getDS(FLAGS)
    print("Reading dataset completed") if FLAGS.verbose else print("", end='')

    # encode the labels
    raw_datasets['y_train1'], labels_dict = labels_to_indices(raw_datasets['y_train1'])
    raw_datasets['y_train2'], labels_dict = labels_to_indices(raw_datasets['y_train2'], labels_dict)
    raw_datasets['y_valid'], _            = labels_to_indices(raw_datasets['y_valid'], labels_dict)
    raw_datasets['y_tests'], _            = labels_to_indices(raw_datasets['y_tests'], labels_dict)

    raw_datasets['y_train_t1'], genre_dict = labels_to_indices(raw_datasets['y_train_t1'])
    raw_datasets['y_train_t2'], genre_dict = labels_to_indices(raw_datasets['y_train_t2'], genre_dict)
    raw_datasets['y_valid_t'], _           = labels_to_indices(raw_datasets['y_valid_t'], genre_dict)
    raw_datasets['y_tests_t'], _           = labels_to_indices(raw_datasets['y_tests_t'], genre_dict)

    train_raw = (raw_datasets['x_train1'], raw_datasets['y_train1'], raw_datasets['y_train_t1'],
                 raw_datasets['x_train2'], raw_datasets['y_train2'], raw_datasets['y_train_t2'])
    valid_raw = (raw_datasets['x_valid'], raw_datasets['y_valid'], raw_datasets['y_valid_t'])
    tests_raw = (raw_datasets['x_tests'], raw_datasets['y_tests'], raw_datasets['y_tests_t'])

    return train_raw, valid_raw, tests_raw


def labels_to_indices(y, labels=None):
    y_ = []

    labels = {} if labels is None else labels
    for y_i in y:
        if y_i not in labels.keys():
            labels[y_i] = len(labels)

        y_.append(labels[y_i])

    return y_, labels
