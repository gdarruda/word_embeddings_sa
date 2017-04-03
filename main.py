import dataset
import model
import csv
import numpy as np
from Callback import My_Callback

max_paragraph_length = dataset.get_max_paragraph_length()


def save_results(results: list, file_name: str):

    file = open("output/" + file_name, 'w')
    writer = csv.DictWriter(file, fieldnames=results[0].keys())

    writer.writeheader()

    for result in results:
        writer.writerow(result)

    file.close()


for i in range(1, 11):

    callback = My_Callback(2.)

    training = [x for x in range(1, 11) if x != i]
    test = [i]

    paragraph_training, labels_training = dataset.get_fold_sample(training)
    paragraph_validation, labels_validation = dataset.get_fold_sample(test)

    X_training = dataset.format_matrix(paragraph_training, max_paragraph_length)
    X_validation = dataset.format_matrix(paragraph_validation, max_paragraph_length)

    Y_training = np.array(labels_training)
    Y_validation = np.array(labels_validation)

    callback.set_validation(X_validation, Y_validation)

    compiled_model = model.get_model(max_paragraph_length)

    compiled_model.fit(X_training, Y_training, epochs=50, batch_size=50, callbacks=[callback])

    save_results(callback.results, "fold_" + str(i) + ".csv")
