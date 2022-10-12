import matplotlib.pyplot as plt
import numpy as np
import dataprovider as data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from joblib import dump, load

if __name__ == "__main__":

    data_provider = data.ImageTrainDataProvider()
    data_provider.Read_All_Images()
    data_provider.Proccess_All_Images()
    data_provider.Setup_Feature_Vectors()

    plt.subplot(2, 3, 1), plt.imshow(data_provider.rock_list[0])
    plt.subplot(2, 3, 2), plt.imshow(data_provider.paper_list[0])
    plt.subplot(2, 3, 3), plt.imshow(data_provider.scissors_list[0])
    plt.subplot(2, 3, 4), plt.imshow(data_provider.processed_rock_list[0], cmap='gray')
    plt.subplot(2, 3, 5), plt.imshow(data_provider.processed_paper_list[0], cmap='gray')
    plt.subplot(2, 3, 6), plt.imshow(data_provider.processed_scissors_list[0], cmap='gray')
    plt.show()

    rocklabels = np.full(len(data_provider.rock_feature_vectors),  1)
    paperlabels = np.full(len(data_provider.paper_feature_vectors),  2)
    scissorslabels = np.full(len(data_provider.scissors_feature_vectors),  3)

    X = np.concatenate((data_provider.rock_feature_vectors, data_provider.paper_feature_vectors, data_provider.scissors_feature_vectors))
    y = np.concatenate((rocklabels, paperlabels, scissorslabels), axis=0)
    import pickle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LinearSVC(random_state=0, tol=1e-5)
    print("model created")
    print("Starting training")
    print(data_provider.rock_feature_vectors[3].shape)
    clf.fit(X_train, y_train)
    print("Training Finished")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(acc)
    print(clf.predict(data_provider.rock_feature_vectors[3].reshape(1, -1)))
    dump(clf, 'clf.joblib')

