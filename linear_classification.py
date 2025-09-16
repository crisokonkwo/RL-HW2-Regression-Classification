from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (20,20)), cmap='gray_r')
    plt.show()


def train(all_images, all_labels, label1, label2):
    # filter the data for the two labels
    pairs = [(img, lbl) for img, lbl in zip(all_images, all_labels) if lbl == label1 or lbl == label2]
    X = np.asarray([img for img, lbl in pairs], dtype=np.float64).reshape(-1, 28, 28)
    Y = np.asarray([1 if lbl == label1 else 0 for img, lbl in pairs], dtype=np.float64)
    print("Number of samples:", X.shape[0], "Size of images:", X.shape[1:])

    # crop the images 
    X_crop = X[:, 4:24, 4:24]
        
    # flatten the images back to vectors
    X = X_crop.reshape(X.shape[0], -1)
    
    # add a column of ones to the dataset (bias term)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    print("After dropping rows and columns and adding bias term, shape of X:", X.shape, "shape of Y:", Y.shape)

    # Multidimensional least squares (X^TX)^(-1)X^TY
    # weights = np.linalg.pinv(X.T @ X) @ X.T @ Y

    
    print("Training with (X^T X)^(-1) X^T for tall/square case")
    XtX = X.T @ X
    inv_XtX = np.linalg.inv(XtX)
    weights = inv_XtX @ X.T @ Y

    return weights


# required for graduate students only
def get_optimal_thresh(images_train, labels_train, w):
    pairs = [(img, lbl) for img, lbl in zip(images_train, labels_train) if lbl == label1 or lbl == label2]
    X = np.asarray([img for img, lbl in pairs], dtype=np.float64).reshape(-1, 28, 28)
    Y = np.asarray([1 if lbl == label1 else 0 for img, lbl in pairs], dtype=np.float64)
    
    X_crop = X[:, 4:24, 4:24]
    X = X_crop.reshape(X.shape[0], -1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    predictions = X @ w
    
    best_thresh, best_acc = 0.5, 0
    for t in np.linspace(0, 1, 1001):
        pred_labels = (predictions > t).astype(int)
        accuracy = (pred_labels == Y).mean()
        if accuracy > best_acc:
            best_acc, best_thresh = accuracy, t

    return best_thresh, best_acc


def test(all_images_test, all_labels_test, label1, label2, w, thresh):
    pairs = [(img, lbl) for img, lbl in zip(all_images_test, all_labels_test) if lbl == label1 or lbl == label2]
    X_test = np.asarray([img for img, lbl in pairs], dtype=np.float64).reshape(-1, 28, 28)
    G_truth = np.asarray([1 if lbl == label1 else 0 for img, lbl in pairs], dtype=np.float64)

    X_crop = X_test[:, 4:24, 4:24]
    X_test = X_crop.reshape(X_test.shape[0], -1)
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    predictions = X_test @ w
    predictions = (predictions > thresh).astype(int)
    accuracy = (predictions == G_truth).mean()

    return accuracy


if __name__ == "__main__":

    # load the data
    mndata = MNIST('./datasets/MNIST/raw')
    images_list, labels_list = mndata.load_training()
    images_list_test, labels_list_test = mndata.load_testing()

    # Your code goes here
    print("Number of training samples:", len(images_list))
    print("Number of testing samples:", len(images_list_test))
    unique_labels = set(labels_list)
    print("Labels:", unique_labels)
    
    # plot_image(images_list[1])
    
    pair_counter = 0
    for label1 in unique_labels:
        for label2 in unique_labels:
            if label1 < label2:
                pair_counter += 1
                print(f"Training classifier on digits {label1} and {label2} ({pair_counter}/45)")
                weights = train(images_list, labels_list, label1, label2)
                
                print("Testing...")
                accuracy = test(images_list_test, labels_list_test, label1, label2, weights, 0.5)
                print(f"Test accuracy: {accuracy*100:.2f}%")
                
                print("Calculating optimal threshold on training set...")
                optimal_thresh, optimal_acc = get_optimal_thresh(images_list, labels_list, weights)
                print(f"Optimal threshold (on training set): {optimal_thresh:.3f}")
                print(f"Training accuracy with optimal threshold: {optimal_acc*100:.2f}%")

                # Test the model on the test set using the optimal threshold
                accuracy_optimal_thresh = test(images_list_test, labels_list_test, label1, label2, weights, optimal_thresh)
                print(f"Test accuracy with optimal threshold: {accuracy_optimal_thresh*100:.2f}%\n")