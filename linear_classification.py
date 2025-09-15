from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()


def train(all_images, all_labels, label1, label2):
    # filter the data for the two labels
    filter = [(img, lbl) for img, lbl in zip(all_images, all_labels) if lbl == label1 or lbl == label2]
    X = np.asarray([img for img, lbl in filter], dtype=np.float64)
    Y = np.asarray([1 if lbl == label1 else 0 for img, lbl in filter])
    
    # add a column of ones to the dataset (bias term)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Multidimensional least squares (X^TX)^(-1)X^TY
    # weights = np.linalg.pinv(X.T @ X) @ X.T @ 

    m, n = X.shape    
    r = np.linalg.matrix_rank(X)
    print("shape:", X.shape, "rank:", r, "min(m,n):", min(m, n))

    # print(f"Y must have one label per row of X (got {len(Y)} vs {m})")

    reg = 1e-6  # small regularization parameter
    if m >= n:
        # X is tall or square (X^T @ X)^{-1} X^T
        XtX = X.T @ X
        # print("X^T X shape:", XtX.shape, "rank:", np.linalg.matrix_rank(XtX))
        # Add regularization to the diagonal to avoid singularity
        XtX_reg = XtX + reg * np.eye(n)
        XtX_inv = np.linalg.inv(XtX_reg)
        weights = XtX_inv @ X.T @ Y
    else:
        # X is wide n>m (X^T (X X^T)^{-1})
        XXt = X @ X.T
        XXt_reg = XXt + reg * np.eye(m)
        XXt_inv = np.linalg.inv(XXt_reg)
        weights = X.T @ XXt_inv @ Y

    return weights


# required for graduate students only
def get_optimal_thresh(images_train, labels_train, w):
    filter = [(img, lbl) for img, lbl in zip(images_train, labels_train) if lbl == label1 or lbl == label2]
    X = np.array([img for img, lbl in filter])
    Y = np.array([1 if lbl == label1 else 0 for img, lbl in filter])
    
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    preds = X @ w
    
    best_thresh, best_acc = 0.5, 0
    for t in np.linspace(0, 1, 201):
        # print(t)
        pred_labels = (preds > t).astype(int)
        correct = (pred_labels == Y)
        num_correct = np.sum(correct)
        acc = num_correct / len(Y)
        if acc > best_acc:
            best_acc, best_thresh = acc, t
            
    return best_thresh


def test(all_images_test, all_labels_test, label1, label2, w, thresh):
    filter = [(img, lbl) for img, lbl in zip(all_images_test, all_labels_test) if lbl == label1 or lbl == label2]
    X_test = np.array([img for img, lbl in filter])
    Y_test = np.array([1 if lbl == label1 else 0 for img, lbl in filter])
    
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    predictions = X_test @ w
    predictions = (predictions > thresh).astype(int)
    correct = (predictions == Y_test)
    num_correct = np.sum(correct)
    accuracy = num_correct / len(Y_test)
    
    return accuracy


if __name__ == "__main__":

    # load the data
    mndata = MNIST('./datasets/MNIST/raw')
    images_list, labels_list = mndata.load_training()
    images_list_test, labels_list_test = mndata.load_testing()

    # Your code goes here
    print("Number of training samples:", len(images_list))
    print("Number of testing samples:", len(images_list_test))
    print("Image size:", np.array(images_list).shape[1])
    unique_labels = set(labels_list)
    print("Labels:", unique_labels)
    
    # plot_image(images_list[0])
    
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
                optimal_thresh = get_optimal_thresh(images_list, labels_list, weights)
                print(f"Optimal threshold (on training set): {optimal_thresh:.3f}")
                # Test the model on the test set using the optimal threshold
                accuracy_optimal_thresh = test(images_list_test, labels_list_test, label1, label2, weights, optimal_thresh)
                print(f"Test accuracy with optimal threshold: {accuracy_optimal_thresh*100:.2f}%\n")