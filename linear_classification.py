from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()


def train(all_images, all_labels, label1, label2, crop):
    # filter the data for the two labels
    pairs = [(img, lbl) for img, lbl in zip(all_images, all_labels) if lbl == label1 or lbl == label2]
    X_train = np.asarray([img for img, lbl in pairs], dtype=np.float64).reshape(-1, 28, 28)
    Y = np.asarray([1 if lbl == label1 else 0 for img, lbl in pairs], dtype=np.float64)
    print("Number of samples:", X_train.shape[0], "Size of images:", X_train.shape[1:])

    # crop the images and flatten the images back to vectors
    r0,r1,c0,c1 = crop
    X_crop = X_train[:, r0:r1, c0:c1]
    print("Cropped image shape:", X_crop.shape[1:])
    X_crop = X_crop.reshape(X_train.shape[0], -1)
    
    # Find all-zero rows and columns
    col_sums = X_crop.sum(axis=0)
    zero_cols = (col_sums == 0)

    # Add tiny noise to those rows/columns to avoid singularity
    X_crop[:, zero_cols] = np.random.normal(0, 1e-15, size=(X_crop.shape[0], zero_cols.sum()))

    print(np.linalg.matrix_rank(X_crop), "out of", X_crop.shape[1], "features are linearly independent")

    # plot_image(X_crop[1200])

    # add a column of ones to the dataset (bias term)
    X = np.hstack([np.ones((X_crop.shape[0], 1)), X_crop])
    print("Number of samples:", X.shape[0], "Number of features:", X.shape[1])
    
    print("After dropping rows and columns and adding bias term, shape of X:", X.shape, "shape of Y:", Y.shape)

    # m, n = X.shape
    # if m >= n:
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    weights = XtX_inv @ X.T @ Y
    
    # Multidimensional least squares (X^TX)^(-1)X^TY
    # weights = np.linalg.pinv(X.T @ X) @ X.T @ Y
    
    return weights


# required for graduate students only
def get_optimal_thresh(images_train, labels_train, w, label1, label2, crop):
    pairs = [(img, lbl) for img, lbl in zip(images_train, labels_train) if lbl == label1 or lbl == label2]
    X_thresh = np.asarray([img for img, lbl in pairs], dtype=np.float64).reshape(-1, 28, 28)
    Y_thresh = np.asarray([1 if lbl == label1 else 0 for img, lbl in pairs], dtype=np.float64)

    r0,r1,c0,c1 = crop
    X_crop = X_thresh[:, r0:r1, c0:c1]
    X_crop = X_crop.reshape(X_thresh.shape[0], -1)
    
    col_sums = X_crop.sum(axis=0)
    zero_cols = (col_sums == 0)

    X_crop[:, zero_cols] = np.random.normal(0, 1e-15, size=(X_crop.shape[0], zero_cols.sum()))

    X = np.hstack((np.ones((X_crop.shape[0], 1)), X_crop))
    
    predictions = X @ w
    
    # Spread threshold options between min and max of the predictions 
    thresh_options = np.linspace(predictions.min(), predictions.max(), 1000)
    
    best_thresh, best_acc = 0.5, 0.0
    for t in thresh_options:
        pred_labels = (predictions > t).astype(int)
        accuracy = (pred_labels == Y_thresh).mean()
        if accuracy > best_acc:
            best_acc, best_thresh = accuracy, t

    return best_thresh, best_acc


def test(all_images_test, all_labels_test, label1, label2, w, thresh, crop):
    pairs = [(img, lbl) for img, lbl in zip(all_images_test, all_labels_test) if lbl == label1 or lbl == label2]
    X_test = np.asarray([img for img, lbl in pairs], dtype=np.float64).reshape(-1, 28, 28)
    G_truth = np.asarray([1 if lbl == label1 else 0 for img, lbl in pairs], dtype=np.float64)

    r0,r1,c0,c1 = crop
    X_crop = X_test[:, r0:r1, c0:c1]
    X_crop = X_crop.reshape(X_test.shape[0], -1)
    
    col_sums = X_crop.sum(axis=0)
    zero_cols = (col_sums == 0)

    X_crop[:, zero_cols] = np.random.normal(0, 1e-15, size=(X_crop.shape[0], zero_cols.sum()))

    X = np.hstack((np.ones((X_crop.shape[0], 1)), X_crop))

    predictions = X @ w
    predictions = (predictions > thresh).astype(int)
    accuracy = (predictions == G_truth).mean()

    return accuracy


def plot_avg_digit(images, labels, digit, crop):
    imgs = np.asarray([img for img, lbl in zip(images, labels) if lbl == digit], dtype=np.float64)
    avg_img = imgs.mean(axis=0).reshape(28, 28)
    
    r0,r1,c0,c1 = crop
    avg_img = avg_img[r0:r1, c0:c1]
    return avg_img


def compare_digits(images, labels, label1=7, label2=9, crop=(4,24,4,24)):
    avg1 = plot_avg_digit(images, labels, label1, crop)
    avg2 = plot_avg_digit(images, labels, label2, crop)
    diff = avg1 - avg2

    fig, axes = plt.subplots(1, 3, figsize=(10,4))
    axes[0].imshow(avg1, cmap="gray_r")
    axes[0].set_title(f"Average {label1}")
    axes[1].imshow(avg2, cmap="gray_r")
    axes[1].set_title(f"Average {label2}")
    im = axes[2].imshow(diff, cmap="bwr")
    axes[2].set_title(f"Difference ({label1}-{label2})")
    fig.colorbar(im, ax=axes[2])
    plt.show()


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
    
    crop = (5,23,5,23) # (row_start, row_end, col_start, col_end)
    
    # plot_image(images_list[1200])
    compare_digits(images_list, labels_list, 4, 9, crop)
    
    pair_counter = 0
    for label1 in unique_labels:
        for label2 in unique_labels:
            if label1 < label2:
                pair_counter += 1
                print(f"Training {label1} vs {label2} ({pair_counter}/45)")
                w = train(images_list, labels_list, label1, label2, crop)

                test_acc = test(images_list_test, labels_list_test, label1, label2, w, 0.5, crop)
                print(f"Test accuracy with 0.5 threshold: {test_acc*100:.2f}%")

                thresh_opt, train_acc_opt = get_optimal_thresh(images_list, labels_list, w, label1, label2, crop)
                test_acc_opt = test(images_list_test, labels_list_test, label1, label2, w, thresh_opt, crop)

                print(f"Best thresh: {thresh_opt:.6f} | Train accuracy: {train_acc_opt*100:.2f}% | Test accuracy: {test_acc_opt*100:.2f}%\n")
                print("--------------------------------------------------")