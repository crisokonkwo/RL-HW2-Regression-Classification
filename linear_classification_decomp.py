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
    pairs = [(img, lbl) for img, lbl in zip(all_images, all_labels) if lbl == label1 or lbl == label2]
    X = np.asarray([img for img, lbl in pairs], dtype=np.float64)
    Y = np.asarray([1 if lbl == label1 else 0 for img, lbl in pairs], dtype=np.float64)
    print("Number of samples:", X.shape[0], "Number of features:", X.shape[1])
    

    # drop redundant columns in X (exact duplicates cause singularity)
    # keep_cols = X.var(axis=0) > 0.0
    # X = X[:, keep_cols]

    # add a column of ones to the dataset (bias term)
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    m, n = X.shape
    # print(f"Rows (samples) of X must match labels in Y (got {len(Y)} vs {m})")
    # print("After dropping constant columns and adding bias term, shape of X:", X.shape, "shape of Y:", Y.shape, "number of kept features:", np.sum(keep_cols))

    # Multidimensional least squares (X^TX)^(-1)X^TY
    # weights = np.linalg.pinv(X.T @ X) @ X.T @ Y

    tol = 1e-15  # eigenvalue tolerance; values below are treated as zero
    if m >= n:
        print("Using (X^T X)^+ X^T (tall/square case)")
        XtX = X.T @ X
        
        eigenvals, eigenvecs = np.linalg.eig(XtX)
        eigenvals = np.real(eigenvals)
        eigenvecs = np.real(eigenvecs)
        
        # if eigenvalues are less than tol, treat as zero to avoid undefined inversions this is equivalent to the linear independence of the columns of X
        lin_ind_col = eigenvals > tol
        rank = int(np.count_nonzero(lin_ind_col))
        print(f"Eigenvalues kept (Linearly Independent columns of X): {rank}/{len(eigenvals)} (tol={tol})")

        # Compute the pseudo-inverse using the new eigenvalues
        inv_diag = np.zeros_like(eigenvals)
        inv_diag[lin_ind_col] = 1.0 / eigenvals[lin_ind_col]
        print("Inverted eigenvalues diagonal shape:", np.diag(inv_diag).shape)

        # pseudoinverse XtX^+ using the eigen decomposition
        XtX_pinv = eigenvecs @ np.diag(inv_diag) @ eigenvecs.T
        weights = XtX_pinv @ X.T @ Y
    else:
        # Wide case: use X^T (X X^T)^+ Y
        print("Using X^T (X X^T)^+ (wide case)")
        XXt = X @ X.T
        
        eigenvals, eigenvecs = np.linalg.eigh(XXt)
        eigenvals = np.real(eigenvals)
        eigenvecs = np.real(eigenvecs)

        lin_ind_col = np.abs(eigenvals) > tol
        rank = int(np.count_nonzero(lin_ind_col))
        print(f"Eigenvalues kept (Linearly Independent columns of X): {rank}/{len(eigenvals)} (tol={tol})")
        
        inv_diag = np.zeros_like(eigenvals)
        inv_diag[lin_ind_col] = 1.0 / eigenvals[lin_ind_col]
        print("Inverted eigenvalues diagonal shape:", np.diag(inv_diag).shape)
        
        XXt_pinv = eigenvecs @ np.diag(inv_diag) @ eigenvecs.T
        weights = X.T @ XXt_pinv @ Y

    return weights


# required for graduate students only
def get_optimal_thresh(images_train, labels_train, w):
    pairs = [(img, lbl) for img, lbl in zip(images_train, labels_train) if lbl == label1 or lbl == label2]
    X = np.asarray([img for img, lbl in pairs], dtype=np.float64)
    Y = np.asarray([1 if lbl == label1 else 0 for img, lbl in pairs], dtype=np.float64)
    
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    predictions = X @ w
    
    best_thresh, best_acc = 0.5, 0
    for t in np.linspace(0, 1, 1001):
        # print(t)
        pred_labels = (predictions > t).astype(int)
        accuracy = (pred_labels == Y).mean()
        if accuracy > best_acc:
            best_acc, best_thresh = accuracy, t

    return best_thresh, best_acc


def test(all_images_test, all_labels_test, label1, label2, w, thresh):
    pairs = [(img, lbl) for img, lbl in zip(all_images_test, all_labels_test) if lbl == label1 or lbl == label2]
    X_test = np.asarray([img for img, lbl in pairs], dtype=np.float64)
    G_truth = np.asarray([1 if lbl == label1 else 0 for img, lbl in pairs], dtype=np.float64)

    # X_test = X_test[:, keep_cols]
    
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
                
                print("Calculating optimal threshold on training set...")
                optimal_thresh, optimal_acc = get_optimal_thresh(images_list, labels_list, weights)
                print(f"Optimal threshold (on training set): {optimal_thresh:.3f}")
                print(f"Training accuracy with optimal threshold: {optimal_acc*100:.2f}%")

                # Test the model on the test set using the optimal threshold
                accuracy_optimal_thresh = test(images_list_test, labels_list_test, label1, label2, weights, optimal_thresh)
                print(f"Test accuracy with optimal threshold: {accuracy_optimal_thresh*100:.2f}%\n")
                
                
                # Optional: Visualize the data and decision boundary using PCA (for 2D visualization)
                # from sklearn.decomposition import PCA
                # import matplotlib.pyplot as plt
                # X = np.asarray([img for img, lbl in pairs], dtype=np.float64)
                # Y = np.asarray([1 if lbl == label1 else 0 for img, lbl in pairs], dtype=np.float64)
                # X = np.hstack((np.ones((X.shape[0], 1)), X))
                # pca = PCA(n_components=2)
                # X_2d = pca.fit_transform(X[:, 1:])  # exclude bias for PCA
                
                # plt.figure(figsize=(8, 6))
                # plt.scatter(X_2d[Y==0, 0], X_2d[Y==0, 1], c='red', label=str(label2), alpha=0.8, edgecolor='k', linewidth=0.5)
                # plt.scatter(X_2d[Y==1, 0], X_2d[Y==1, 1], c='blue', label=str(label1), alpha=0.8, edgecolor='k', linewidth=0.5)
                
                # # Project weights into PCA space
                # w_pca = pca.transform(weights[1:].reshape(1, -1))  # exclude bias term
                # w_bias = weights[0]  # bias term
                
                # # Transform weights into PCA 2D space
                # w_2d = np.array([w_pca[0, 0], w_pca[0, 1]])

                # # Decision boundary: w0 + w1*x + w2*y = 0.5  ->  y = -(w1*x + w0 - 0.5)/w2
                # xx = np.linspace(np.min(X_2d[:, 0]), np.max(X_2d[:, 0]), 100)
                # yy = -(w_2d[0]*xx + w_bias - 0.5) / (w_2d[1]+1e-12)  # add small term to avoid division by zero
                # plt.plot(xx, yy, linestyle='-', color='green', label="Decision boundary")
                # plt.title(f"Least-Squares Classifier: Training Data + Decision Boundary for {label1} vs {label2}")
                # plt.legend()
                # plt.show()