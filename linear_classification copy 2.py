from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.decomposition import PCA


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()


def train(all_images, all_labels, label1, label2):
    # filter the data for the two labels
    filter = [(img, lbl) for img, lbl in zip(all_images, all_labels) if lbl == label1 or lbl == label2]
    X = np.array([img for img, lbl in filter])
    Y = np.array([1 if lbl == label1 else 0 for img, lbl in filter])
    
    # add a column of ones to the dataset (bias term)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # compute the weights using multidimensional least squares (X^TX)^(-1)X^TY
    # weights = np.linalg.pinv(X.T @ X) @ X.T @ Y
    
    """_summary_
    # the numpy.linalg.pinv function primarily uses Singular Value Decomposition (SVD) to compute the Moore-Penrose pseudoinverse of a matrix.
    # This is particularly useful in linear regression problems where the design matrix may not be square or may be ill-conditioned.
    # it first performs the SVD on the input matrix X, which factors the matrix into X=U @ Sigma @ V^{T}, where X is an m x n matrix
    # U and V are orthogonal matrices. U will be an m x m matrix, and V will be an n x n matrix.
    # Sigma m x n is a diagonal matrix containing the singular values of X. The singular values sigma _{i} are always non-negative and are often ordered from largest to smallest. 
    # The pseudoinverse is then computed using the singular values in Sigma, where non-zero singular values are inverted, and zero singular values remain zero.
    # Finally, the pseudoinverse X^{+} is constructed as X^{+}=V Sigma^{+} U^{T}.
    # This method is numerically stable and effective for solving linear systems, especially in the context of least squares problems.
    # It is widely used in machine learning and statistics for fitting linear models to data.
    # In the context of linear regression, using the pseudoinverse allows us to find the best-fitting weights even when the system of equations is underdetermined or overdetermined.
    # By applying the pseudoinverse, we can obtain a solution that minimizes the sum of squared differences between the observed and predicted values, leading to an optimal linear model for classification tasks.
    
    Strategy:
    1. Perform SVD on the input matrix X to obtain U, Sigma, and V^{T}.
    2
    
    """
    # Compute a Moore–Penrose pseudoinverse using only matrix multiplication and inverse (no SVD, no pinv)
    m, n = X.shape
    print("m, n:", m, n)
    if m >= n:
        # X is tall or square (X^T @ X)^{-1} X^T
        # Calculate the matrix \(X^{T}X\)
        XtX = X.T @ X
        
        # Find the eigenvalues and eigenvectors of X^{T}X
        eigenvals, eigenvecs = np.linalg.eig(XtX)

        # Singular values are the square roots of the non-zero eigenvalues
        singvals = np.sqrt(np.maximum(eigenvals, 0))
        
        # Arrange the singular values in descending order.
        sorted_index = np.argsort(singvals)[::-1]
        singvals = singvals[sorted_index]
        
        # Sort eigenvectors according to the sorted indices of singular values        
        V = eigenvecs[:, sorted_index]
        print("V shape:", V.shape)
        
        # The rank of matrix \(X\) is found by putting \(X\) in echelon form and counting the pivots.
        # This can be done using Gaussian elimination or by finding the row echelon form.
        # In practice, we often use the singular values to determine the rank.
        # The rank of the  matrix using the singular values (singular values > 1e-10 are considered non-zero)
        rank = np.count_nonzero(singvals > 1e-10)
        # print("rank:", rank, "out of", len(singvals))

        # Construct the diagonal matrix Sigma of the same size as X placing the singular values along the diagonal in descending order and filling the rest with zeros.
        Sigma = np.zeros((m, n))
        for i in range(rank):
            Sigma[i, i] = singvals[i]
        print("Sigma shape:", Sigma.shape)
        
        # Left singular vectors
        # Compute U using the formula U = X @ V @ Sigma^{+}
        Sigma_pinv = np.zeros((n, m))
        for i in range(rank):
            Sigma_pinv[i, i] = 1 / singvals[i]
        print("Sigma_pinv shape:", Sigma_pinv.shape)
        
        
        U = np.zeros((m, m))
        for i in range(rank):
            U[:, i] = (X @ V[:, i]) / singvals[i]
        print("U shape:", U.shape)
        
        #TODO: Fill extra columns of U (if m>r) with orthonormal basis e.g. via Gram-Schmidt
        if m > rank:
            # Use QR decomposition to find an orthonormal basis for the remaining columns
            Q, R = np.linalg.qr(U[:, :rank])
            U[:, :rank] = Q[:, :rank]
            # Fill the remaining columns with orthonormal vectors
            for i in range(rank, m):
                vec = np.random.rand(m)
                for j in range(i):
                    vec -= np.dot(vec, U[:, j]) * U[:, j]
                vec /= np.linalg.norm(vec)
                U[:, i] = vec

        # Finally, we can compute the pseudoinverse using the formula X^{+} = V @ Sigma^{+} @ U^{T}
        X_pinv = V @ Sigma_pinv @ U.T
        
        weights = X_pinv @ Y
    # else:
    #     # X is wide X^T (X X^T)^{-1}
    #     XtX = X.T @ X
    #     XtX_inv = np.linalg.inv(XtX)
    #     X_pinv = XtX_inv @ X.T

        weights = X_pinv @ Y

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
                # optimal_thresh = get_optimal_thresh(images_list, labels_list, weights)
                # print(f"Optimal threshold (on training set): {optimal_thresh:.3f}")
                # # Test the model on the test set using the optimal threshold
                # accuracy_optimal_thresh = test(images_list_test, labels_list_test, label1, label2, weights, optimal_thresh)
                # print(f"Test accuracy with optimal threshold: {accuracy_optimal_thresh*100:.2f}%\n")