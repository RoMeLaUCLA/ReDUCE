import os
from sklearn import cluster
from sklearn import preprocessing
from util import load_data, ints_to_one_hot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pickle
import pdb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib import offsetbox


# TODO: This is to do unsupervised learning to generate clustered data, change name !


def main():
    # Parameters
    split_test = 0.05  # percentage to withhold from predicting (supervised)
    rand = 0  # random seed
    cluster = "solution"  # "solution" or "feature" will train first cluster using
    save = True  # save data and model
    normalize = True  # normalize the data before fitting if so adjust eps by dividing by 10
    plot2d = [True, (0, 1)]  # plot 2 dimension of tuple elements (x, y), can only plot up to 20 clusters

    # Parameters for OPTICS (unsupervised) fitting--basic ones
    min_samples = 5  # 5 Number of samples in neighborhood for point to be considered as a core point
    xi = 0.01  # 0.01 Determines the minimum steepness on the reachability plot that constitutes a cluster boundary
    min_cluster_size = 0.018  # 0.018 Minimum number of samples in cluster, expressed as a fraction of the number of samples

    # Parameters for DBSCAN which can recover OPTICS with certain threshold
    eps = 0.80  # 0.86 0.7, 70 Threshold on reachability, this will affect clustering and outliers

    # Parameters for Random Forest--basic
    criterion = 'gini'  # Splitting criterion: gini, entropy
    n_estimators = 1000  # Number of estimators

    # Load data
    dir_path = os.path.dirname(os.path.realpath(__file__))  # Path to directory of main
    fld = "/../data/cluster4"
    data_paths = [dir_path + fld + '/dataset_unsupervised_learning_part4.p',
                  dir_path + fld + '/dataset_unsupervised_learning_part5.p',
                  dir_path + fld + '/dataset_unsupervised_learning_part6.p']

    all_data = load_data(data_paths)

    # Pick which side to cluster
    data = np.array(all_data[cluster])

    # Normalize data
    if normalize:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

    print(f"===============CLUSTERING===============")
    # Unsupervised clustering using OPTICS
    clust = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    clust.fit(data)  # Clustering on all data

    # Adjust reachability threshold for DBSCAN
    labels_dbscan = cluster_optics_dbscan(
        reachability=clust.reachability_,
        core_distances=clust.core_distances_,
        ordering=clust.ordering_,
        eps=eps
    )

    # TODO: Adjust labeling by changing to how the clusters are spatially located
    #       by doing this can escape blowing up in output dimension can also try
    #       ordinal labelling but associated distances of similarity differs

    print(f"Number of clusters: {np.max(labels_dbscan) + 1}")
    print(f"Number of outliers: {np.sum(labels_dbscan == -1)}")
    print(f"Number of usable data for training: {len(data) - np.sum(labels_dbscan == -1)}/{len(data)}")

    # Get indices that are not outliers
    non_outlier_idx = [i for i in list(range(len(labels_dbscan))) if labels_dbscan[i] != -1]

    # Remove data outliers
    solution_data = np.array([all_data["solution"][i] for i in non_outlier_idx])
    feature_data = np.array([all_data["feature"][i] for i in non_outlier_idx])
    labels = ints_to_one_hot([np.array([labels_dbscan[i] for i in non_outlier_idx])])[0]  # Change to one hot
    all_data["label"] = np.argmax(labels, axis=1).tolist()

    print(f"===============PREDICTION===============")
    # Split data into training and testing sets
    # Normalize data

    if normalize:
        feature_scaler = MinMaxScaler()
        feature_data = feature_scaler.fit_transform(feature_data)

    train_features, test_features, train_labels, test_labels = train_test_split(feature_data, labels,
                                                                                test_size=split_test, random_state=rand)

    # Convert to one hot
    train_labels_one_hot, test_labels_one_hot = (train_labels, test_labels)  # already one hot

    # TODO: Try simple neural network or K-nearest neighbors

    # Instantiate model with 1000 decision trees
    rf_classifier = RandomForestClassifier(
        criterion=criterion,
        n_estimators=n_estimators,
        random_state=rand
    )

    rf_classifier.fit(train_features, train_labels_one_hot)

    # Use forest to predict test
    predictions = rf_classifier.predict(test_features)

    # Count and get rid of zero predictions
    # handle_unknown so not to raise error with give zero vector
    predictions_fix = []
    test_labels_one_hot_fix = []
    all_count = len(predictions)
    zero_count = 0
    for i in range(len(predictions)):
        if predictions[i].tolist() != np.zeros_like(predictions[i]).tolist():
            predictions_fix.append(predictions[i].tolist())
            test_labels_one_hot_fix.append(test_labels_one_hot[i].tolist())
        else:
            zero_count += 1

    predictions = np.array(predictions_fix)
    test_labels_one_hot = np.array(test_labels_one_hot_fix)

    # print("Predictions are :")
    # print(predictions)
    # print("===========================================================")
    # print("Test labels are :")
    # print(test_labels_one_hot)

    # Calculate absolute errors
    errors = np.sum(np.any(predictions - test_labels_one_hot, axis=1))
    total = len(test_labels_one_hot)
    num_right = total - errors

    # Print out mean absolute error
    print(f"Number of zero predictions: {zero_count}/ {all_count}")
    print(f"Mean absolute error: {round(np.mean(errors), 2)}")
    print(f"Correct predictions vs total: {num_right} / {total}")
    print(f"Percent correct: {num_right / total * 100:.2f} %")

    if save:
        print(f"==================SAVE==================")
        # Save data with labels
        save_path = os.path.dirname(os.path.realpath(__file__))
        with open(save_path + "/saves/combined_data.pkl", 'wb') as f:
            pickle.dump(all_data, f)

        # Save model
        with open(save_path + "/saves/model.pkl", "wb") as f:
            pickle.dump(rf_classifier, f)

        if normalize:
            # Save feature scaling
            with open(save_path + "/saves/feature_scaler.pkl", "wb") as f:
                pickle.dump(feature_scaler, f)

    if plot2d[0]:
        print(f"==================PLOT==================")

        # Create figure
        plt.figure(figsize=(10, 7))
        G = gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(G[1, 1])
        ax2 = plt.subplot(G[1, 0])
        ax3 = plt.subplot(G[0, 0])
        ax4 = plt.subplot(G[0, 1])

        # Create a list of colors for plotting clusters
        colors = plt.get_cmap('tab20').colors  # Can only plot up to 20 clusters
        c_even = [colors[idx] for idx in range(len(colors)) if idx % 2 == 0]
        c_odd = [colors[idx] for idx in range(len(colors)) if (idx + 1) % 2 == 0]
        colors = c_even + c_odd  # Shuffle order of similar colors for contrast

        # Get dimensional index of elements to plot
        x_data = np.array(all_data["feature"])[:, plot2d[1][0]]
        y_data = np.array(all_data["feature"])[:, plot2d[1][1]]

        # Plot data
        for idx in range(np.max(labels_dbscan + 1)):
            x = x_data[labels_dbscan == idx]
            y = y_data[labels_dbscan == idx]
            ax1.plot(x, y, color=colors[idx], alpha=0.9, marker='.', linestyle='None')
        ax1.plot(x_data[labels_dbscan == -1], y_data[labels_dbscan == -1], 'k+', alpha=0.2)
        ax1.set_title(f'Clustering Feature at {eps} epsilon cut\nDBSCAN')
        ax1.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-1'])

        # Get dimensional index of elements to plot
        x_data = np.array(all_data["solution"])[:, plot2d[1][0]]
        y_data = np.array(all_data["solution"])[:, plot2d[1][1]]

        # Plot data
        for idx in range(np.max(labels_dbscan + 1)):
            x = x_data[labels_dbscan == idx]
            y = y_data[labels_dbscan == idx]
            ax2.plot(x, y, color=colors[idx], alpha=0.9, marker='.', linestyle='None')
        ax2.plot(x_data[labels_dbscan == -1], y_data[labels_dbscan == -1], 'k+', alpha=0.2)
        ax2.set_title(f'Clustering Solution at {eps} epsilon cut\nDBSCAN')
        ax2.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-1'])

        # TSNE Representation of the data
        if normalize:
            solution_scaler = MinMaxScaler()
            solution_data = solution_scaler.fit_transform(solution_data)
        manifold = TSNE(
            n_components=2,
            init='pca',
            random_state=rand
        )
        X = manifold.fit_transform(solution_data)

        # Scale and visualize the embedding vectors
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        for i in range(X.shape[0]):
            ax3.text(X[i, 0], X[i, 1], str(all_data["label"][i]),
                     color=plt.cm.Set1(all_data["label"][i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        # TODO: Fix color coding scheme

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                # imagebox = offsetbox.AnnotationBbox(
                #     offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                #     X[i])
                text_image = offsetbox.TextArea(str(all_data["label"][i]))
                imagebox = offsetbox.AnnotationBbox(text_image, X[i])
                ax3.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        ax3.set_title('TSNE Solutions with OPTICS labeling')

        # TSNE for feature data--this may not mean much since clusters is on solution side
        manifold = TSNE(
            n_components=2,
            init='pca',
            random_state=rand
        )
        X = manifold.fit_transform(feature_data)

        # Scale and visualize the embedding vectors
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        for i in range(X.shape[0]):
            ax4.text(X[i, 0], X[i, 1], str(all_data["label"][i]),
                     color=plt.cm.Set1(all_data["label"][i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                # imagebox = offsetbox.AnnotationBbox(
                #     offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                #     X[i])
                text_image = offsetbox.TextArea(str(all_data["label"][i]))
                imagebox = offsetbox.AnnotationBbox(text_image, X[i])
                ax4.add_artist(imagebox)
        ax4.set_title('TSNE Features with OPTICS labeling')

        # TODO: Add another clustering for overlapping regions in feature space to
        #       get multiple policies for a sort of robustness

        print("Plotting...")
        plt.tight_layout()
        plt.show()

        # pdb.set_trace()


if __name__ == "__main__":
    import sys, os

    sys.exit(main())