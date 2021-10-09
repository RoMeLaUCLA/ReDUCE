import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from datetime import datetime
from PIL import Image as im


def to_pandas(data, normalize=True, use_normalization=[]):
    """

    :param data:
    :param normalize:
    :param use_normalization: pre-calculated normalization scaler. If this parameter is given, the function uses it to normalize.
    :return:
    """
    cols = [
        'center1x','center1y','center2x','center2y','center3x','center3y',
        'angle1','angle2','angle3',
        'width1','width2','width3',
        'height1','height2','height3',
        'mode0','mode1',
        'width_in_hand',
        'height_in_hand'
        ]
    np_data = np.zeros((len(data), len(cols)))
    count = 0
    for d in data:
        center = np.array(d['center']).flatten()
        angle  = np.array(d['angle'])
        width  = np.array(d['width'])
        height = np.array(d['height'])
        # Turn two modes into one hot encoding
        mode   = np.array([1, 0]) if d['mode'] == 0 else np.array([0, 1]) 
        width_in_hand = np.array(d['width_in_hand'])
        height_in_hand = np.array(d['height_in_hand'])
        np_data[count, :] = np.hstack((center, angle, width, height, mode, width_in_hand, height_in_hand))
        count += 1

    # Normalize data types to [0, 1]
    if normalize:
        if use_normalization == []:
            min_max_scaler = preprocessing.MinMaxScaler()
            np_data_scaled = min_max_scaler.fit_transform(np_data)
            # for i in range(len(cols)):
            #     val = np.expand_dims(np_data[:, i], axis=1)  # Make it a column vector
            #     np_data[:, i] = min_max_scaler.fit_transform(val).flatten()

            return pd.DataFrame(np_data_scaled, columns=cols), min_max_scaler

        else:
            min_max_scaler = use_normalization
            np_data_scaled = min_max_scaler.transform(np_data)

            return pd.DataFrame(np_data_scaled, columns=cols)

    else:
        return pd.DataFrame(np_data, columns=cols)


def main():
    rand = 0  # seed decision tree model
    split = 0.3  # percentage to split data
    normalize = False # normalize the data
    save_model = True  # save the trained model
    visualize = True  # visualize wrongly classified during test

    fn = "classification_data"
    ext = "pkl"
    # rel_path = "simulation2D/bin_problem_python/labeled_data/"
    rel_path = "labeled_data/"

    # Open file
    with open(f"{rel_path + fn + '.' + ext}", "rb") as input_file:
        data_raw = pickle.load(input_file)

    # Transform and convert to pandas
    if normalize:
        data, normalization_scaler = to_pandas(data_raw, normalize=normalize)
    else:
        data = to_pandas(data_raw, normalize=normalize)

    # Labels want to predict
    labels = np.squeeze(np.dstack((data['mode0'], data['mode1'])))

    # Remove the labels features
    data = data.drop('mode0', axis=1)
    data = data.drop('mode1', axis=1)

    # Save column names of data
    col_names = list(data.columns)

    # Change to numpy array
    data_np = np.array(data)
    print(data_np)

    # Split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(data_np, labels, test_size=split, random_state=rand)

    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(
        criterion='gini',  # gini, entropy
        n_estimators=1000,
        random_state=rand
        )

    # Train model
    rf.fit(train_features, train_labels)

    # Use forest to predict test
    predictions = rf.predict(test_features)

    # Calculate absolute errors
    errors = abs(predictions - test_labels)
    num_right = int(len(errors) - np.sum(errors) / 2)
    total = len(errors)

    # Print out mean absolute error
    print(f"Mean absolute error: {round(np.mean(errors), 2)}")
    print(f"Correct predictions vs total: {num_right} / {total}")
    print(f"Percent correct: {num_right/total*100:.2f} %")

    # Save model
    if save_model:
        ext = datetime.now().strftime("%H%M%S")
        # with open(f"simulation2D/bin_problem_python/models/randomforest_{ext}.pkl", "wb") as f:
        with open(f"models/randomforest_{ext}.pkl", "wb") as f:
            pickle.dump(rf, f)
        if normalize:
            with open(f"models/normalization_scaler_{ext}.pkl", "wb") as f:
                pickle.dump(normalization_scaler, f)
    
    # Visualize wrongly classified
    if visualize:
        wrongs = test_features[np.sum(errors,axis=1) != 0]
        images = []
        in_hand = []
        for w in wrongs:
            idx = data[data.isin(w)].dropna().index.item()
            raw = data_raw[idx]
            images.append(raw["image"])
            in_hand.append((raw["width_in_hand"], raw["height_in_hand"]))
            # print(raw)

        # Draw incorrectly classified items
        predictions = predictions[[np.sum(errors,axis=1) != 0]]
        test_labels = test_labels[[np.sum(errors,axis=1) != 0]]
        plt.ion()
        plt.show()
        for i in range(len(images)):
            print(f"Wrongly classified {i+1}/{len(images)}")
            print(f"Classified as: {0 if np.all(predictions[i] == np.array([1, 0])) else 1}")
            print(f"Correct label: {0 if np.all(test_labels[i] == np.array([1, 0])) else 1}") 
            print(f"Item in hand: {in_hand[i]}")  
            plt.imshow(images[i].convert("RGB"))
            plt.draw()
            plt.pause(1.0)
            # print("test")


if __name__ == "__main__":
    import sys

    sys.exit(main())