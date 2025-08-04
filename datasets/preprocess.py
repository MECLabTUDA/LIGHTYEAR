import os
import json
import numpy as np


def save_images_from_dict(data_dict, output_dir):
    """
    Saves each image in the dictionary as a .npy file.

    Args:
        data_dict (dict): A dictionary with keys 'x' and 'y'.
                          'x' should be a list of lists (each list of length 784).
                          'y' should be a list of integer labels.
        output_dir (str): Directory where .npy files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    data_dict = data_dict["user_data"]

    for user in data_dict:
        x_list = data_dict[user]["x"]
        y_list = data_dict[user]["y"]

        for idx, (image_flat, label) in enumerate(zip(x_list, y_list)):
            image_array = np.array(image_flat).reshape((28, 28))
            filename = f"user_{user}_{idx}_label_{label}.npy"
            filepath = os.path.join(output_dir, filename)
            np.save(filepath, image_array)

        print(f"Saved {len(x_list)} images to '{output_dir}'.")


if __name__ == '__main__':
    
    train_files = sorted(os.listdir('train'))
    test_files  = sorted(os.listdir('test'))
    

    for idx, (train_f, test_f) in enumerate(zip(train_files, test_files)):
        
        path = f'clients/client_{idx}'
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(path, 'test'), exist_ok=True)


        with open(os.path.join('train', train_f), 'r') as f:
            train = json.load(f)

        with open(os.path.join('test', test_f), 'r') as f:
            test = json.load(f)

        save_images_from_dict(train, os.path.join(path, 'train'))
        save_images_from_dict(test, os.path.join(path, 'test'))





















