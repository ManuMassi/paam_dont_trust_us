import csv
import os

from itertools import islice
import numpy as np

# Now we read all the data into memory.
class RAMDADataset():
    def __init__(self, mode: str, feature_dirs: list, data_distribution_path: str, shuffle: bool = True, sampling_mode: str = 'downsampling'):
        self.mode = mode
        self.feature_dirs = feature_dirs
        self.data_distribution_path = data_distribution_path
        self.shuffle = shuffle
        self.sampling_mode = sampling_mode

        # Construct dataset
        self.benign_data, self.malicious_data, self.ben_apk, self.mal_apk = self._construct_dataset()
    
    def _construct_dataset(self):
        """
        Construct the dataset.
        """
        # Read the data distribution file
        # if not os.path.exists(self.data_distribution_path):
        #     raise FileNotFoundError(f'The data distribution file does not exist: {self.data_distribution_path}')
        # with open(self.data_distribution_path, 'r') as f:
        #     dic = eval(f.read())

        train_df, test_df = self.data_distribution_path
        dic  = {
            'train': train_df,
            'test': test_df
        }
        
        benign_data = []
        malicious_data = []
        ben_apk = []
        mal_apk = []
        # Traverse the feature directories
        for feature_dir in self.feature_dirs:
            # Read feature file
            feature_file = os.path.join(feature_dir, 'features.csv')
            if not os.path.exists(feature_file):
                raise FileNotFoundError(f'The feature file does not exist: {feature_file}')
            with open(feature_file, 'r') as f:
                csv_data = csv.reader(f)
                for line in islice(csv_data, 1, None):
                    sha256 = line[0]
                    vector = [float(x) for x in line[1:]]
                    labels = dic[self.mode].loc[
                        dic[self.mode]["sha256"] == sha256,
                        "label"
                    ].values
                    if len(labels) == 0:
                        continue
                    label = labels[0]
                    if sha256 in dic[self.mode]['sha256'].to_list():
                        if label == 0:
                            benign_data.append(vector)
                            ben_apk.append(sha256)
                        else:
                            malicious_data.append(vector)
                            mal_apk.append(sha256)
        
        benign_data = np.array(benign_data)
        malicious_data = np.array(malicious_data)

        # If validation or test, return the data directly
        if self.mode in ['val', 'test']:
            return benign_data, malicious_data, ben_apk, mal_apk
        
        if self.sampling_mode == 'downsampling':
            if len(benign_data) > len(malicious_data):
                benign_data = benign_data[:len(malicious_data)]
            else:
                malicious_data = malicious_data[:len(benign_data)]
        elif self.sampling_mode == 'oversampling':
            if len(benign_data) > len(malicious_data):
                repeat_num = len(benign_data) // len(malicious_data) + 1
                malicious_data = np.vstack([np.tile(malicious_data, (repeat_num, 1))[:len(benign_data), :]])
            else:
                repeat_num = len(malicious_data) // len(benign_data) + 1
                benign_data = np.vstack([np.tile(benign_data, (repeat_num, 1))[:len(malicious_data), :]])
        else:
            raise ValueError(f'Invalid sampling mode: {self.sampling_mode}')
        
        if self.shuffle:
            np.random.shuffle(benign_data)
            np.random.shuffle(malicious_data)

        return benign_data, malicious_data, ben_apk, mal_apk
    
    def len(self):
        return len(self.benign_data) + len(self.malicious_data)

    def get_data(self):
        return self.benign_data, self.malicious_data