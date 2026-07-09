from qiskit.primitives import Sampler  # Should work
from qiskit_machine_learning.kernels import FidelityQuantumKernel  # Should work

from pathlib import Path
# import tensorflow as tf
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import requests
import seaborn as sns

#Importing Libraries
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression

import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

#from qiskit.utils import algorithm_globals
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
# from qiskit_algorithms.state_fidelities import ComputeUncompute
# from qiskit_machine_learning.kernels import FidelityQuantumKernel

# from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit_machine_learning.algorithms.classifiers import QSVC
# from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime import QiskitRuntimeService#, SamplerV2 as Sampler

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import transpile
from sklearn.svm import SVC
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from qiskit.circuit.library import RealAmplitudes
# from qiskit_algorithms.optimizers import COBYLA
# from qiskit.primitives import BaseSampler
#from qiskit.primitives import Sampler
#from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, SamplerV2

np.random.seed(42)
#algorithm_globals.random_seed = 123

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

#######################################NEW DATASET
class KDD:
    def __init__(self, test_size=0.2, random_state=42):
        # Load KDDCup99 dataset
        data = fetch_kddcup99(as_frame=True, percent10=True)
        df = data.frame

        target_col = df.columns[-1]  # Get the last column name
        
        # Separate features and target
        self.X = df.drop(columns=[target_col])
        # self.y = df[target_col]
        # print(self.y[:10])
        # # Encode categorical features
        # for col in self.X.select_dtypes(include=["object"]).columns:
        #     self.X[col] = LabelEncoder().fit_transform(self.X[col])
        
        # # Encode labels (handle bytes if necessary)
        # if self.y.dtype == 'object' or isinstance(self.y.iloc[0], bytes):
        #     # Convert bytes to string if needed
        #     self.y = self.y.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        # self.y = LabelEncoder().fit_transform(self.y)
        y_raw = df[target_col]
        if y_raw.dtype == 'object' or isinstance(y_raw.iloc[0], bytes):
            y_raw = y_raw.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
        # Convert: normal. = 0, everything else = 1
        self.y = (y_raw != 'normal.').astype(int)
        
        # Encode categorical features
        for col in self.X.select_dtypes(include=["object"]).columns:
            self.X[col] = LabelEncoder().fit_transform(self.X[col])
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        
        # Compute Pearson ranking once
        self.rank_features = self._pearson_rank_features()
    
    def _pearson_rank_features(self):
        """
        Rank features using absolute Pearson correlation with the target.
        Returns a list of feature names ordered from highest to lowest correlation.
        """
        df_corr = pd.concat(
            [self.X_train, pd.Series(self.y_train, name="target", index=self.X_train.index)],
            axis=1
        )
        correlations = df_corr.corr()["target"].drop("target").abs()
        ranked_features = correlations.sort_values(ascending=False).index.tolist()
        return ranked_features
    
    def dataset(self, dimensions, samples=100000):
        """
        Return train/test sets using top-N Pearson-ranked features,
        balanced with 50% class 0 and 50% class 1.
        """
        selected_features = self.rank_features[:dimensions]

        # Number of samples per class
        samples_per_class = samples // 2
        unique, counts = np.unique(self.y, return_counts=True)

        # print("Class distribution:")
        # for cls, cnt in zip(unique, counts):
        #     print(f"Class {cls}: {cnt} samples")

        # Indices for each class
        idx_class_0 = np.where(self.y == 0)[0]
        idx_class_1 = np.where(self.y == 1)[0]
        print("Total samples for class (0/normal, 1/abnormal):",idx_class_0.shape," , ",idx_class_1.shape)

        # Reproducible random sampling
        rng = np.random.default_rng(seed=42)
        idx_0_sampled = rng.choice(idx_class_0, size=samples_per_class, replace=False)
        idx_1_sampled = rng.choice(idx_class_1, size=samples_per_class, replace=False)
        
        print(idx_0_sampled.shape)
        print(idx_1_sampled.shape)

        # Combine and shuffle
        sampled_indices = np.concatenate([idx_0_sampled, idx_1_sampled])
        rng.shuffle(sampled_indices)

        # Subset X and y
        X_sampled = self.X.iloc[sampled_indices][selected_features]
        y_sampled = self.y[sampled_indices]

        # Train-test split (no stratification needed — already balanced)
        X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
            X_sampled,
            y_sampled,
            test_size=0.2,
            random_state=42
        )

        return X_train_sel, X_test_sel, y_train_sel, y_test_sel

#######################################
class Metrics():
    def get_confusion_matrix_elements(self, test_labels, predictions):
            TP = np.sum((test_labels == 1) & (predictions == 1))
            TN = np.sum((test_labels == 0) & (predictions == 0))
            FP = np.sum((test_labels == 0) & (predictions == 1))
            FN = np.sum((test_labels == 1) & (predictions == 0))
            return TP, TN, FP, FN

    def mean_qubits(self, backend, property):
        mean = 0
        for i in range(0,backend.num_qubits):
            if(property == 'readout_error'):
                mean += backend.properties().readout_error(i)
            elif(property=='t1'):
                mean += backend.properties().t2(0)
            elif(property=='t2'):
                mean += backend.properties().t2(0)
        mean=mean/backend.num_qubits
        return mean

    def json_qiskit(self, name):
        TOKEN = "482cfbd6dec57e562dd79640807643b60b7388db6a0ff63e30b4bd92d4d55601a902c6f005413b8a69f6677daf23c82fa14626594ab3656d9c8c0e9229f920de"

        response = requests.request(
            "GET",
            "https://api.quantum-computing.ibm.com/runtime/workloads/me",
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer "+TOKEN
            },
            )
        # Parse the JSON response
        usage_seconds = 0
        estimated_running_time_seconds = 0
        data = response.json()
        # Iterate over the 'workloads' list to find the latest backend with the name
        for workload in data['workloads']:
            if workload['backend'] == name:
                usage_seconds = workload.get('usage_seconds', None)
                estimated_running_time_seconds = workload.get('estimated_running_time_seconds', None)
                break
            else:
                #print(f"Backend {name} not found in the workloads.")
                continue
        return usage_seconds, estimated_running_time_seconds

    def calculate_h_m_s_ms(self, end, start):
        # Calculate elapsed time in seconds
        elapsed_time_seconds = end - start
        # Convert elapsed time into hours, minutes, seconds
        hours = int(elapsed_time_seconds // 3600)
        minutes = int((elapsed_time_seconds % 3600) // 60)
        seconds = int(elapsed_time_seconds % 60)
        milliseconds = int((elapsed_time_seconds - int(elapsed_time_seconds)) * 1000)
        return elapsed_time_seconds#[hours, minutes, seconds, milliseconds]

    def calculate_metrics(self, test_labels, predictions):
            TP, TN, FP, FN = self.get_confusion_matrix_elements(test_labels, predictions)
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0 #recall=sensitivity
            sensitivity = recall
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            # calculate sensitivity and specificity
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return TP, TN, FP, FN, accuracy, precision, sensitivity, specificity, f1_score

    def calculate_metrics_hardware(self, backend):
        #Read the json according to the backend you are using and get the: 'usage_seconds', 'estimated_running_time_seconds'
        usage_seconds, estimated_running_time_seconds = self.json_qiskit(backend.name)
        mean_readout_error = self.mean_qubits(backend,'readout_error')
        mean_t1 = self.mean_qubits(backend,'t1')
        mean_t2 = self.mean_qubits(backend,'t1')

        return usage_seconds, estimated_running_time_seconds,mean_readout_error, mean_t1, mean_t2

    def should_skip_execution(self, results_file, samples, start_at):
        if os.path.exists(results_file):
            df_existing = pd.read_csv(results_file)
            # If there are existing records with the same sample count and dimensions greater than or equal to start_at, return True
            if ((df_existing["Samples"] == samples) & (df_existing["Dimension"] >= start_at)).any():
                return True
        
        return False

    def metrics_of_evaluation_classicaly(self, svc, dimension, end, start, test_features, test_labels):
        predictions = svc.predict(test_features)
        TP, TN, FP, FN, accuracy, precision, sensitivity, specificity, f1_score = self.calculate_metrics(test_labels, predictions)
        usage_time = end - start
        df_results = pd.DataFrame({
            'Dimension': dimension,
            'TP': [TP],
            'TN': [TN],
            'FP': [FP],
            'FN': [FN],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Sensitivity': [sensitivity],
            'Specificity': [specificity],
            'F1 Score': [f1_score],
            'Elapsed Time (s)': [usage_time],
            'Usage (s)': [usage_time],
        })
        return df_results

#######################################
class QPCA():
    def __init__(self):
        self.metrics = Metrics()
        self.kdd_data = KDD()

    def classical_pca(self,X_train, y_train, X_test, y_test, dimension):
        kernelpca_rbf = KernelPCA(n_components=dimension, kernel='rbf') #we changed 2 to dimentions
        kernelpca_rbf.fit(X_train)
        train_features_rbf = kernelpca_rbf.transform(X_train)
        test_features_rbf = kernelpca_rbf.transform(X_test)
        LR = LogisticRegression() #other models
        LR.fit(train_features_rbf, y_train)
        return LR, train_features_rbf, test_features_rbf
        
    def main_cpca_correlation(self,samples=500, start_at=2, end_at=10, results_file = "results/cpca.csv"):
        kdd_data = KDD()
        file_exists = os.path.exists(results_file)
        for dimension in range(start_at, end_at):
            if not self.metrics.should_skip_execution(results_file, samples, dimension):
                # Pass samples parameter to dataset method
                X_train, X_test, y_train, y_test = kdd_data.dataset(
                    dimensions=dimension,
                    samples=samples
                )
                
                print(f"Dimension {dimension}, Total Samples: {samples}")
                print("Shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
                
                start = time.time()
                model, X_train_transformed, X_test_transformed = self.classical_pca(
                    X_train, y_train, X_test, y_test, dimension)
                
                df_model = self.metrics.metrics_of_evaluation_classicaly(
                    model, dimension, time.time(), start, X_train_transformed, y_train
                )
                df_model["Samples"] = samples
                
                file_exists = os.path.exists(results_file)
                df_model.to_csv(results_file, mode='a', index=False, header=not file_exists)
        
        return True

    def qpca_precomputed_qkernel(self,X_train, y_train, X_test, y_test, dimension):
        # # Hrdware setup
        # backend = service.backend(name)
        # sampler = Sampler(backend)
        
        feature_map = ZZFeatureMap(feature_dimension=dimension, reps=2)
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

        # Compute kernel matrix
        kernel_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
        kernel_matrix_test = quantum_kernel.evaluate(x_vec=X_test)

        # # Perform clustering
        # clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
        # predicted_labels = clustering.fit_predict(kernel_matrix)

        # return predicted_labels, backend
        kernelpca_q = KernelPCA(n_components=dimension, kernel='precomputed')
        train_features_q = kernelpca_q.fit_transform(kernel_matrix_train)
        test_features_q = kernelpca_q.fit_transform(kernel_matrix_test) 
        LR = LogisticRegression()
        LR.fit(train_features_q, y_train)
        return LR, train_features_q, test_features_q

    def main_qpca_q_correlation(self,samples=500, start_at=0, end_at=20, results_file="results/qpca_qkernel.csv"):
        
        # **Check if the CSV already exists**
        file_exists = os.path.exists(results_file)

        for dimension in range(start_at, end_at):
            if self.metrics.should_skip_execution(results_file, samples, dimension) == False:
                X_train, X_test, y_train, y_test = self.kdd_data.dataset(dimension, samples)
                print("Shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

                start = time.time()
                model, X_train, X_test = self.qpca_precomputed_qkernel(X_train, y_train, X_test, y_test, dimension)

                # df_model = metrics_of_evaluation(dimension, model, time.time(), start, X_train, y_train)
                df_model = self.metrics.metrics_of_evaluation_classicaly(
                    model, dimension, time.time(), start, X_train, y_train
                )
                df_model["Samples"] = samples  # Add sample size info
                # **Append new results while keeping existing content**
                df_model.to_csv(results_file, mode='a', index=False, header=not file_exists)

                # **Ensure the header is written only once**
                file_exists = True  

        return True

    def qpca_rbf_qkernel(self,X_train, y_train, X_test, y_test, dimension):
        # sampler = Sampler(backend)
        
        feature_map = ZZFeatureMap(feature_dimension=dimension, reps=2)
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

        # Compute kernel matrix
        kernel_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
        kernel_matrix_test = quantum_kernel.evaluate(x_vec=X_test)

        kernelpca_q = KernelPCA(n_components=dimension, kernel='rbf')
        train_features_q = kernelpca_q.fit_transform(kernel_matrix_train)
        test_features_q = kernelpca_q.fit_transform(kernel_matrix_test) 
        LR = LogisticRegression()
        LR.fit(train_features_q, y_train)
        return LR, train_features_q, test_features_q

    def main_qpca_rbf_correlation(self, samples=500, start_at=0, end_at=20, results_file="results/qpca_qData.csv"):    
        # **Check if the CSV already exists**
        file_exists = os.path.exists(results_file)

        for dimension in range(start_at, end_at):
            if self.metrics.should_skip_execution(results_file, samples, dimension) == False:
                X_train, X_test, y_train, y_test = self.kdd_data.dataset(dimension, samples)
                print("Shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

                start = time.time()
                model, X_train, X_test = self.qpca_rbf_qkernel(X_train, y_train, X_test, y_test, dimension)

                # df_model = metrics_of_evaluation(dimension, model, time.time(), start, X_train, y_train)
                df_model = self.metrics.metrics_of_evaluation_classicaly(
                    model, dimension, time.time(), start, X_train, y_train
                )
                df_model["Samples"] = samples  # Add sample size info
                # **Append new results while keeping existing content**
                df_model.to_csv(results_file, mode='a', index=False, header=not file_exists)

                # **Ensure the header is written only once**
                file_exists = True  

        return True