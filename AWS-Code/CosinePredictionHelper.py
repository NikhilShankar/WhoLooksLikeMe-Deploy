import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from typing import Dict
from WLLMSimilarityCalculatorAdvancedCorrected import SimilarityCalculatorAdvancedCorrected2
from WLLMModelLoader import WLLMModelLoader
import random
import boto3

class CosinePredictionHelper:
    def __init__(self, models: Dict[str, str], N: int, image_dataset_s3_path: str):
        self.models = models
        self.N = N
        self.image_dataset_s3_path = image_dataset_s3_path
        self.s3_client = boto3.client('s3')
        self.loaded_models = {}
        for key, s3_model_path in models.items():
            keras_s3_file = f"{s3_model_path}/best_model.keras"
            self.loaded_models[key] = WLLMModelLoader(keras_s3_file, s3_client=self.s3_client).embedding_model

    def run_pipeline(self, test_image_s3_path: str, save_s3_path: str, plot=True):
        # Step 1: Calculate predictions
        predictions, times, total_time = self.calculate_predictions(test_image_s3_path)

        # Step 2: Create dataframe
        df = self.create_dataframe(predictions)

        # Step 3: Plot results
        if plot:
            results_plot_s3_path = f"{save_s3_path}/results_bar_graph.png"
            self.plot_results_avg(df, results_plot_s3_path)

            # Step 4: Plot times
            times_plot_s3_path = f"{save_s3_path}/times_bar_graph.png"
            self.plot_times(times, times_plot_s3_path)

            # Step 5: Save results
            results_file_s3_path = f"{save_s3_path}/results.csv"
            self.save_results(df, times, results_file_s3_path)

        top_avg_personalities, top_score_personalities = self.create_top_n_plot(df, self.N)
        return top_avg_personalities, top_score_personalities

    def calculate_predictions(self, test_image_s3_path: str):
        predictions = {}
        times = {}

        for model_name, s3_model_path in self.models.items():
            start_time = time.time()

            embedding_folder_s3_path = f"{s3_model_path}/embeddings"
            similarity_calculator = SimilarityCalculatorAdvancedCorrected2(
                embedding_folder_s3_path, s3_client=self.s3_client
            )

            pred_by_avg_N, pred_by_total_N = similarity_calculator.calculate_similarity(
                test_image_s3_path, self.loaded_models[model_name], 100
            )

            predictions[model_name] = {
                "pred_by_avg_N": pred_by_avg_N,
                "pred_by_total_N": pred_by_total_N,
            }
            times[model_name] = time.time() - start_time

        total_time = sum(times.values())
        return predictions, times, total_time

    def create_dataframe(self, predictions):
        data = defaultdict(list)

        for model_name, preds in predictions.items():
            for index, personality in enumerate(preds["pred_by_avg_N"]):
                data["personality"].append(personality[0])
                data["model_name"].append(model_name)
                data["pred_avg"].append(personality[1])
                data["pred_score"].append(preds["pred_by_total_N"][index][1])

        df = pd.DataFrame(data)
        df = df.pivot(index="model_name", columns="personality", values=["pred_avg", "pred_score"])
        df = df.sort_index(axis=1, level=1)

        return df

    def plot_results_avg(self, df, save_path: str):
        avg_df = df["pred_avg"]
        df_copy = avg_df.loc[:, avg_df.mean(axis=0).nlargest(self.N).index]
        df_copy.plot(kind="bar", figsize=(20, 8), legend=True)
        plt.title("Prediction Results based on Average")
        plt.ylabel("Scores")
        plt.xlabel("Models")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_path)

    def plot_times(self, times, save_path: str):
        plt.figure(figsize=(6, 4))
        plt.bar(times.keys(), times.values(), color='skyblue')
        plt.title("Time Taken Per Model")
        plt.ylabel("Time (seconds)")
        plt.xlabel("Models")
        plt.tight_layout()
        plt.savefig(save_path)

    def save_results(self, df, times, file_path: str):
        df.to_csv(file_path)
        time_df = pd.DataFrame(times.items(), columns=["Model", "Time"])
        time_df.to_csv(file_path.replace(".csv", "_times.csv"), index=False)

    def create_top_n_plot(self, df, top_n: int):
        top_avg = df["pred_avg"].mean(axis=0).nlargest(top_n)
        top_score = df["pred_score"].mean(axis=0).nlargest(top_n)
        return top_avg, top_score
