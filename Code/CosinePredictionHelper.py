import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from WLLMSimilarityCalculatorAdvancedCorrected import SimilarityCalculatorAdvancedCorrected2
from WLLMModelLoader import WLLMModelLoader
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random


class CosinePredictionHelper:
    def __init__(self, models: Dict[str, str], N: int, image_dataset_path: str):
        self.models = models
        self.N = N
        self.image_dataset_path = image_dataset_path
        self.loaded_models = {}
        for key, value in models.items():
            keras_file = os.path.join(value, "best_model.keras")
            self.loaded_models[key] = WLLMModelLoader(keras_file).embedding_model



    def run_pipeline(self, test_image_path: str, save_path: str, plot=True):
        # Step 1: Calculate predictions
        predictions, times, total_time = self.calculate_predictions(test_image_path)
        # Step 2: Create dataframe
        df = self.create_dataframe(predictions)
        # Step 3: Plot results
        if plot:
            results_plot_path = os.path.join(save_path, "results_bar_graph.png")
            self.plot_results_avg(df, results_plot_path)

            # Step 4: Plot times
            times_plot_path = os.path.join(save_path, "times_bar_graph.png")
            self.plot_times(times, times_plot_path)

            # Step 5: Save results
            results_file_path = os.path.join(save_path, "results.csv")
            self.save_results(df, times, results_file_path)

        top_n_plot_path = os.path.join(save_path, "top_n_plot.png")
        top_avg_personalities, top_score_personalities = self.create_top_n_plot(df, self.N, top_n_plot_path, plot)
        if plot:
            # Step 6: Create and save top N plot
            self.plot_personality_images(self.image_dataset_path, test_image_path, top_avg_personalities, N=self.N)
            self.plot_personality_images(self.image_dataset_path, test_image_path, top_score_personalities, N=self.N)
        return top_avg_personalities, top_score_personalities

    def calculate_predictions(self, test_image_path: str):
        predictions = {}
        times = {}

        for model_name, model_path in self.models.items():
            start_time = time.time()

            
            embedding_folder = os.path.join(model_path, "embeddings")

            # Assuming `SimilarityCalculatorAdvancedCorrected2` is defined elsewhere
            similarity_calculator = SimilarityCalculatorAdvancedCorrected2(embedding_folder)

            pred_by_avg_N, pred_by_total_N = similarity_calculator.calculate_similarity(test_image_path, self.loaded_models[model_name], 100)

            predictions[model_name] = {
                "pred_by_avg_N": pred_by_avg_N,
                "pred_by_total_N": pred_by_total_N
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

    @staticmethod
    def load_results(file_path: str):
        df = pd.read_csv(file_path, index_col=[0], header=[0, 1])
        time_df = pd.read_csv(file_path.replace(".csv", "_times.csv"))
        return df, time_df

    @staticmethod
    def display_results(file_path: str):
        df, time_df = CosinePredictionHelper.load_results(file_path)

        df.plot(kind="bar", figsize=(15, 8))
        plt.title("Prediction Results")
        plt.ylabel("Scores")
        plt.xlabel("Models")
        plt.tight_layout()
        plt.show()

        time_df.plot(x="Model", y="Time", kind="bar", figsize=(10, 6))
        plt.title("Time Taken Per Model")
        plt.ylabel("Time (seconds)")
        plt.xlabel("Models")
        plt.tight_layout()
        plt.show()

    def create_top_n_plot(self, df, top_n: int, save_path: str, plot=True):
        top_avg = df["pred_avg"].mean(axis=0).nlargest(top_n)
        top_score = df["pred_score"].mean(axis=0).nlargest(top_n)
        if plot:
            top_avg_personalities = top_avg.index
            top_score_personalities = top_score.index
            plt.figure(figsize=(15, 10))
            # Dummy plotting logic for simplicity; Replace with actual image plotting
            for i, personality in enumerate(top_avg_personalities):
                plt.text(i, top_avg[personality], personality, ha="center", va="bottom")
            plt.savefig(save_path)
            plt.close()

        return top_avg, top_score


    def plot_personality_images(self, dataset_path: str, test_image_path, top_avg, N: int):
        personality_names = top_avg.index
        fig, axes = plt.subplots(1, N+1, figsize=(N * 3, 3))
        img = plt.imread(test_image_path)
        axes[0].axis("off")
        axes[0].imshow(img)
        axes[0].set_title("TEST IMAGE")
        for i in range(N):
            if i >= len(personality_names):
                break

            personality = personality_names[i]
            folder_path = os.path.join(dataset_path, personality)
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                print(f"No images found for personality: {personality}")
                continue

            # Load the first image
            image_path = os.path.join(folder_path, random.choice(image_files))
            img = plt.imread(image_path)

            # Plot image
            axes[i+1].imshow(img)
            axes[i+1].axis("off")
            axes[i+1].set_title(f"{personality} - Rank - {i+1} ")
        

        plt.title("Rankings based on Average")
        plt.tight_layout()
        plt.show()

    def plot_personality_images(self, dataset_path: str, test_image_path, top_score, N: int):
        personality_names = top_score.index
        fig, axes = plt.subplots(1, N+1, figsize=(N * 3, 3))
        img = plt.imread(test_image_path)
        axes[0].axis("off")
        axes[0].imshow(img)
        axes[0].set_title("TEST IMAGE")
        for i in range(N):
            if i >= len(personality_names):
                break

            personality = personality_names[i]
            folder_path = os.path.join(dataset_path, personality)
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                print(f"No images found for personality: {personality}")
                continue

            # Load the first image
            image_path = os.path.join(folder_path, random.choice(image_files))
            img = plt.imread(image_path)

            # Plot image
            axes[i+1].imshow(img)
            axes[i+1].axis("off")
            axes[i+1].set_title(f"{personality} - Rank - {i+1} ")

        plt.title("Rankings based on Score")
        plt.tight_layout()
        plt.show()