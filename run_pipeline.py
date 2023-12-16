from pipelines.training_pipeline import  train_pipeline
from zenml.client import Client
if __name__ == "__main__":
    #Run the Pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/home/abhishek13/Sales Forcasting/Sales-forcasting-using-mlops/data/Walmart Data Analysis and Forcasting.csv")

#    mlflow ui --backend-store-uri "file:/home/abhishek13/.config/zenml/local_stores/9533138d-41b9-4ede-b8b0-828a8b80a2d8/mlruns"