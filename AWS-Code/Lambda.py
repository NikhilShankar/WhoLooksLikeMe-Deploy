import boto3
import os
import tempfile
from cosine_prediction_helper import CosinePredictionHelper

# Initialize S3 client
s3_client = boto3.client('s3')

# Define the S3 bucket name
BUCKET_NAME = "your-s3-bucket-name"

# Model map with S3 paths
modelmap = {
    "ModelA": "s3://your-s3-bucket-name/FinalizedModelsForAWS/WLLM-Model-0001",
    "ModelB": "s3://your-s3-bucket-name/FinalizedModelsForAWS/WLLM-Model-0002",
    "ModelC": "s3://your-s3-bucket-name/FinalizedModelsForAWS/WLLM-Model-0003"
}

# Dataset and predictions path on S3
image_dataset_path = "s3://your-s3-bucket-name/dataset"
predictions_save_path = "s3://your-s3-bucket-name/predictions"

# Initialize the CosinePredictionHelper
combinedCosinePredictor = CosinePredictionHelper(models=modelmap, N=5, image_dataset_path=image_dataset_path)

def lambda_handler(event, context):
    """
    Lambda function to accept an image, save it to S3, and predict using the model.
    """
    # Read the image from the event
    image_content = event['body']  # Assumes the image is sent as base64 in the body of the request

    # Create a temporary file to save the image locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_content)
        local_image_path = temp_file.name

    # Define the temporary S3 path
    image_s3_key = "temp/" + os.path.basename(local_image_path)
    image_s3_path = f"s3://{BUCKET_NAME}/{image_s3_key}"

    # Upload the image to S3
    s3_client.upload_file(local_image_path, BUCKET_NAME, image_s3_key)

    # Run the pipeline using the S3 image path
    top_average, top_score = combinedCosinePredictor.run_pipeline(image_s3_path, predictions_save_path, plot=False)

    # Clean up the temporary local file
    os.remove(local_image_path)

    # Return the result
    return {
        "statusCode": 200,
        "body": {
            "top_average": top_average,
            "top_score": top_score
        }
    }
