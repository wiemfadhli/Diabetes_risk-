from mlflow.tracking import MlflowClient

client = MlflowClient()

# List all registered models using search_registered_models
models = client.search_registered_models()

for model in models:
    print(model.name)