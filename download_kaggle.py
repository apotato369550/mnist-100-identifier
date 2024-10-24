import kaggle

kaggle.api.authenticate()

# Download latest version
kaggle.api.dataset_download_files("martininf1n1ty/mnist100", path='.', unzip=True)