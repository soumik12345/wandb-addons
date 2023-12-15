sudp apt-get update
sudo apt-get install ffmpeg libsm6 libxext6 -y
sudo pip install --upgrade pip
sudo pip install -e ".[dev, docs]"
sudo pre-commit install