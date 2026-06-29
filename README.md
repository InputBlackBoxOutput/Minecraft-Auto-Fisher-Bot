# Minecraft Auto Fisher Bot 🐟🐠🐡🦈🎣
#### An over-engineered bot using computer vision and deep learning for automatic fishing in Minecraft

![Demo GIF](demo.gif)

## How does this tool work?

1. Capture screenshot and then crop it to the region of interest
1. Pass the cropped screenshot to a Convolutional Neural Network to obtain a prediction
1. Use the prediction and a timing constraint to determine when to reel
1. If it is time to reel, reel in the find, throw the bait and repeat

## Development setup
```
# GUI and bot 
python -m venv env
source env/Scripts/activate
pip install -r requirements.txt

python gui.py 
```
```
# Train and test model
cd model

python -m venv env
source env/Scripts/activate
pip install -r requirements.txt

python train.py
```
