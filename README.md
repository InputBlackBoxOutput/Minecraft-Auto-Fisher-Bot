# Minecraft Auto Fisher Bot 🐟🐠🐡🦈🎣
#### An over-engineered bot using computer vision and deep learning for automatic fishing in Minecraft

## How to use?
1. Start Minecraft and align the window to your left split screen 
1. Enter a world, finding a place to fish and stand at the edge of the water body.
1. Craft or use commands to give yourself a fishing rod and right click to start fishing
1. Place your mouse about 15-20 pixels above the bobber and pause the game
1. Start the program and switch to minecraft before the 3 second timer runs out

## How it works?

1. Capture screenshot and then crop it to the region of interest
1. Pass the cropped screenshot to a Convolutional Neural Network to obtain a prediction
1. Use the prediction and a timing constraint to determine when to reel
1. If it is time to reel, reel in the find, throw the bait and repeat

### Made with lots of ⏱️, 📚 and ☕ by InputBlackBoxOutput