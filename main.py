from ultralytics import YOLO
from zed_camera import Zed


# The goal of this main.py should be to return the following:
#   1. After checking that the shark is there for 3+ frames, we send the confidence level in index 1 of a tuple
#   a. note: using Adjusted-minimal confidence we'll decide what counts as confident enough, i.e. if darker waters produce less confident numbers than when detecting darker waters be okay with lower confidence levels like 0.4 despite typically seeing 0.5 for example.
#   2. Return the distance from the object to the ZED camera, this should be calculated multiple times, if 1 calculation is very different from the others, make the model calculate another one and compare (scrap outliers)
#   3. Return the positioning of the object, if it's to the left/right then move the sub accordingly, if you need to rotate, send the YAW information to the DVL


def main():
     
     best = r"v1 (1).pt"
     zed_cam = Zed(best)
     zed_cam.run()


if __name__ == '__main__':
     main()