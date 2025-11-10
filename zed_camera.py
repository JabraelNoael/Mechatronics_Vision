 #Zed Camera
import cv2 
from ultralytics import YOLO


class Zed:

    def __init__(self, version: str):

        self.model = YOLO(version)
    

    def run(self) -> None:

        zed = cv2.VideoCapture(1)

        while True:

            frame_grabbed, frame  = zed.read()
            if not frame_grabbed:
                break

            #Splits Stereo Frames
            h, w, _ = frame.shape
            left = frame[:,:w//2]

            #Right side if preferred(Switch param left )
            right = frame[:,w//2:]
    
            results = self.model(left,verbose = False)
            annotated = results[0].plot()

            cv2.imshow("Zed left Camera",annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        zed.release()
        cv2.destroyAllWindows()
        print('Model Finished Running')