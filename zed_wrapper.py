import pyzed.sl as sl
import cv2
import copy
import statistics

class Zed:

    """
        discord: @kialli, @seaniiiiii 
        github: @kchan5071, @Gabriel-Sean13

        Wrapper class for ZED camera.

        Usage:
        - Initialize
        - Open camera
        - Then you can:
            * get color image
            * get IMU data
            * get Euclidean distance image
            * get Euclidean distance at a point
            * get median Euclidean distance over a rectangle
    """

    def __init__(self):
 
        #Creates a sl.Camera object to handle ZED
        self.zed = sl.Camera()
        #Creates a configuration object on how we want to set up the camera we open
        self.init_params = sl.InitParameters()
        #Set the camera resolution to 1280x720(HD)
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        #Ask the Camera to run at 60fps might change given computation expenses
        self.init_params.camera_fps = 60
        #Use coordinate system using the right hand rule where the Y points up
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        #Chooses a nueral network for a depth estimation algo
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        #Creates parameters for positional tracking using IMU
        self.tracking_parameters = sl.PositionalTrackingParameters()
        #Tell the SDK to fuse IMU data with video tracking for better estimation
        self.tracking_parameters.enable_imu_fusion = True
        #Control runtime behavioir of grab()
        self.runtime_parameters = sl.RuntimeParameters()
        #Set unit of measurement in Meters
        self.init_params.coordinate_units = sl.UNIT.METER
        #Ignore depth distance closer than 1 merter
        self.init_params.depth_minimum_distance = 1.0
        #Translation object but not used in this file, idk why kai has this 
        self.py_translation = sl.Translation()
    

    #Function to open the camera
    def open(self):
  
        """
            Open the ZED camera and return the state.
            return:
                state: sl.ERROR_CODE.SUCCESS is expected.
        """
        state = self.zed.open(self.init_params)
        return state


    #Function on getting a Color Image
    def get_image(self):

        """
            Get color image from ZED camera.
            return:
                image: np_array (H, W, 4 or 3 channels depending on SDK)

            We deepcopy to avoid Python GC issues on the underlying buffer.
        """
        #Zeds internal image/measuring container
        image_zed = sl.Mat()
        try:
        
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                #Fills the image_zed container with the left cameras RGB(A) image
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                #Cretes a full independent copy in Python memory of a numpy array(image_zed)
                return copy.deepcopy(image_zed.get_data())
        except RuntimeError:
            print(RuntimeError)
            pass

    #This whole function is not needed or used in vision_main.py or YoloDetection.py idk why Kai had this
    #Grabbing IMU Data
    def get_imu(self):
        """
            Get IMU data from ZED.
            return:
                quaternion: sl.float4 (orientation)
                linear_acceleration: sl.float3
                angular_velocity: sl.float3
        """
        #Allocates a container object that ZED will fill with IMU data
        sensors_data = sl.SensorsData()
        #Check if a new frame is grabbed succesfully
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            #Fill the container sensor_data 
            self.zed.get_sensors_data(sensors_data,sl.TIME_REFERENCE.CURRENT)
            #get and stor pose and orientation as quaternion
            quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
            #Grab accelarion and gyro reading from IMU
            linear_acceleration = sensors_data.get_imu_data().get_linear_acceleration()
            angular_velocity = sensors_data.get_imu_data().get_angular_velocity()

            return quaternion, linear_acceleration, angular_velocity
            
        #If Grab fails return nothing
        return None,None,None
    
    #Get Image Distance
    def get_distance_image(self):
        """
            Gets Euclidean distance image from ZED.

            return:
                image: np_array (distance in meters per pixel)

            Uses MEASURE.DISTANCE which is the straight-line distance
            from the camera center to each 3D point.
        """
        #Create a container to hold distance values
        distance_mat = sl.Mat()
        #Grab a new frame to compute distance
        #If frame grab is a success
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            #Fills distance_mat with the Euclidean distance (in meters)
            self.zed.retrieve_measure(distance_mat, sl.MEASURE.DISTANCE)
            #Numpy array view of the distance matrix
            image = distance_mat.get_data()
            #clone it 
            return copy.deepcopy(image)
         

    #Get distnace form a single pixel
    def get_distance_at_point(self,x,y):
        """
            Gets Euclidean distance at a specific pixel (x, y).

            return:
                distance: float (meters)
                -1 if out of bounds or if grab failed.
        """
        #Ask Zed for camera info
        cam_info = self.zed.get_camera_information()
        width = cam_info.camera_configuration.resolution.width
        height = cam_info.camera_configuration.resolution.height

        #Bounds check ( indices go from 0 to width-1 / height-1 )
        #Prevents indexing outside the image
        if x < 0 or y < 0 or x >= width or y >= height:
            #Means its a bad distance
            return -1.0
         
        #Try to grab a new frame
        if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
            return -1.0
         
        #create distance mat and retreive Euclidean distance
        distance_mat = sl.Mat(width,height,sl.MAT_TYPE.F32_C1)
        #Fills distance_mat with Eclidean distance values for this current frame
        self.zed.retrieve_measure(distance_mat,sl.MEASURE.DISTANCE)

        status,distance = distance_mat.get_value(int(x),int(y))
        if status != sl.ERROR_CODE.SUCCESS:
            return -1.0
         
        return float(distance)
    
    #Get the mdeian distance sample points
    def get_median_distance(self, x1, y1, x2, y2):
        """
            Gets 5 Euclidean distance sample points in the rectangle and returns the median.

            Sample pattern (X = sampled):

            -----------------------
            |                     |
            |          X          |
            |                     |
            |    X     X     X    |
            |                     |
            |          X          |
            |                     |
            -----------------------

            We use MEASURE.DISTANCE so it's Euclidean distance from camera
            to each sampled 3D point, in meters.

            return:
                median: float (meters), or -1.0 on failure.
        """ 
        #Same as before to read camera resolution to validate coordinates
        cam_info = self.zed.get_camera_information()
        width = cam_info.camera_configuration.resolution.width
        height = cam_info.camera_configuration.resolution.height

        #Ensure all four corners are withing the images bounds

        #validate rectangle coordinates
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return -1.0
        if x1 >= width or x2 >= width or y1 >= height or y2 >= height:
            return -1.0
         
        #Grabe a new frame
        if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
            return -1.0
         
        # Create distance mat and retrieve Euclidean distance
        distance_mat = sl.Mat(width, height, sl.MAT_TYPE.F32_C1)
        self.zed.retrieve_measure(distance_mat, sl.MEASURE.DISTANCE)

        # Take 5 sample points and compute median
        depth = [None] * 5

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Center
        _, depth[0] = distance_mat.get_value(int(cx), int(cy))
        # Left-middle
        _, depth[1] = distance_mat.get_value(int((x1 + cx) // 2), int(cy))
        # Top-middle
        _, depth[2] = distance_mat.get_value(int(cx), int((y1 + cy) // 2))
        # Right-middle
        _, depth[3] = distance_mat.get_value(int((cx + x2) // 2), int(cy))
        # Bottom-middle
        _, depth[4] = distance_mat.get_value(int(cx), int((cy + y2) // 2))


        # Filter out invalid values (NaN, inf, <= 0)
        valid_depths = [
            float(d) for d in depth
            if d is not None and d > 0 and d != float('inf')
        ]

        if not valid_depths:
            return -1.0

        median = statistics.median(valid_depths)
        return median


#Run Function to test if Zed opens 
if __name__ == '__main__':
    zed = Zed()
    state = zed.open()
    if state != sl.ERROR_CODE.SUCCESS:
        print("Could not open ZED:", state)
    else:
        while True:
            image = zed.get_image()
            if image is not None:
                cv2.imshow("image_test", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        zed.zed.close()
