# eyeblinks_removal

1. Install Required Libraries:

Ensure you have the necessary Python libraries installed. You can install them using pip if you haven't already:

> pip install opencv-python imutils numpy scipy dlib


2. Download a Facial Landmark Predictor:

You need a pre-trained facial landmark predictor file (a .dat file) for dlib. You can find one at the following URL: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Download the file and extract it to a location on your computer.

3. Prepare an Input Video:

Prepare an input video on which you want to perform eye blink detection. Make sure you know the path to this video.

4. Run the Program:

Open a terminal or command prompt and navigate to the directory where you saved the Python script (the program). Run the program by executing the following command:

> python script_name.py --shape-predictor path_to_shape_predictor_file.dat --video path_to_input_video.mp4

Replace script_name.py with the name of the Python script containing the code, path_to_shape_predictor_file.dat with the actual path to the facial landmark predictor file you downloaded in step 2, and path_to_input_video.mp4 with the path to your input video.

5. Interact with the Program:

Once you run the program, it will open a window displaying the input video with overlays showing the blink counter and eye aspect ratio (EAR). You can interact with the program as follows:

Press 'q' in the window to close the video and exit the program.
After the program completes, it will print information about the blink count and the duration of the video.

6. Output:

The program will create an output video file named unblink_output_video.mp4. This video will contain the frames where blinks are removed, effectively creating a blink-free version of the input video.
