import cv2 as cv

# Replace "your_video_file_path.mp4" with the actual file path of your video
video_path = (r"C:\มอกะเสด\มหาลัย\ปี4\project_End\video_patient\test2.mp4")

# Create a VideoCapture object to open the video file
cap = cv.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break  # If the frame couldn't be read, the end of the video has been reached

        # Display the frame for ROI selection
        cv.imshow('Select ROI', frame)

        # Allow the user to select the ROI using the mouse
        roi = cv.selectROI('Select ROI', frame, fromCenter=False, showCrosshair=True)

        # Extract the coordinates of the ROI
        x, y, w, h = roi
        print(x, y, w, h)
        roi_coordinates = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        # Print the ROI coordinates
        print("ROI Coordinates:")
        for coord in roi_coordinates:
            print(coord)

        # Draw a rectangle around the selected ROI on the frame
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with the selected ROI
        #cv.imshow('Selected ROI', frame)

        # Wait for a small amount of time (e.g., 25 milliseconds) and check for key presses
        key = cv.waitKey(25)

        # If the 'q' key is pressed, exit the loop and stop displaying frames
        if key == ord('q'):
            break

# Release the VideoCapture and close any open windows
cap.release()
cv.destroyAllWindows()