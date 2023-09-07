import cv2 as cv

# Load the image
video_path = (r"C:\มอกะเสด\มหาลัย\ปี4\project_End\video_patient\temp.mp4")
cap=cv.VideoCapture(video_path)

# Display the image and allow the user to select the ROI using the mouse
roi = cv.selectROI('Select ROI', img, fromCenter=False, showCrosshair=True)

# Extract the coordinates of the ROI
x, y, w, h = roi
roi_coordinates = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

# Print the ROI coordinates
print("ROI Coordinates:")
for coord in roi_coordinates:
    print(coord)

# Draw a rectangle around the selected ROI
cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the selected ROI
cv.imshow('Selected ROI', img)
cv.waitKey(0)
cv.destroyAllWindows()