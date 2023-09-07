#week3
import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np

frame_counter = 0
FONTS = cv.FONT_HERSHEY_COMPLEX

# Left eyes indices 
LEFT_EYE = [ 
    362, 382, 381, 380, 
    374, 373, 390, 249, 
    263, 466, 388, 387, 
    386, 385,384, 398 
]
# right eyes indices
RIGHT_EYE = [ 
    33, 7, 163, 144, 
    145, 153, 154, 155, 
    133, 173, 157, 158, 
    159, 160, 161 , 246
]
# iris landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]  

# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord
'''def landmarksDetection3D(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coords_3d = [(int(point.x * img_width), int(point.y * img_height), point.z) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1) for p in mesh_coords_3d]

    return mesh_coords_3d'''

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    # cv.imshow('mask', mask)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left

def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # create fixd part for eye with 
    piece = int(w/3) 

    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color

def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye =''
    if max_index==0:
        pos_eye="RIGHT"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.MAGENTA]
    return pos_eye, color

# Function to calculate the center of the iris
'''def calculate_iris_center(iris_landmarks):
    x_sum = sum(landmark[0] for landmark in iris_landmarks)
    y_sum = sum(landmark[1] for landmark in iris_landmarks)
    center_x = x_sum / len(iris_landmarks)
    center_y = y_sum / len(iris_landmarks)
    return (center_x, center_y)
................an so on.........................................
            # Calculate the center of the iris for the right and left eye
            #iris_landmarks_right = [mesh_coords[p] for p in RIGHT_EYE]
            iris_center_right = calculate_iris_center(right_coords)
            #iris_landmarks_left = [mesh_coords[p] for p in LEFT_EYE]
            iris_center_left = calculate_iris_center(left_coords)

            # Calculate the distance between the center and the tail of the iris for the right eye
            iris_tail_landmark_right = right_coords[0]  # Assuming index 8 represents the tail of the iris 0
            distance_to_iris_tail_right = euclidean_distance(iris_center_right, (iris_tail_landmark_right.x, iris_tail_landmark_right.y))
                        
            iris_tail_landmark_left = left_coords[8]  # Assuming index 8 represents the tail of the iris 8
            distance_to_iris_tail_left = euclidean_distance(iris_center_left, (iris_tail_landmark_left.x, iris_tail_landmark_left.y))'''

#for input video if dont use comment it.
Video_path = r"C:\มอกะเสด\มหาลัย\ปี4\project_End\video_patient\left_BPPV\YTD.mp4"

camera = cv.VideoCapture(Video_path)
map_face_mesh = mp.solutions.face_mesh

with map_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    start_time = time.time()
    while True:
        frame_counter += 10  # frame counter
        ret, frame = camera.read()  # getting frame from camera
        if not ret:
            break  # no more frames break

        frame = cv.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:        
            '''distance of the particular landmark(nose maybe?)
            print(results.multi_face_landmarks[0].landmark)'''
            mesh_coords = landmarksDetection(frame, results, False)
            print(len(mesh_coords))
            right_iris_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_iris_coords = [mesh_coords[p] for p in LEFT_EYE]
            print("Right Iris 3D Coordinates:", len(right_iris_coords))
            print("Left Iris 3D Coordinates:", len(left_iris_coords))
            
            crop_right, crop_left = eyesExtractor(frame, right_iris_coords, left_iris_coords)

            cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

            

            eye_position_right, color = positionEstimator(crop_right)
            utils.colorBackgroundText(frame, f'R: {eye_position_right}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
            eye_position_left, color = positionEstimator(crop_left)
            utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
            

            '''iris_center_right = np.mean(right_coords, axis=0)  # Calculate the mean of the iris landmarks
            iris_tail_landmark_right = right_coords[0]  # Assuming index 0 represents the tail of the iris
            distance_to_iris_tail_right = euclidean_distance(iris_center_right, iris_tail_landmark_right)

            iris_center_left = np.mean(left_coords, axis=0)  # Calculate the mean of the iris landmarks
            iris_tail_landmark_left = left_coords[8]  # Assuming index 8 represents the tail of the iris
            distance_to_iris_tail_left = euclidean_distance(iris_center_left, iris_tail_landmark_left)

            print("Distance to right iris tail:", distance_to_iris_tail_right)
            print("Distance to left iris tail:", distance_to_iris_tail_left)'''

            #print(left_coords)

        end_time = time.time() - start_time
        fps = frame_counter / end_time

        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        # writing image for thumbnail drawing shape
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        cv.imshow('frame', frame)
        key = cv.waitKey(30)
        if key == ord('q') or key == ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release