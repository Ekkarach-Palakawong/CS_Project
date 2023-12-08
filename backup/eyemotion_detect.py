import math
import csv
import cv2 as cv
import numpy as np 
import mediapipe as mp
from scipy.fft import fft
from moviepy.editor import *

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ] 
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

RIGHT_IRIS=[474,475,476,477]
LEFT_IRIS=[469,470,471,472]
L_H_LEFT = [33]   #right eye right most landmark
L_H_RIGHT = [133] #right eye left most landmark
R_H_LEFT = [362]  #left eye right most landmark
R_H_RIGHT = [263] #left eye left most landmark

frame_counter = 0
fps = 0

def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_dist = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_dist
    iris_position = ""
    if ratio <= 0.42:
        iris_position = "right"
    elif ratio > 0.42 and ratio <= 0.56 :
        iris_position = "center"
    else:
        iris_position = "left"
    return iris_position, ratio

def iris_position_polar(iris_center, right_point, left_point):
    # Calculate the vectors from the iris center to the right and left points
    vector_right = right_point - iris_center
    vector_left = left_point - iris_center

    # Calculate the angles in radians using arctan2
    angle_right = np.arctan2(vector_right[1], vector_right[0])
    angle_left = np.arctan2(vector_left[1], vector_left[0])

    # Calculate the radius (distance) from the iris center to the points
    radius_right = np.linalg.norm(vector_right)
    radius_left = np.linalg.norm(vector_left)

    return angle_right, radius_right, angle_left, radius_left

# Load the video file
Video_path = r"C:\Users\pnaSu\Desktop\openCV_project\video_patient\right_BPPV\right1.mp4"
mp_face_mesh =  mp.solutions.face_mesh
camera = cv.VideoCapture(Video_path)
with mp_face_mesh.FaceMesh( 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5 
) as face_mesh: 
    try:
        file1 = open('right1_Lefteye.csv', 'w', newline = '' )
        file2 = open('right1_Righteye.csv', 'w', newline = '' )
        #file3 = open('totalframe.csv', 'w', newline = '' )
        writer1 = csv.writer(file1)
        writer2 = csv.writer(file2)
        #writer3 = csv.writer(file3)

        total_frames = int(camera.get(cv.CAP_PROP_FRAME_COUNT))
        fps = int(camera.get(cv.CAP_PROP_FPS))

        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv.flip(frame, 1) 
            frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
            rgb_frame = cv.cvtColor(frame, cv. COLOR_BGR2RGB) 
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            gray_roi = cv.GaussianBlur(rgb_frame, (7, 7), 0)
            _, threshold = cv.threshold(gray_roi, 5, 255, cv.THRESH_BINARY_INV)
            if results.multi_face_landmarks:
                mesh_points=np.array(
                    [
                        np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                        for p in results.multi_face_landmarks[0].landmark
                    ]
            )

                #cropeye
                cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0,255,0), 1, cv.LINE_AA)
                cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0,255,0), 1, cv.LINE_AA)
                
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype = np.int32)
                center_right = np.array([r_cx, r_cy], dtype = np.int32)

                #cropiris
                cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(l_radius), (255,0,255), 1, cv.LINE_AA)
                cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255,255,255), -1, cv.LINE_AA)
                cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0,255,255), -1, cv.LINE_AA)
                
                frame_counter += 1
                time_in_seconds = frame_counter / fps
                #print(time_in_seconds)

                righteye_iris_pos, right_ratio = iris_position(
                    center_right, mesh_points[R_H_RIGHT], 
                    mesh_points[R_H_LEFT][0]
                )

                lefteye_iris_pos, left_ratio = iris_position(
                    center_left, mesh_points[L_H_RIGHT], 
                    mesh_points[L_H_LEFT][0]
                )

                #iris position 
                #writer1.writerow(center_right)
                #writer2.writerow(center_left)

                right_eye_angle, right_eye_radius, left_eye_angle, left_eye_radius = iris_position_polar(
                    center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0]
                )
                
                right_eye_angle_deg = np.degrees(right_eye_angle) 
                left_eye_angle_deg = np.degrees(left_eye_angle)

                #iris position but polar
                writer1.writerow([left_eye_angle])
                writer2.writerow([right_eye_angle])  
                #print(right_eye_angle)            

                #print("Right Eye Iris Angle (degrees):", right_eye_angle_deg)
                #print("Right Eye Iris Radius:", right_eye_radius)
                #print("Left Eye Iris Angle (degrees):", left_eye_angle_deg)
                #print("Left Eye Iris Radius:", left_eye_radius)

                #print("Left Iris Position (x, y):", center_left)
                #print("Right Iris Position (x, y):", center_right)
                

                cv.putText(
                    frame, f"Left: {lefteye_iris_pos} {left_ratio:.2f}",
                    (30, 50), 
                    cv.FONT_HERSHEY_PLAIN, 
                    2, 
                    (0,255,0), 
                    3, 
                    cv.LINE_AA
                )

                cv.putText(
                    frame, f"Right: {righteye_iris_pos} {right_ratio:.2f}",
                    (30, 75), 
                    cv.FONT_HERSHEY_PLAIN, 
                    2, 
                    (0,255,0), 
                    3, 
                    cv.LINE_AA
                )
            #cv.imshow("Threshold", threshold)
            #cv.imshow("gray roi", gray_roi)
            cv.imshow('img', frame)
            if cv.waitKey(30) == ord('q'):
                break
        #writer3.writerow([total_frames,time_in_seconds,fps]) 
        #print(time_in_seconds) 
        #print(fps)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(e)
    else:
        #print(file.read())
        #file1.close()
        file2.close()
        #file3.close()

camera.release()
cv.destroyAllWindows()