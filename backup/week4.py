import cv2 as cv
import numpy as np 
import mediapipe as mp
import math
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ] 
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

RIGHT_IRIS=[474,475,476,477]
LEFT_IRIS=[469,470,471,472]
L_H_LEFT = [33]   
L_H_RIGHT = [133] 
R_H_LEFT = [362]  
R_H_RIGHT = [263] 


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
    elif ratio > 0.42 and ratio <= 0.57 :
        iris_position = "center"
    else:
        iris_position = "left"
    return iris_position, ratio

def track_iris_optical_flow(prev_frame_gray, curr_frame_gray, prev_centers, prev_irises):
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    new_centers = []
    for i, (prev_center, prev_iris) in enumerate(zip(prev_centers, prev_irises)):
        prev_iris = prev_iris.astype(np.float32)
        new_corner, status, _ = cv.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, prev_iris, None, **lk_params)
        if status is not None and status[0] == 1:
            new_center = new_corner.ravel()
            new_centers.append(new_center)

    return new_centers

prev_frame = None
prev_landmarks = None
prev_center = None
prev_irises = None  
new_centers = []

motion_display = np.zeros((480, 640, 3), dtype=np.uint8)

mp_face_mesh =  mp.solutions.face_mesh
cap = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh( 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5 
) as face_mesh: 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        rgb_frame = cv.cvtColor(frame, cv. COLOR_BGR2RGB) 
        img_h, img_w = frame.shape[:2] 
        results = face_mesh.process(rgb_frame) 
        if results.multi_face_landmarks: 
            mesh_points=np.array([
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                    for p in results.multi_face_landmarks[0].landmark
                ])
            
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype = np.int32)
            center_right = np.array([r_cx, r_cy], dtype = np.int32)
            cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255,255,255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0,255,255), -1, cv.LINE_AA)


            if prev_frame is not None and prev_landmarks is not None:
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                prev_gray_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
                prev_irises = [mesh_points[RIGHT_IRIS].astype(np.float32), mesh_points[LEFT_IRIS].astype(np.float32)]

                new_centers = track_iris_optical_flow(
                    prev_gray_frame,
                    gray_frame,
                    prev_centers,
                    prev_irises,
            )
            
            prev_frame = frame.copy()
            prev_landmarks = mesh_points
            prev_centers = [center_left, center_right]
            
            righteye_iris_pos, right_ratio = iris_position(
                center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0]
            )
            lefteye_iris_pos, left_ratio = iris_position(
                center_left, mesh_points[L_H_RIGHT], mesh_points[L_H_LEFT][0]
            )
            
            if len(new_centers) == 2:
                prev_x_left = prev_centers[0][0]
                curr_x_left = new_centers[0][0]
                if curr_x_left > prev_x_left:
                    print("Torsional Left")
                else:
                    print("Torsional Right")

            cv.putText(frame, f"Left Iris pos: {lefteye_iris_pos} {left_ratio:.2f}",
                (30, 50), cv.FONT_HERSHEY_PLAIN,2,(0,255,0),3,cv.LINE_AA)

            cv.putText(frame, f"Right Iris pos: {righteye_iris_pos} {right_ratio:.2f}",
                (30, 30), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3, cv.LINE_AA)
            
        #cv.imshow('Iris Motion Torsional', motion_display)
        cv.imshow('img', frame)
        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()