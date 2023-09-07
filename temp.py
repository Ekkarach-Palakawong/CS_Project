import cv2 as cv
import numpy as np 
import mediapipe as mp 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ] 
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_IRIS=[474,475,476,477]
RIGHT_IRIS=[469,470,471,472]
# Load the video file
Video_path = r"C:\Users\pnaSu\Desktop\openCV_project\myVideo.mp4"
mp_face_mesh =  mp.solutions.face_mesh
cap = cv.VideoCapture(Video_path)

with mp_face_mesh.FaceMesh( 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5 
) as face_mesh: 

# Create the Haar cascade for eye detection
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    while True:
        # Read the frame
        ret, frame = cap.read()
        
        if not ret:
            break
        frame = cv.flip(frame, 1) 
        rgb_frame = cv.cvtColor(frame, cv. COLOR_BGR2RGB) 
        img_h, img_w = frame.shape[:2] 
        results = face_mesh.process(rgb_frame) 
        if results.multi_face_landmarks:
            #print(results.multi_face_landmarks)
            #print(results.multi_face_landmarks[0].landmark) 
            #[print(p.x,p.y) for p in results.multi_face_landmarks[0].landmark]
            mesh_points=np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )
            #print(mesh_points)
            #polylines track
            cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            
            #OpenCv circle track
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            #print(l_cx, l_cy, r_cx, r_cy, l_radius)
            center_left = np.array([l_cx, l_cy], dtype = np.int32)
            center_right = np.array([r_cx, r_cy], dtype = np.int32)
            #print(center_left , center_right)
            cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)

        cv.imshow('img', frame)
        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()