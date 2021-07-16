import mediapipe as mp
import cv2
import time 

mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

previous_time = 0
current_time = 0 

#webcam feed
capture = cv2.VideoCapture(1)
#Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while capture.isOpened():
        ret, frame = capture.read()

        #recolor feed
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #make detections
        results = holistic.process(img)
        #print(results.face_landmarks)

        #face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        #recolor image back to BGR for rendering
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        #1 draw face landmarks
        mp_draw.draw_landmarks(img, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
        mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

        #2 right hand
        mp_draw.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        mp_draw.DrawingSpec(color=(255,150,0), thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(235,150,0), thickness=2, circle_radius=2))

        #3 left hand
        mp_draw.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(255,150,0), thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(235,150,0), thickness=2, circle_radius=2))
        
        #4 pose detection
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_draw.DrawingSpec(color=(100,50,0), thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(100,50,0), thickness=2, circle_radius=2))

        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time
        
        
        cv2.putText(img, str(int(fps)), (10,20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.475,
        (100, 255, 170), 2)

        cv2.imshow('Holistic Model Detection', img)

       
        if cv2.waitKey(10) and 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()