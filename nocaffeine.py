import cv2
import mediapipe as mp
import time
import math

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


start_time = time.time()
coffee_count = 0


is_sipping = False


with mp_holistic.Holistic(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't grab frame")
            break


        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        height, width, _ = image.shape

        hand_near_mouth = False 

        
        
        if results.face_landmarks:
            if results.face_landmarks:
                mouth_x = (results.face_landmarks.landmark[13].x +
                           results.face_landmarks.landmark[14].x) / 2 * width
                mouth_y = (results.face_landmarks.landmark[13].y +
                           results.face_landmarks.landmark[14].y) / 2 * height


                cv2.circle(image, (int(mouth_x), int(mouth_y)), 5, (0, 255, 0), -1)
            else:
                mouth_x, mouth_y = None, None
        else:
            mouth_x, mouth_y = None, None

        

        if mouth_x is not None and mouth_y is not None:

            if results.left_hand_landmarks:
                left_index = results.left_hand_landmarks.landmark[12]
                left_x = int(left_index.x * width)
                left_y = int(left_index.y * height)

                cv2.circle(image, (left_x, left_y), 5, (255, 0, 0), -1)

                distance = math.dist([left_x, left_y], [mouth_x, mouth_y])

                if distance < width * 0.05:
                    hand_near_mouth = True


            if results.right_hand_landmarks:
                right_index = results.right_hand_landmarks.landmark[12]
                right_x = int(right_index.x * width)
                right_y = int(right_index.y * height)
               
                cv2.circle(image, (right_x, right_y), 5, (255, 0, 0), -1)
               
                distance = math.hypot(right_x - mouth_x, right_y - mouth_y)
                
                if distance < 55:
                    hand_near_mouth = True


        if hand_near_mouth and not is_sipping:
            coffee_count += 1
            is_sipping = True

        elif not hand_near_mouth:
            is_sipping = False

        elapsed_time = time.time() - start_time

        cv2.putText(image, f"Coffee Sip Count: {coffee_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(image, f"Study Timer: {int(elapsed_time)} sec", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Coffee Drinking Count', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()