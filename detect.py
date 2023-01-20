import cv2

from utils import detect_landmarks, draw_landmarks, draw_edge


def webcam_detection():
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    while webcam.isOpened():
        status, frame = webcam.read()

        if status:
            
            frame = cv2.flip(frame, 1)
            landmarks = detect_landmarks(frame, img_type="np")
            for i in range(len(landmarks)):
                frame = draw_landmarks(image=frame, img_type="np", landmarks=landmarks[i])
                frame = draw_edge(image=frame, img_type="np", landmarks=landmarks[i])

            cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    webcam_detection()