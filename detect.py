import cv2

import torch

from models import Xception

import dlib

from utils import draw_landmarks, draw_edge, to_np, to_tensor_image


face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_faces(image, output_type="dlib.rectangle"):
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    face_coordinates = face_detector(gray_img)

    if output_type == "dlib.rectangle": return face_coordinates

    elif output_type == "list":
        results = []

        for face in face_coordinates:
            results.append([
                [face.left(), face.top()],
                [face.right(), face.bottom()]
            ])

        return results
    
    
def detect_landmarks_dlib(image):

    face_coordinates = detect_faces(image)
    
    results = []
    
    for coordinate in face_coordinates:
        landmarks = landmark_detector(image, coordinate)
        
        results.append(
            torch.tensor([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        )

    return results


def detect_landmarks_xception(model, device, image, img_type):

    img = to_tensor_image(image, img_type, device)
    
    landmarks = model(img.unsqueeze(0))
    landmarks = landmarks.cpu().detach().view(1, 68, 2)

    return [landmarks[0]]


def webcam_detection(mode="dlib", model=None, device=None):
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    while webcam.isOpened():
        status, frame = webcam.read()

        if status:
            
            frame = cv2.flip(frame, 1)
            if mode == "dlib":
                np_frame = to_np(frame, img_type="np")
                landmarks = detect_landmarks_dlib(np_frame)

            elif mode == "xception":
                with torch.no_grad():
                    landmarks = detect_landmarks_xception(model, device, frame, img_type="np")

            for i in range(len(landmarks)):
                frame = draw_landmarks(image=frame, img_type="np", landmarks=landmarks[i])
                frame = draw_edge(image=frame, img_type="np", landmarks=landmarks[i])

            cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Xception(middle_repeat_n=4).to(device)
    model.load_state_dict(torch.load("./facial_landmark_detection.pt"))
    model.eval()

    # webcam_detection(mode="dlib")
    webcam_detection(mode="xception", model=model, device=device)