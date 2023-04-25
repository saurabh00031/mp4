import cv2
import dlib
import pyttsx3

# initialize the voice assistant
engine = pyttsx3.init()

# define the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# capture the default camera
cap = cv2.VideoCapture(0)

# define the lip landmarks indices
lip_indices = list(range(48, 61)) + [67, 66, 65, 64]

# define the lip-sync text
text = "Hello, how can I help you today?"

# set the voice assistant voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 0 for male, 1 for female

# define the lip-sync animation function
def lip_sync(frame, landmarks):
    lip_mean = sum(landmarks[lip_indices]) / len(lip_indices)
    lip_mean = lip_mean.astype(int)
    cv2.circle(frame, tuple(lip_mean), 5, (0, 0, 255), -1)

# start the lip-sync animation and voice assistant
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        lip_sync(frame, landmarks.parts())

    # show the frame and play the voice assistant audio
    cv2.imshow('frame', frame)
    engine.say(text)
    engine.runAndWait()

    # stop the lip-sync animation and voice assistant when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
