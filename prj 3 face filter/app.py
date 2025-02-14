import cv2
import mediapipe as mp
import numpy as np

# Load sunglasses image with alpha channel
try:
    sunglasses = cv2.imread(r'D:\Luminar\dl\dlprj\prj 3 face filter\sunglasses_rgba.png', cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        raise FileNotFoundError("Sunglasses image not found. Please check the file path.")
except Exception as e:
    print(e)
    sunglasses = None

# Load mouth overlay image (e.g., tongue or teeth)
try:
    mouth_overlay = cv2.imread(r'D:\Luminar\dl\dlprj\prj 3 face filter\open_mouth_filter.png', cv2.IMREAD_UNCHANGED)
    if mouth_overlay is None:
        raise FileNotFoundError("Mouth overlay image not found. Please check the file path.")
except Exception as e:
    print(e)
    mouth_overlay = None

# Initialize MediaPipe Face Mesh
facemesh = mp.solutions.face_mesh
mesh = facemesh.FaceMesh(
    static_image_mode=False,  # Set to False for video
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5  # Add this to improve tracking
)

mpdraw = mp.solutions.drawing_utils
face_spec = mpdraw.DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 255))
r_eye_spec = mpdraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
l_eye_spec = mpdraw.DrawingSpec(thickness=1, circle_radius=1, color=(255, 0, 0))
mouth_spec = mpdraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255))

def is_mouth_open(landmarks):
    upper_lip = landmarks[13].y
    lower_lip = landmarks[14].y
    chin = landmarks[152].y
    forehead = landmarks[10].y

    mouth_opening = abs(lower_lip - upper_lip)
    face_height = abs(chin - forehead)

    # Adjust threshold based on testing
    return (mouth_opening / face_height) > 0.1

def is_eye_open(landmarks, upper, lower, face_height):
    eye_opening = abs(landmarks[upper].y - landmarks[lower].y)
    return (eye_opening / face_height) > 0.035

def is_smiling(landmarks):
    # Get coordinates of lip corners
    left_corner_x = landmarks[61].x
    left_corner_y = landmarks[61].y
    right_corner_x = landmarks[291].x
    right_corner_y = landmarks[291].y

    # Calculate horizontal distance between lip corners
    lip_distance = abs(right_corner_x - left_corner_x)

    # Reference point for upper lip height
    upper_lip_center_y = landmarks[13].y

    # Condition 1: Both mouth corners are raised above the upper lip center
    corners_raised = left_corner_y < upper_lip_center_y and right_corner_y < upper_lip_center_y

    # Condition 2: Lip distance increases (wider smile)
    lip_distance_threshold = 0.1  # Example threshold (relative to face width)
    face_width = abs(landmarks[454].x - landmarks[234].x)  # Estimate face width using outer face landmarks
    lip_distance_increases = lip_distance > (lip_distance_threshold * face_width)

    # Smiling if both conditions are met
    return corners_raised and lip_distance_increases

def overlay_sunglasses(img, landmarks):
    """ Overlay sunglasses on detected face. """
    if sunglasses is None:
        return img  # Skip overlay if sunglasses image is not loaded

    left_eye_x, left_eye_y = int(landmarks[33].x * img.shape[1]), int(landmarks[33].y * img.shape[0])
    right_eye_x, right_eye_y = int(landmarks[263].x * img.shape[1]), int(landmarks[263].y * img.shape[0])

    # Compute width & height of sunglasses
    width = abs(right_eye_x - left_eye_x) * 2
    height = int(width * 0.5)

    # Resize sunglasses
    resized_glasses = cv2.resize(sunglasses, (width, height))

    # Add an alpha channel if the image doesn't have one
    if resized_glasses.shape[2] == 3:
        alpha_channel = np.ones((resized_glasses.shape[0], resized_glasses.shape[1]), dtype=resized_glasses.dtype) * 255
        resized_glasses = cv2.merge((resized_glasses, alpha_channel))

    # Get position
    x_offset = left_eye_x - int(width * 0.3)
    y_offset = left_eye_y - int(height * 0.5)

    # Ensure the sunglasses don't go out of the frame
    y1, y2 = max(0, y_offset), min(img.shape[0], y_offset + height)
    x1, x2 = max(0, x_offset), min(img.shape[1], x_offset + width)

    if y2 - y1 > 0 and x2 - x1 > 0:
        resized_glasses = cv2.resize(resized_glasses, (x2 - x1, y2 - y1))
        alpha_s = resized_glasses[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_s * resized_glasses[:, :, c] +
                                    alpha_l * img[y1:y2, x1:x2, c])
    return img

def overlay_mouth(img, landmarks):
    """ Overlay mouth effect when mouth is open. """
    if mouth_overlay is None:
        return img  # Skip overlay if mouth overlay image is not loaded

    # Get mouth landmarks
    left_mouth = (int(landmarks[61].x * img.shape[1]), int(landmarks[61].y * img.shape[0]))
    right_mouth = (int(landmarks[291].x * img.shape[1]), int(landmarks[291].y * img.shape[0]))
    upper_lip = (int(landmarks[13].x * img.shape[1]), int(landmarks[13].y * img.shape[0]))
    lower_lip = (int(landmarks[14].x * img.shape[1]), int(landmarks[14].y * img.shape[0]))

    # Compute width & height of mouth overlay
    width = abs(right_mouth[0] - left_mouth[0]) * 1.2  # Slightly larger than mouth width
    height = abs(lower_lip[1] - upper_lip[1]) * 2.0  # Adjust height based on mouth opening

    # Ensure width and height are valid
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        return img  # Skip overlay if width or height is invalid

    # Resize mouth overlay
    resized_mouth = cv2.resize(mouth_overlay, (width, height))

    # Get position
    x_offset = left_mouth[0] - int(width * 0.1)
    y_offset = upper_lip[1] - int(height * 0.3)

    # Ensure the mouth overlay doesn't go out of the frame
    y1, y2 = max(0, y_offset), min(img.shape[0], y_offset + height)
    x1, x2 = max(0, x_offset), min(img.shape[1], x_offset + width)

    if y2 - y1 > 0 and x2 - x1 > 0:
        resized_mouth = cv2.resize(resized_mouth, (x2 - x1, y2 - y1))
        alpha_s = resized_mouth[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_s * resized_mouth[:, :, c] +
                                    alpha_l * img[y1:y2, x1:x2, c])
    return img


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mesh.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_face_landmarks:
        for me in result.multi_face_landmarks:
            landmarks = me.landmark

            if is_smiling(landmarks):
                img = overlay_sunglasses(img, landmarks)     
            if is_mouth_open(landmarks):
                img = overlay_mouth(img, landmarks)

            # Face height calculation
            face_height = abs(landmarks[152].y - landmarks[10].y)

            # Eye and mouth detection
            smile_status = "smiling" if is_smiling(landmarks) else "not smiling"
            mouth_status = "Mouth Open" if is_mouth_open(landmarks) else "Mouth Closed"
            left_eye_status = "Left Eye Open" if is_eye_open(landmarks, 159, 145, face_height) else "Left Eye Closed"
            right_eye_status = "Right Eye Open" if is_eye_open(landmarks, 386, 374, face_height) else "Right Eye Closed"

            # Display text
            cv2.putText(img, smile_status, (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, mouth_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, left_eye_status, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, right_eye_status, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

video = cv2.VideoCapture(0)
while True:
    suc, img = video.read()
    if not suc:
        break
    img = preprocessing(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()