import cv2
import mediapipe as mp


def process_image(image, face_detection):

    H, W, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_face = face_detection.process(image_rgb)

    if output_face.detections is not None:

        for detections in output_face.detections:
            location_data = detections.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            final_image = cv2.rectangle(image, (x1,y1), (x1 + w, y1 + h), (0, 0, 255), 5)

    return final_image


# read image
img = cv2.imread('Photos/Family Photo.png')
image = cv2.resize(img, [800, 600])

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:

    final_image = process_image(image, face_detection)

    cv2.imshow('image', final_image)
    cv2.waitKey(0)
