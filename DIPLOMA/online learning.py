import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


# Load the base MobileNetV2 model, excluding the top layer (used for classification)
base_model = MobileNetV2(weights='imagenet', include_top=False)
# Initialize CSRT tracker from OpenCV
tracker = cv2.legacy.TrackerCSRT_create()

# Add new layers for object tracking
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # Two classes: object vs. background

# Define the model that we'll train
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base layers to retain pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize image to the input size for MobileNet
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Preprocess image as per MobileNetV2 requirements
    return img


def online_learning(model, img, label):
    # Preprocess the image
    img = preprocess_image(img)

    # Create a label for the object (1 for object, 0 for background)
    label = np.array([label])
    label = tf.keras.utils.to_categorical(label, num_classes=2)

    # Fine-tune the model on the single image (online learning)
    model.fit(img, label, epochs=1, verbose=0)

def detect_and_track(model, frame, bbox):
    # Crop the image using the bounding box
    x, y, w, h = map(int, bbox)

    # Crop the image using the bounding box
    cropped_img = frame[y:y + h, x:x + w]

    # Preprocess the image for the model
    img = preprocess_image(cropped_img)

    # Predict object presence
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]

    return class_idx, confidence


def main():
    # Load a video or webcam feed
    cap = cv2.VideoCapture(0)  # Use webcam, or pass 'video.mp4' for video

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        return

    # Select ROI (Region of Interest) manually for the first frame
    bbox = cv2.selectROI(frame, False)
    tracker.init(frame, bbox)  # Initialize the CSRT tracker

    # Tracking label (1 for object)
    tracking_label = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker with the current frame
        success, bbox = tracker.update(frame)

        if success:
            # Detect and track object using the AI model
            class_idx, confidence = detect_and_track(model, frame, bbox)

            # Draw bounding box if object is confidently tracked
            if class_idx == tracking_label and confidence > 0.5:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Online learning: fine-tune on the tracked object
                online_learning(model, frame[y:y+h, x:x+w], tracking_label)

        # Display the tracking result
        cv2.imshow("Tracking", frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
