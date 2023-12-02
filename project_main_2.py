from roboflow import Roboflow
import supervision as sv
import cv2
import matplotlib.pyplot as plt

rf = Roboflow(api_key="TCBsNOYUEBBmbox9rfUM")
project = rf.workspace().project("head-count-ko384")
model = project.version(1).model

# Open a connection to the webcam (usually 0 for built-in webcams)
cap = cv2.VideoCapture(0)

# Initialize count variable
people_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Break the loop if the video capture fails
    if not ret:
        print("Failed to capture frame")
        break

    # Perform object detection on the frame
    result = model.predict(frame, confidence=40, overlap=30).json()
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_roboflow(result)
    detections = detections[detections.class_id == 0]  # Filter by class (assuming people have class_id 0)

    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()

    # Annotate the frame with bounding boxes and labels
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Update people count
    people_count = len(detections)

    # Add the number of people as text on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = int(0.02 * frame.shape[1])
    margin_top_percentage = 0.05
    margin_top = int(margin_top_percentage * frame.shape[0])
    thickness = 2

    cv2.putText(annotated_frame, f'People Count: {people_count}', (10, font_scale + margin_top),
                font, font_scale / 20, (0, 0, 255), thickness, cv2.LINE_AA)

    # Display the annotated frame using matplotlib
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.show()

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
