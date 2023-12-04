from roboflow import Roboflow
import supervision as sv
import cv2
import matplotlib.pyplot as plt

rf = Roboflow(api_key="TCBsNOYUEBBmbox9rfUM")
project = rf.workspace().project("head-count-ko384")
model = project.version(2).model

cap = cv2.VideoCapture(0)


people_count = 0

while True:
    
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    result = model.predict(frame, confidence=40, overlap=30).json()
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_roboflow(result)
    detections = detections[detections.class_id == 0]

    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()

    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    people_count = len(detections)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = int(0.02 * frame.shape[1])
    margin_top_percentage = 0.05
    margin_top = int(margin_top_percentage * frame.shape[0])
    thickness = 2

    cv2.putText(annotated_frame, f'People Count: {people_count}', (10, font_scale + margin_top),
                font, font_scale / 20, (0, 0, 255), thickness, cv2.LINE_AA)

    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
