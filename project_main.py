from roboflow import Roboflow
import supervision as sv
import cv2

rf = Roboflow(api_key="TCBsNOYUEBBmbox9rfUM")
project = rf.workspace().project("head-count-ko384")
model = project.version(1).model

result = model.predict("./pic-5.jpg", confidence=40, overlap=30).json()

labels = [item["class"] for item in result["predictions"]]

detections = sv.Detections.from_roboflow(result)

print(len(detections))

# filter by class
detections = detections[detections.class_id == 0]
print(len(detections))

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

image = cv2.imread("./pic-5.jpg")

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

# Add the number of predictions as bold text on the image with larger font size (2), red color, and a margin-top of 50 pixels
font = cv2.FONT_HERSHEY_SIMPLEX
margin_top = 50
thickness = 3  # Set thickness for bold effect

cv2.putText(annotated_image, f'Predictions: {len(detections)}', (10, 30 + margin_top), font, 2, (0, 0, 255), thickness, cv2.LINE_AA)

sv.plot_image(image=annotated_image, size=(16, 16))
