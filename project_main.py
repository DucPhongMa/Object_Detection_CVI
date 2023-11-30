from roboflow import Roboflow
import supervision as sv
import cv2

rf = Roboflow(api_key="TCBsNOYUEBBmbox9rfUM")
project = rf.workspace().project("head-count-ko384")
model = project.version(1).model

result = model.predict("./pic-1.jpg", confidence=40, overlap=30).json()

labels = [item["class"] for item in result["predictions"]]

detections = sv.Detections.from_roboflow(result)

print(len(detections))

# filter by class
detections = detections[detections.class_id == 0]
print(len(detections))

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

image = cv2.imread("./pic-1.jpg")
image_height, image_width, _ = image.shape

# Calculate text size and margin-top as a percentage of the image size
font_scale = int(0.02 * image_width)  # Adjust the multiplier as needed
margin_top_percentage = 0.05  # Adjust as needed
margin_top = int(margin_top_percentage * image_height)

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

# Add the number of predictions as bold text on the image with dynamically calculated font size, red color, and margin-top
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2  # Adjust as needed

cv2.putText(annotated_image, f'Predictions: {len(detections)}', (10, font_scale + margin_top), font, font_scale / 20, (0, 0, 255), thickness, cv2.LINE_AA)

sv.plot_image(image=annotated_image, size=(16, 16))
