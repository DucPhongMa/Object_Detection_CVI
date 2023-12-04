from roboflow import Roboflow
import supervision as sv
import matplotlib.pyplot as plt
import cv2

rf = Roboflow(api_key="TCBsNOYUEBBmbox9rfUM")
project = rf.workspace().project("head-count-ko384")
model = project.version(2).model

# Load the original image
original_image = cv2.imread("./pic-6.jpg")

result = model.predict("./pic-6.jpg", confidence=40, overlap=30).json()

labels = [item["class"] for item in result["predictions"]]

detections = sv.Detections.from_roboflow(result)

print(len(detections))

detections = detections[detections.class_id == 0]
print(len(detections))

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

image = cv2.imread("./pic-6.jpg")
image_height, image_width, _ = image.shape

font_scale = int(0.02 * image_width) 
margin_top_percentage = 0.05 
margin_top = int(margin_top_percentage * image_height)

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2 

cv2.putText(annotated_image, f'Predictions: {len(detections)}', (10, font_scale + margin_top), font, font_scale / 20, (0, 0, 255), 
            thickness, cv2.LINE_AA)

plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()

plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title("Annotated Image")
plt.axis("off")
plt.show()
