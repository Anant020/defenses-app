import cv2
from ultralytics import YOLO
import numpy as np
import time

# Load the YOLOv8 model
# For first time use, you'll need to download a model or train your own
# model = YOLO("yolov8n.pt")  # Use a pre-trained model (for general objects)
model = YOLO("yolov8_weapon_model.pt")  # Use your trained weapon model

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set the confidence threshold
conf_threshold = 0.5

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Record the start time for FPS calculation
    start_time = time.time()
    
    # Run YOLOv8 inference on the frame
    results = model(frame, conf=conf_threshold)
    
    # Get the boxes, confidence values, and class IDs
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
    confs = results[0].boxes.conf.cpu().numpy()  # Get confidence values
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs
    
    # Get the class names
    class_names = model.names if hasattr(model, 'names') else {0: 'Weapon'}  # Default to 'Weapon' if no names are available
    
    # Draw the detections on the frame
    for box, conf, class_id in zip(boxes, confs, class_ids):
        # Extract coordinates
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label
        label = f"{class_names[class_id]}: {conf:.2f}"
        
        # Calculate label size
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw label background
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Add label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Calculate FPS
    fps = 1 / (time.time() - start_time)
    
    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the number of weapons detected
    num_weapons = len(boxes)
    if num_weapons > 0:
        print(f"Weapon detected in frame: {num_weapons}")
    
    # Display the frame
    cv2.imshow("Weapon Detection (YOLOv8)", frame)
    
    # Break the loop when 'ESC' is pressed
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Load Yolo
# net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# classes = ["Weapon"]
# count =0
# # Open webcam (0 is usually the default webcam)
# cap = cv2.VideoCapture(0)

# # Check if webcam opened successfully
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
    
# while True:
#     ret, img = cap.read()
    
#     # If frame is not successfully captured, break
#     if not ret:
#         print("Failed to grab frame")
#         break
#     # width = 512
#     # height = 512
#     height, width, channels = img.shape

#     # Detecting objects
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
    
#     layer_names = net.getLayerNames()
    
#     # Fix for different OpenCV versions
#     try:
#         output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#     except:
#         output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
#     colors = np.random.uniform(0, 255, size=(len(classes), 3))
#     outs = net.forward(output_layers)

#     # Showing information on the screen
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)

#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
#     # Check if weapons are detected
#     if len(indexes) > 0:
#         count = count + 1
#         print(f"Weapon detected in frame , Log : ",count)
    
#     font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             color = colors[class_ids[i]]
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

#     cv2.imshow("Weapon Detection", img)
    
#     # Press Esc to exit
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
        
# cap.release()
# cv2.destroyAllWindows()