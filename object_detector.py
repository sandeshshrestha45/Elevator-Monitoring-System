import cv2
import numpy as np

def load_class_names(labels_path):
    with open(labels_path, "r") as f:
        return [line.strip() for line in f.readlines()]

class ObjectDetector:
    def __init__(self, weights, config, labels_path, confidence_threshold=0.5):
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.class_names = load_class_names(labels_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.4

    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(layer_names)

        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        filtered_boxes, filtered_labels, filtered_confidences = [], [], []
        for i in indices:
            i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
            filtered_boxes.append(boxes[i])
            filtered_labels.append(self.class_names[class_ids[i]])
            filtered_confidences.append(confidences[i])

        return filtered_boxes, filtered_labels, filtered_confidences

    def draw_bbox(self, frame, bbox, labels, confidences):
        for box, label, conf in zip(bbox, labels, confidences):
            x, y, w, h = box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
