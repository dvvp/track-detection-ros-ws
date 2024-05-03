import math

import cv2
import numpy as np

from helpers.utils import draw_detections, nms, sigmoid, xywh2xyxy


class Segment:
    def __init__(
        self,
        input_shape=[1, 3, 192, 320],
        input_height=192,
        input_width=320,
        conf_thres=0.9,
        iou_thres=0.5,
        num_masks=32,
    ):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks

        self.input_names = "images"
        self.input_shape = input_shape
        self.input_height = input_height
        self.input_width = input_width
        self.output_names = ["output0", "output1"]

    def segment_objects_from_oakd(self, output0, output1):

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(
            output0
        )
        self.mask_maps = self.process_mask_output(mask_pred, output1)

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input_for_oakd(self, shape):

        self.img_height = shape[0]
        self.img_width = shape[1]

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4 : 4 + num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., : num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 :]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return (
            boxes[indices],
            scores[indices],
            class_ids[indices],
            mask_predictions[indices],
        )

    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(
            self.boxes, (self.img_height, self.img_width), (mask_height, mask_width)
        )

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (
            int(self.img_width / mask_width),
            int(self.img_height / mask_height),
        )
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(
                scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC
            )

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(
            boxes,
            (self.input_height, self.input_width),
            (self.img_height, self.img_width),
        )

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        # print(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(
            image, self.boxes, self.scores, self.class_ids, mask_alpha
        )

    def calculate_centroid(self, mask_map):
        contours, _ = cv2.findContours(mask_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour_area = 0
        max_contour_index = -1
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_contour_area:
                max_contour_area = area
                max_contour_index = idx
        if max_contour_index != -1:
            M = cv2.moments(contours[max_contour_index])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Calculate x error
                x_error = cX - (self.img_width // 2)  # Distance from the centerline
                return (cX, cY, x_error)
        return None

    def draw_masks(self, image):
        for i in range(len(self.boxes)):
            centroid_data = self.calculate_centroid(self.mask_maps[i])
            if centroid_data is not None:
                centroid_x, centroid_y, error_x = centroid_data
                contours, _ = cv2.findContours(self.mask_maps[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(image, contours, True, (0, 255, 0), thickness=2)

                # Calculate center of the image
                center_x = self.img_width // 2
                
                cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                # Draw line from centroid to center of the image
                cv2.line(image, (centroid_x, centroid_y), (center_x, centroid_y), (255, 0, 0), 2)
                return image, error_x
        return image, None
    
    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [input_shape[1], input_shape[0], input_shape[1], input_shape[0]]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]
        )
        return boxes