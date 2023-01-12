import cv2 as cv
import numpy as np
from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks


def click(event, x, y, flags, param):
    global tracker
    if event == cv.EVENT_LBUTTONDOWN:
        for i in list(tracker.tracks.keys()):
            if tracker.tracks[i].bbox[0] < x and tracker.tracks[i].bbox[0] + tracker.tracks[i].bbox[2] > x and tracker.tracks[i].bbox[1] < y and tracker.tracks[i].bbox[1] + tracker.tracks[i].bbox[3] > y:
                tracker.tracks[i].display = not tracker.tracks[i].display
                
def main(video_path, model):
    global tracker
    if video_path == "0":
        video_path = 0
    cap = cv.VideoCapture(video_path)
    while True:
        ok, image = cap.read()

        if not ok:
            print("Cannot read the video feed.", video_path)
            break
        
        image = cv.resize(image, (700, 500))
        
        all_bboxes, all_confidences, all_class_ids = model.detect(image)
        
        tracks = tracker.update(all_bboxes, all_confidences, all_class_ids)
        
        
        
        bboxes = []
        confidences = []
        class_ids = []
        for i in list(tracker.tracks.keys()):
            if tracker.tracks[i].display == True:
                bboxes.append(tracker.tracks[i].bbox)
                class_ids.append(tracker.tracks[i].class_id)
                confidences.append(tracker.tracks[i].detection_confidence)
        #print(class_ids, len(class_ids), len(bboxes), len(confidences))
        bboxes = np.array(bboxes, dtype='i')
        
        updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)

        updated_image = draw_tracks(updated_image, tracks)

        cv.imshow("image", updated_image)
        #mouse click
        cv.setMouseCallback("image", click)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Object detections in input video using YOLOv3 trained on COCO dataset.'
    )

    parser.add_argument(
        '--video', '-v', type=str, default="./../video_data/test.mp4", help='Input video path.')

    parser.add_argument(
        '--weights', '-w', type=str,
        default="./../pretrained_models/yolo_weights/yolov3.weights",
        help='path to weights file of YOLOv3 (`.weights` file.)'
    )

    parser.add_argument(
        '--config', '-c', type=str,
        default="./../pretrained_models/yolo_weights/yolov3.cfg",
        help='path to config file of YOLOv3 (`.cfg` file.)'
    )

    parser.add_argument(
        '--labels', '-l', type=str,
        default="./../pretrained_models/yolo_weights/coco_names.json",
        help='path to labels file of coco dataset (`.names` file.)'
    )

    parser.add_argument(
        '--gpu', type=bool,
        default=False, help='Flag to use gpu to run the deep learning model. Default is `False`'
    )

    parser.add_argument(
        '--tracker', type=str, default='CentroidTracker',
        help="Tracker used to track objects. Options include ['CentroidTracker', 'CentroidKF_Tracker', 'SORT']")

    args = parser.parse_args()

    if args.tracker == 'CentroidTracker':
        tracker = CentroidTracker(max_lost=10, tracker_output_format='mot_challenge')
    elif args.tracker == 'CentroidKF_Tracker':
        tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
    elif args.tracker == 'SORT':
        tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
    elif args.tracker == 'IOUTracker':
        tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
                             tracker_output_format='mot_challenge')
    else:
        raise NotImplementedError

    model = YOLOv3(
        weights_path=args.weights,
        configfile_path=args.config,
        labels_path=args.labels,
        confidence_threshold=0.5,
        nms_threshold=0.2,
        draw_bboxes=True,
        use_gpu=args.gpu
    )

    main(args.video, model)
