先切到對的資料夾
cd ./examples/example_scripts
運行test.mp4
python mot_YOLOv3.py --video ./../video_data/test.mp4 --weights ./../pretrained_models/yolo_weights/yolov3.weights --config ./../pretrained_models/yolo_weights/yolov3.cfg --labels ./../pretrained_models/yolo_weights/coco_names.json
使用鏡頭運行
python mot_YOLOv3.py --video 0 --weights ./../pretrained_models/yolo_weights/yolov3.weights --config ./../pretrained_models/yolo_weights/yolov3.cfg --labels ./../pretrained_models/yolo_weights/coco_names.json