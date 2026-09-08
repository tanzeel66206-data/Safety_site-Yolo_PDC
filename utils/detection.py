from ultralytics import YOLO


MODEL_PATH = "models/best.pt"

model = YOLO(MODEL_PATH)


def detect_image(image, confidence=0.25):

    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence_score = float(box.conf[0])

        detections.append({
            "class_name": result.names[class_id],
            "confidence": confidence_score
        })

    annotated_image = result.plot()

    # BGR → RGB
    annotated_image = annotated_image[:, :, ::-1]

    return annotated_image, detections