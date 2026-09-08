from utils.detection import detect_image
from PIL import Image
import matplotlib.pyplot as plt


image = Image.open("test_images/test.jpg")

annotated_image, result = detect_image(image)

print("\nDetected objects:")

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    print(
        f"{result.names[class_id]} "
        f"→ {confidence:.2%}"
    )

plt.figure(figsize=(14, 10))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("SafeSight Detection")
plt.show()