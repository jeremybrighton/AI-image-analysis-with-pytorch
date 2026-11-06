store for me this code

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from IPython.display import Video
import torch
import torchvision
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes, make_grid



print("Platform:", sys.platform)
print("Python version:", sys.version)
print("---")
print("CV2 version : ", cv2.__version__)
print("torch version : ", torch.__version__)
print("torchvision version : ", torchvision.__version__)


dhaka_image_dir = Path("data_images")

print("Data directory:", dhaka_image_dir)


dhaka_files = list(dhaka_image_dir.iterdir())
dhaka_files[-5:]


file_extension_counts = Counter(Path(file).suffix for file in dhaka_files)

for extension, count in file_extension_counts.items():
    print(f"Files with extension {extension}: {count}")


images_dir = dhaka_image_dir / "images"
annotations_dir = dhaka_image_dir / "annotations"

images_dir.mkdir(exist_ok=True)
annotations_dir.mkdir(exist_ok=True)


for file in dhaka_files:
    if file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".xml"):
        print(file)



  dhaka_files = [f for f in dhaka_image_dir.iterdir() if f.is_file()]


dhaka_image_dir = Path("data_images") / "train"
dhaka_files = list(dhaka_image_dir.iterdir())




for file in dhaka_files:
    if file.suffix.lower() in (".jpg", ".jpeg", ".png"):
        target_dir = images_dir
    elif file.suffix.lower() == ".xml":
        target_dir = annotations_dir
    file.rename(target_dir / file.name)




from collections import Counter

print("images:     ", len(images_files), Counter(p.suffix.lower() for p in images_files))
print("annotations:", len(annotations_files), Counter(p.suffix.lower() for p in annotations_files))
print("left in train:", Counter(p.suffix.lower() for p in dhaka_image_dir.iterdir() if p.is_file()))

img_names = {p.stem for p in images_files}
xml_names = {p.stem for p in annotations_files}
print("images without xml:", list(img_names - xml_names)[:5])
print("xml without image: ", list(xml_names - img_names)[:5])




xml_filepath = annotations_dir / "01.xml"
!head -n 25 $xml_filepath



tree = ET.parse(xml_filepath)
root = tree.getroot()
print(root.tag)



width = int(root.find("size").find("width").text)
height = int(root.find("size").find("height").text)
print(f"image width: {width}  image height: {height}")



width = int(root.find("size").find("width").text)
height = int(root.find("size").find("height").text)
print(f"image width: {width}  image height: {height}")



bounding_boxes = []
labels = []
for obj in root.findall("object"):
    label = obj.find("name").text
    labels.append(label)
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)
    bounding_boxes.append([xmin, ymin, xmax, ymax])

for label, bounding_box in zip(labels, bounding_boxes):
    print(f"{label}: {bounding_box}")



bboxes_tensor = torch.tensor(bounding_boxes, dtype=torch.float32)
print(bboxes_tensor)

image_path = images_dir / "01.jpg"
image = read_image(str(image_path))
print(image.shape)


image = draw_bounding_boxes(
    image=image,
    boxes=bboxes_tensor,
    labels=labels,
    width=3,
    fill=False,
    font="arial.ttf",
    font_size=10,
)





video_dir = Path("data_video")
video_name = "dhaka_traffic.mp4"
video_path = video_dir / video_name

print(video_path)



Video(video_path, embed=True)



video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print("Error: Could not open video.")
else:
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame rate: {frame_rate}")
    print(f"Total number of frames: {frame_count:,}")



success, first_frame = video_capture.read()
if success:
    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    plt.title("First Frame")
    plt.axis("off")
    plt.show()
else:
    print("Error: Could not read frame.")



video_capture.set(cv2.CAP_PROP_POS_FRAMES, 100)
success, later_frame = video_capture.read()
if success:
    plt.imshow(cv2.cvtColor(later_frame, cv2.COLOR_BGR2RGB))
    plt.title("First Frame")
    plt.axis("off")
    plt.show()
else:
    print("Error: Could not read frame.")




frames_dir = video_directory / "extracted_frames" 
frames_dir.mkdir(exist_ok=True)



frame_count = 0

while True:
    success, frame = video_capture.read()
    if not success:
        break

    # Save frames at the frame_rate
    if frame_count % frame_rate == 0:
        frame_path = frames_dir / f"frame_{frame_count}.jpg"
        cv2.imwrite(frame_path, frame)

    frame_count += 1

video_capture.release()



def display_sample_images(dir_path, sample=5):
    image_list = []
    images = sorted(dir_path.iterdir())
    if images:
        sample_images = images[:sample]
        for sample_image in sample_images:
            image = read_image(str(sample_image))
            resize_transform = transforms.Resize((240, 240))
            image = resize_transform(image)
            image_list.append(image)
    grid = make_grid(image_list, nrow=5)
    image = to_pil_image(grid)
    return image


display_sample_images(frames_dir, sample=10)


 We've covered several important aspects of working with image and video data for object detection, focusing on a traffic dataset. Here are the key things :

* Data organization: We explored how to load and organize our project dataset, separating images from their corresponding XML annotations.
* Handling different data sources: We worked with both a pre-existing image dataset and extracted frames from a YouTube video, showing how to prepare data from various sources.
* Video processing: We learned how to extract frames from a video at regular intervals, which is crucial for preparing video data for object detection tasks.
* XML parsing: We demonstrated how to parse XML annotations, which contain valuable information about object locations and classifications in our images.
* Bounding box visualization: We visualized bounding boxes on image data, an essential skill for understanding and validating our object detection results.

In the next lesson, we'll build upon these skills to use a pre-trained model for object detection.
