# =============================================================================
# Object Detection with YOLOv8 (Ultralytics)
# =============================================================================
# Object detection = identifying WHAT objects are in an image/video and WHERE
# they are (as bounding boxes). YOLO ("You Only Look Once") is a popular, fast,
# pretrained detector.
#
# This script covers running YOLO on:
#   1. A single image
#   2. A directory (batch) of images
#   3. A video (Python API and command line)
#
# It's written as a notebook-style script: the "# %%" markers split it into
# cells that run in VS Code / Jupyter, but it also runs top-to-bottom as a
# normal .py file.
#
# Assumed project layout (adjust the paths in Section 2 if yours differs):
#   data_video/
#       dhaka_traffic.mp4          <- source video
#       extracted_frames/          <- frames extracted from the video (.jpg)
#   output/                        <- truncated clips are written here
#   runs/detect/                   <- YOLO writes predictions here
#
# Requirements: ultralytics, torch, torchvision, pillow, ffmpeg (system tool),
# and IPython if you want inline video playback in a notebook.
# =============================================================================

# %% ---------------------------------------------------------------------------
# 1. Imports and environment check
# ------------------------------------------------------------------------------
import subprocess
import sys
from collections import Counter
from pathlib import Path

import PIL
import torch
import torchvision
import ultralytics
from IPython.display import Video  # only renders inside Jupyter/VS Code notebooks
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import make_grid
from ultralytics import YOLO

# Print versions so anyone re-running this can reproduce the environment.
# Version mismatches (especially ultralytics/torch) are a common source of bugs.
print("Platform:", sys.platform)
print("Python version:", sys.version)
print("---")
print("PIL version         :", PIL.__version__)
print("torch version       :", torch.__version__)
print("torchvision version :", torchvision.__version__)
print("ultralytics version :", ultralytics.__version__)

# Pick the GPU if available, otherwise fall back to CPU. The original notebook
# hard-coded "cuda", which crashes on machines without an NVIDIA GPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# %% ---------------------------------------------------------------------------
# 2. Paths used throughout the script
# ------------------------------------------------------------------------------
DATA_DIR = Path("data_video", "extracted_frames")   # folder of extracted frames
VIDEO_PATH = Path("data_video", "dhaka_traffic.mp4")  # full source video
OUTPUT_DIR = Path("output")                            # truncated clips go here
OUTPUT_DIR.mkdir(exist_ok=True)                        # create it if missing

# %% ---------------------------------------------------------------------------
# 3. Load the pretrained model
# ------------------------------------------------------------------------------
# "yolov8s.pt" = YOLOv8 *small*: a good balance of speed and accuracy.
# (n = nano is fastest, x = extra-large is most accurate.)
# The weights are downloaded automatically the first time.
yolo = YOLO(task="detect", model="yolov8s.pt")

# `yolo.names` maps class IDs (ints) to class names. The pretrained model was
# trained on the COCO dataset, so it knows 80 common classes.
print(yolo.names)

# Look up a specific class ID, e.g. ID 23.
class_assigned_to_23 = yolo.names[23]
print(f"{class_assigned_to_23} corresponds to 23")

# %% ---------------------------------------------------------------------------
# 4. Which of OUR classes does COCO NOT cover?
# ------------------------------------------------------------------------------
# The Dhaka traffic dataset has classes that COCO does not include. A pretrained
# model can't detect these, which is the motivation for fine-tuning a custom
# model later (the next lesson covers data augmentation to help with that).
classes_not_in_yolo = [
    "ambulance",
    "army vehicle",
    "auto rickshaw",
    "garbagevan",
    "human hauler",
    "minibus",
    "minivan",
    "pickup",
    "policecar",
    "rickshaw",
    "scooter",
    "suv",
    "taxi",
    "three wheelers (CNG)",
    "van",
    "wheelbarrow",
]

# Sanity-check: confirm that each of those classes is truly missing from YOLO.
coco_class_names = set(yolo.names.values())
for cls_name in classes_not_in_yolo:
    status = "NOT in" if cls_name not in coco_class_names else "IN"
    print(f"{cls_name:<22} {status} the pretrained YOLO classes")

# %% ---------------------------------------------------------------------------
# 5. Run inference on a single image (default settings)
# ------------------------------------------------------------------------------
image_path = DATA_DIR / "frame_1050.jpg"

# Calling the model directly runs prediction. YOLO returns a LIST of `Results`
# objects (one per image), even when you pass a single image.
result = yolo(image_path)
print(f"Type of result: {type(result)}")
print(f"Length of result: {len(result)}")

# Same thing for a second frame.
image_path_task = DATA_DIR / "frame_2575.jpg"
result_task = yolo(image_path_task)
print(f"Type of result_task: {type(result_task)}")

# %% ---------------------------------------------------------------------------
# 6. Run inference with explicit options via .predict()
# ------------------------------------------------------------------------------
# Useful arguments:
#   conf     - minimum confidence (0-1). Detections below it are discarded.
#   save     - save the image with bounding boxes drawn on it.
#   save_txt - save detections as .txt label files (class + normalized box).
result = yolo.predict(image_path, conf=0.5, save=True, save_txt=True)
result_task = yolo.predict(image_path_task, conf=0.5, save=True, save_txt=True)

# %% ---------------------------------------------------------------------------
# 7. Inspect the Results object
# ------------------------------------------------------------------------------
# `result[0]` is the Results object for the first (only) image.
# `.boxes` holds every detected bounding box, with class IDs and confidences.
print(result[0])
print(result[0].boxes)

# `.boxes.cls` = class ID for each detection (as a tensor).
print(result[0].boxes.cls)
print(f"Number of objects detected: {len(result[0].boxes.cls)}")

number_of_detected_objs = len(result_task[0].boxes.cls)
print(f"Number of objects detected in frame_2575.jpg: {number_of_detected_objs}")

# %% ---------------------------------------------------------------------------
# 8. Count detected objects per class
# ------------------------------------------------------------------------------
# Convert each class ID tensor value to an int, look up its name, then count.
object_counts = Counter(yolo.names[int(cls)] for cls in result[0].boxes.cls)
print(object_counts)

# Same for the second frame. NOTE: the original code built a plain list here,
# which has no `.most_common()`; it must be wrapped in a Counter.
object_counts_task = Counter(yolo.names[int(cls)] for cls in result_task[0].boxes.cls)

# `.most_common(n=1)` returns [(class_name, count)] for the most frequent class.
most_common_class, count_of_class = object_counts_task.most_common(n=1)[0]
print(f"Most common class: {most_common_class}")
print(f"Number of detected {most_common_class}: {count_of_class}")

# %% ---------------------------------------------------------------------------
# 9. Confidence scores
# ------------------------------------------------------------------------------
# `.boxes.conf` = confidence (0-1) for each detection.
print(result[0].boxes.conf)
print(f"Number of objects detected: {len(result[0].boxes.conf)}")

length_of_confidence_tensor = len(result_task[0].boxes.conf)
print(f"Number of objects detected (frame_2575): {length_of_confidence_tensor}")

# Count only detections above a stricter threshold (75%).
# `(tensor > 0.75)` gives booleans; `.sum()` counts the True values;
# `.item()` converts the 1-element tensor to a Python number.
CONF_THRESHOLD = 0.75
number_of_confident_objects = (result[0].boxes.conf > CONF_THRESHOLD).sum().item()
print(f"Objects detected in frame_1050.jpg with >{CONF_THRESHOLD:.0%} confidence: "
      f"{number_of_confident_objects}")

number_of_confident_objects_task = (result_task[0].boxes.conf > CONF_THRESHOLD).sum().item()
print(f"Objects detected in frame_2575.jpg with >{CONF_THRESHOLD:.0%} confidence: "
      f"{number_of_confident_objects_task}")

# %% ---------------------------------------------------------------------------
# 10. Bounding box formats
# ------------------------------------------------------------------------------
# YOLO exposes boxes in several formats:
#   xywh  - center x, center y, width, height          (in PIXELS)
#   xywhn - same as above                              (NORMALIZED to 0-1)
#   xyxy  - top-left x, y, bottom-right x, y           (in PIXELS)
#   xyxyn - same as above                              (NORMALIZED to 0-1)
print(result[0].boxes.xywh)

# `orig_shape` = (height, width) of the ORIGINAL image, before resizing.
print(result[0].orig_shape)

# First box, normalized (provided directly by YOLO):
print(result[0].boxes.xywhn[0])

# Verify by hand: dividing pixel values by the image size gives the same
# normalized numbers. Previously the size was hard-coded as 640x360; reading it
# from `orig_shape` is safer if the frame size ever changes.
h, w = result[0].orig_shape
scale = torch.tensor([w, h, w, h], device=result[0].boxes.xywh.device)
print(result[0].boxes.xywh[0] / scale)

# Same check for the second frame.
original_shape_task = result_task[0].orig_shape
print(f"Original shape of frame_2575.jpg (h, w): {original_shape_task}")

h_t, w_t = original_shape_task
scale_t = torch.tensor([w_t, h_t, w_t, h_t], device=result_task[0].boxes.xywh.device)
normalized_xywh_task = result_task[0].boxes.xywh / scale_t
print(f"Normalized xywh box for frame_2575.jpg: {normalized_xywh_task[0]}")

# Corner-format boxes, in pixels and normalized.
print(result[0].boxes.xyxy)
print(result[0].boxes.xyxyn)

# %% ---------------------------------------------------------------------------
# 11. Where were the results saved?
# ------------------------------------------------------------------------------
# YOLO creates a new folder per run: runs/detect/predict, predict2, predict3...
# `save_dir` is returned as a string, so wrap it in Path for convenience.
location_of_results = Path(result[0].save_dir)
print(f"Results saved to {location_of_results}")
print("Exists:", location_of_results.exists())

location_of_results_task = Path(result_task[0].save_dir)
print(f"Results for frame_2575.jpg saved to {location_of_results_task}")

# %% ---------------------------------------------------------------------------
# 12. Inference speed
# ------------------------------------------------------------------------------
# `.speed` = milliseconds spent on preprocess, inference and postprocess.
print(result[0].speed)
print(f"Total time in milliseconds: {sum(result[0].speed.values()):.1f}")

total_time = sum(result_task[0].speed.values())
print(f"Total time (frame_2575) in milliseconds: {total_time:.1f}")

# %% ---------------------------------------------------------------------------
# 13. View the annotated image and the saved label file
# ------------------------------------------------------------------------------
# Annotated image (bounding boxes drawn). The last expression in a notebook
# cell is displayed automatically; in a plain script use `.show()`.
annotated = Image.open(location_of_results_task / "frame_2575.jpg")
annotated  # noqa: B018  (displays in notebooks)

# Label files (from save_txt=True): each line is
#   <class_id> <x_center> <y_center> <width> <height>   (all normalized 0-1)
with (location_of_results / "labels" / "frame_1050.txt").open("r") as f:
    print(f.read())

with (location_of_results_task / "labels" / "frame_2575.txt").open("r") as f:
    print(f.read())

# %% ---------------------------------------------------------------------------
# 14. Helper: show a grid of sample images from a folder
# ------------------------------------------------------------------------------
def display_sample_images(dir_path, sample=5, per_row=5, size=(240, 240)):
    """Return a PIL image containing a grid of the first `sample` .jpg files.

    Parameters
    ----------
    dir_path : str | Path   Folder containing .jpg images.
    sample   : int          How many images to include.
    per_row  : int          Images per row in the grid.
    size     : (int, int)   (height, width) each image is resized to, so that
                            all images are the same size (required by make_grid).

    Returns None if the folder has no .jpg files.
    """
    dir_path = Path(dir_path)

    # Sort so the images always appear in a predictable order.
    images = sorted(dir_path.glob("*.jpg"))
    if not images:
        return None

    resize = transforms.Resize(size)
    image_list = [resize(read_image(str(p))) for p in images[:sample]]

    # make_grid stitches the tensors into one image tensor.
    grid = make_grid(image_list, nrow=per_row)
    # Convert the tensor to a PIL Image so it displays nicely.
    return transforms.ToPILImage()(grid)


# Preview the input frames (25 then 10).
display_sample_images(DATA_DIR, sample=25)
display_sample_images(DATA_DIR, sample=10)

# %% ---------------------------------------------------------------------------
# 15. Run inference on MULTIPLE images (a batch)
# ------------------------------------------------------------------------------
# First 25 frames. NOTE: iterdir() order is arbitrary, so we sort for
# reproducibility. YOLO accepts a list of paths (strings are safest).
images_path = sorted(DATA_DIR.iterdir())[:25]

results = yolo.predict(
    [str(p) for p in images_path],
    conf=0.5,
    save=True,
    save_txt=True,
    project=Path("runs", "detect"),  # parent output folder
    name="multiple_frames",          # sub-folder name for this run
)
print(results[0].save_dir)

# Last 10 frames (a second batch, using YOLO's default confidence of 0.25).
images_path_task = sorted(DATA_DIR.iterdir())[-10:]
print(f"Number of frames in list: {len(images_path_task)}")

results_task = yolo.predict(
    [str(p) for p in images_path_task],
    save=True,
    project="runs/detect",
    name="multiple_frames_task",
)
print(f"Results from task saved to: {results_task[0].save_dir}")

# Visualize the annotated outputs as grids.
display_sample_images(results[0].save_dir, sample=25)
display_sample_images(results_task[0].save_dir, sample=10)


# %% ---------------------------------------------------------------------------
# 16. Video: cut short clips with ffmpeg
# ------------------------------------------------------------------------------
# Running YOLO on a full video is slow, so we trim it first.
# `-c copy` copies streams without re-encoding (fast, but cuts land on
# keyframes so timing can be slightly off). `-y` overwrites existing output.
def trim_video(src, dst, start, end):
    """Cut `src` from `start` to `end` (HH:MM:SS) into `dst` using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-ss", start, "-to", end, "-y", "-i", str(src), "-c", "copy", str(dst)],
        check=True,  # raise an error if ffmpeg fails
    )


video_truncated_path = OUTPUT_DIR / "dhaka_traffic_truncated.mp4"            # 0:00-0:30
video_truncated_path_task = OUTPUT_DIR / "dhaka_traffic_truncated_task.mp4"  # 0:30-1:00

trim_video(VIDEO_PATH, video_truncated_path, "00:00:00", "00:00:30")
trim_video(VIDEO_PATH, video_truncated_path_task, "00:00:30", "00:01:00")

# Play the original and the second clip inline (notebooks only).
Video(VIDEO_PATH)
Video(str(video_truncated_path_task))

# %% ---------------------------------------------------------------------------
# 17. Video inference with the Python API
# ------------------------------------------------------------------------------
# stream=True returns a GENERATOR: frames are processed one at a time instead
# of all being stored in memory (important for long videos).
# Nothing happens until you iterate over the generator.
results_video = yolo.predict(
    str(video_truncated_path),
    conf=0.5,
    save=True,  # writes the annotated video to runs/detect/video_source
    stream=True,
    project=Path("runs", "detect"),
    name="video_source",
)

# Consume the generator so every frame is processed and the video is saved.
for _ in results_video:
    continue

# %% ---------------------------------------------------------------------------
# 18. Video inference with the command-line interface (CLI)
# ------------------------------------------------------------------------------
# The same model can be run from the terminal. Output is hidden (DEVNULL) to
# keep the notebook clean; remove `stdout=` to see progress.
def run_yolo_cli(source, name):
    subprocess.run(
        [
            "yolo", "task=detect", "mode=predict", "conf=0.5",
            "model=yolov8s.pt", f"source={source}",
            "project=runs/detect", f"name={name}",
        ],
        stdout=subprocess.DEVNULL,
        check=True,
    )


def avi_to_mp4(avi_path, mp4_path):
    """YOLO's CLI saves video as .avi, which notebooks can't play; convert it.

    NOTE: the output extension can differ between ultralytics versions
    (.avi vs .mp4). Check the runs/detect/<name>/ folder if this fails.
    """
    subprocess.run(["ffmpeg", "-y", "-i", str(avi_path), str(mp4_path)], check=True)


# First clip
run_yolo_cli(video_truncated_path, "command_line")
avi_to_mp4("runs/detect/command_line/dhaka_traffic_truncated.avi", "output.mp4")
Video("output.mp4")

# Second clip
run_yolo_cli(video_truncated_path_task, "command_line_task")
avi_to_mp4(
    "runs/detect/command_line_task/dhaka_traffic_truncated_task.avi",
    "output_task.mp4",
)
Video("output_task.mp4")

# =============================================================================
# Summary / what's next
# =============================================================================
# - Object detection identifies the objects in an image or video.
# - YOLO is a fast, popular pretrained detector. We used it on a single image,
#   a directory of images, and a video source.
# - The pretrained model only knows COCO's 80 classes, so it misses
#   Dhaka-specific vehicles (rickshaws, CNG three-wheelers, etc.).
# - Next lesson: data augmentation techniques, which help when customizing
#   (fine-tuning) a pretrained detection model on our own classes.
# =============================================================================
