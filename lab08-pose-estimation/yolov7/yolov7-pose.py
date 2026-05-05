import time
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
from torchvision import transforms

from utils.datasets import letterbox
from utils.general  import non_max_suppression_kpt
from utils.plots    import output_to_keypoint, plot_skeleton_kpts


def pose_video(frame):
    original_h, original_w = frame.shape[:2]
    # Letterbox resizing.
    img = letterbox(frame, input_size, stride=64, auto=True)[0]
    # Convert the array to 4D.
    img = transforms.ToTensor()(img)
    # Convert the array to Tensor.
    img = torch.tensor(np.array([img.numpy()]))
    # Load the image into the computation device.
    img = img.to(device)
    
    # Gradients are stored during training, not required while inference.
    with torch.no_grad():
        t1 = time.time()
        output, _ = model(img)
        t2 = time.time()
        fps = 1/(t2 - t1)
        output = non_max_suppression_kpt(output, 
                                         0.25,    # Conf. Threshold.
                                         0.65,    # IoU Threshold.
                                         nc=1,   # Number of classes.
                                         nkpt=17, # Number of keypoints.
                                         kpt_label=True)
        
        output = output_to_keypoint(output)

    # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    nimg = img[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        
    nimg = cv2.resize(nimg, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return nimg, fps


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv7 pose estimation on a video.")
    parser.add_argument(
        "--video-file",
        default="../media/skydiving.mp4",
        help="Path to the input video.",
    )
    parser.add_argument(
        "--weights",
        default="yolov7-w6-pose.pt",
        help="Path to the YOLOv7 pose weights file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path. Defaults to <input-name>_yolov7.avi.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Forward pass input size.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display frames while processing. Omit this on headless systems.",
    )
    return parser.parse_args()


args = parse_args()
SCRIPT_DIR = Path(__file__).resolve().parent

#------------------------------------------------------------------------------#
# Change forward pass input size.
input_size = args.input_size

#---------------------------INITIALIZATIONS------------------------------------#

# Select the device based on hardware configs.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('Selected Device : ', device)

# Load keypoint detection model.
weights_path = Path(args.weights)
if not weights_path.exists() and not weights_path.is_absolute():
    weights_path = SCRIPT_DIR / weights_path
if not weights_path.exists():
    raise FileNotFoundError(
        f"YOLOv7 pose weights not found: {weights_path}. "
        "Download yolov7-w6-pose.pt and place it in the yolov7 directory, "
        "or pass its path with --weights."
    )

weights = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
model = weights['model']
# Load the model in evaluation mode.
_ = model.float().eval()
# Load the model to computation device [cpu/gpu/tpu]
model.to(device)

vid_path = Path(args.video_file)
if not vid_path.exists() and not vid_path.is_absolute():
    vid_path = SCRIPT_DIR / vid_path
if not vid_path.exists():
    raise FileNotFoundError(f"Input video not found: {vid_path}")

save_name = vid_path.stem
output_path = Path(args.output) if args.output else Path(f"{save_name}_yolov7.avi")
cap = cv2.VideoCapture(str(vid_path))
fps = int(cap.get(cv2.CAP_PROP_FPS))
ret, frame = cap.read()
if not ret:
    raise RuntimeError(f"Unable to read the first frame from {vid_path}")
out = None

#-------------------------------------------------------------------------------#


if __name__ == '__main__':
    print(f"Processing {vid_path}")
    print(f"Saving output to {output_path}")
    while ret:
        img, fps_ = pose_video(frame)

        cv2.putText(img, 'FPS : {:.2f}'.format(fps_), (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(img, 'YOLOv7', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        if out is None:
            out_h, out_w = img.shape[:2]
            out = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc('M','J','P','G'),
                fps if fps > 0 else 10,
                (out_w, out_h),
            )
            if not out.isOpened():
                raise RuntimeError(f"Unable to create output video writer for {output_path}")

        out.write(img[...,::-1])
        if args.show:
            cv2.imshow('Output', img[...,::-1])
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        ret, frame = cap.read()

    cap.release()
    if out is not None:
        out.release()
    if args.show:
        cv2.destroyAllWindows()
