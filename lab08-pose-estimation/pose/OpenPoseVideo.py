import cv2
import time
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--video_file", default="../media/skydiving.mp4", help="Input Video")
parser.add_argument(
    "--output",
    default=None,
    help="Output video path. Defaults to <input-name>_openpose.avi.",
)
parser.add_argument(
    "--show",
    action="store_true",
    help="Display frames while processing. Omit this on headless systems.",
)

args = parser.parse_args()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODE = "MPI"

if MODE == "COCO":
    protoFile = os.path.join(SCRIPT_DIR, "coco", "pose_deploy_linevec.prototxt")
    weightsFile = os.path.join(SCRIPT_DIR, "coco", "pose_iter_440000.caffemodel")
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE == "MPI" :
    protoFile = os.path.join(SCRIPT_DIR, "mpi", "pose_deploy_linevec_faster_4_stages.prototxt")
    weightsFile = os.path.join(SCRIPT_DIR, "mpi", "pose_iter_160000.caffemodel")
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


inWidth = 368
inHeight = 368
threshold = 0.1


input_source = args.video_file
if not os.path.exists(input_source) and not os.path.isabs(input_source):
    input_source = os.path.normpath(os.path.join(SCRIPT_DIR, input_source))
if not os.path.exists(input_source):
    raise FileNotFoundError(f"Input video not found: {input_source}")

if not os.path.exists(protoFile):
    raise FileNotFoundError(f"OpenPose prototxt not found: {protoFile}")
if not os.path.exists(weightsFile):
    raise FileNotFoundError(
        f"OpenPose weights not found: {weightsFile}. "
        "Download the required .caffemodel file listed in the lab README."
    )

cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()
if not hasFrame:
    raise RuntimeError(f"Unable to read the first frame from {input_source}")

save_name = os.path.splitext(os.path.basename(input_source))[0]
output_path = args.output or f"{save_name}_openpose.avi"
print(f"Processing {input_source}")
print(f"Saving output to {output_path}")
vid_writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc('M','J','P','G'),
    cap.get(cv2.CAP_PROP_FPS) or 10,
    (frame.shape[1],frame.shape[0]),
)
if not vid_writer.isOpened():
    raise RuntimeError(f"Unable to create output video writer for {output_path}")

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

while hasFrame:
    t = time.time()
    frameCopy = np.copy(frame)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    if args.show:
        cv2.imshow('Output-Keypoints', frameCopy)
        cv2.imshow('Output-Skeleton', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    vid_writer.write(frame)
    hasFrame, frame = cap.read()

vid_writer.release()
cap.release()
if args.show:
    cv2.destroyAllWindows()
