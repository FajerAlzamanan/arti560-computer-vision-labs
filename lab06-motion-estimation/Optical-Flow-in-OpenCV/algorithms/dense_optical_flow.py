import os
import cv2
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SAVE_AT_FRAME = 30  # save a representative frame after this many frames

# Map algorithm method objects to result filenames
_METHOD_NAMES = {
    "calcOpticalFlowSparseToDense": "LucasKanade_Dense",
    "calcOpticalFlowFarneback": "Farneback",
    "calcOpticalFlowDenseRLOF": "RLOF",
}


def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # Determine output filename from the method name
    method_name = getattr(method, "__name__", "dense_optical_flow")
    result_name = _METHOD_NAMES.get(method_name, method_name)

    # read the video
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Save a representative frame to results/
        frame_idx += 1
        if frame_idx == SAVE_AT_FRAME:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            out_path = os.path.join(RESULTS_DIR, f"{result_name}.png")
            cv2.imwrite(out_path, bgr)
            print(f"[Saved] {out_path}")

        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame
