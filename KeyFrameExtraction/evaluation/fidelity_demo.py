import numpy as np
from fidelity import *
from main import *

if __name__ == '__main__':

    print(sys.argv[1])
    print("Opening video")
    cap = cv2.VideoCapture(os.path.abspath(os.path.expanduser(sys.argv[1])))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame count: " + str(frame_count))
    summary = random_summary(96, frame_count)
    path = sys.argv[1]
    fid = fidelity(summary, frame_count, path, True)
    print("Fidelity: " + str(fid))
    #extract_from_indices(summary, frame_count, sys.argv[1])