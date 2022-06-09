# Shot Detect

The algorithm aims to pick key-frames of a video.

## Requirement

Make sure you have [python](http://www.python.org/) 2.7+ and 
[opencv](https://github.com/opencv) 3.3+, 

## Usage

```Python
from shotdetect import shotDetector

video_path = '/Your_video_dir/test.mp4'
detector = shotDetector(video_path)
detector.run()
detector.pick_frame('/The_Dir_To_Save_Frames', "File_name_prefix")
```

Then you can get files in the directory you have specified above. These files are in format 'File\_name\_prefix@starttime\-endtime.jpg'.

If you use opencv2, then you must change 'cv2.cv.CV\_CAP\_PROP\_FRAME\_COUNT' and 'cv2.cv.CV\_CAP\_PROP\_FPS' to 'cv2.CAP\_PROP\_FRAME\_COUNT' and 'cv2.CAP\_PROP\_FPS' respectively.

