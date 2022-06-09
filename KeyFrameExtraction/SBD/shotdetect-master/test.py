from shotdetect import shotDetector

video_path = 'C:\\Users\\Leo\\Desktop\\Edit.mp4'
detector = shotDetector(video_path)
detector.run()
detector.pick_frame('C:\\Users\\Leo\\Desktop\\ABCCC', "File_name_prefix")