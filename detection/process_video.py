#!/bin/python

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

from moviepy.editor import VideoFileClip
from detection.road import Road
from detection.file import full_path
from detection.config import DetectionConfig

def process_video(video_name):
    """
  	process a video and save output. Designed to be executed from the command line.
    :param video_name: String, name of video to process. Should be found in input_videos folder
    """
	print("Processing your video!")
	clip = VideoFileClip(full_path("input_videos/" + video_name))

	road = Road(config=DetectionConfig(classifier_name="LinearSVC"))
	
	new_clip = clip.fl_image(road.process_image)
	new_clip.write_videofile(full_path("output_videos/" + video_name), audio=False)

	print("Finished processing your video!")
	print("Processed", road.frame_count, "images.")

# I cheat here and execute the script by calling its name from the command line
process_video("project_video.mp4")
