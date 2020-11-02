
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import sys

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('-vid', '--vid_path', type=str, default='vids/LightBoxVid.mov');
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())



# Read in a video from files
input_video = sys.argv[2]
vid_read = cv2.VideoCapture("{}".format(input_video))
split_video_name = input_video.split(".")
split_video_name = split_video_name[0].split("/")
isolated_video_name = split_video_name[1]


# Make a directory for all frames within the video
if not os.path.exists("vid_result_{}".format(isolated_video_name)):
	os.makedirs("vid_result_{}".format(isolated_video_name))

# Initialize frame number
frame_number = 0
frame_list = []


# Extract images from input video file
while(True):
	video_left, frame_count = vid_read.read()
	
	if video_left:
		frame_name = './vid_result_{}/frame'.format(isolated_video_name) + str(frame_number) + '.jpg'
		frame_list.append(frame_name)
		print("Creating {}".format(frame_name))
		frame_number += 1
		cv2.imwrite(frame_name, frame_count)
	else:
		break




plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')
plt.show()
