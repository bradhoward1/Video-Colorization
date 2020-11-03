# Video Colorization

This repository contains software that aims to colorize black and white videos. Courtesy of Richard Zhang (https://github.com/richzhang/colorization), we 
have obtained a machine learning algorithm that is able to colorize black and white images without any user input. Our goal is to break down videos into their
individual frames, apply this algorithm on each frame, and acquire a recolorized video as a result. Depending on the progress of the project, we may need to 
utililze segmentation methods in order to better colorize individual components of each frame. 


## Use (Up to Date)

To demo the video colorization software we have created, follow these steps:
1. Clone this repository locally
2. Enter a Python Virtual Environment
3. Install requirements.txt packages using pip
4. Utilize the following format when running the program `video_color.py`:

      `python3 video_color.py -vid vids/JTVid.mp4`
   
   Other available videos in the `vids` directory include `LightBoxVid.mov` and `OldVideo.mp4`. To run the algorithm on these videos, simply replace `JTVid.mp4` in the file path above with the name of the file you want to view. You may run this with files of your own as well.


## Limitations

  There are a few limitations that we still need to solve. One is the memory storage aspect of the project. The resultant video
is a few magnitudes larger in storage space than the input video. Also, right now we store each frame from the input video as well
as each frame from the resultant video. This alone is a lot of memory, so we need to figure out what we want to do with these
images.

  Furthermore, right now we do not have the audio with the video. This is a minor detail but it would be interesting to be able to 
keep the audio along with the video.

  Finally, another limitation we face is timing. It takes the algorithm multiple seconds to run on each extracted frame, so an 8 second video could easily take a 
  few minutes to finish. We will work on speeding up the process if possible.

  We will continue to make further innovations as well, so stay tuned.


## Citations

For the colorizers packages, as well as some of the base code used within our demo, we would like to cite Richard Zhang's team and all of their hard work. Thank you for your amazing contributions!

```
@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, Phillip and Geng, Xinyang and Lin, Angela S and Yu, Tianhe and Efros, Alexei A},
  journal={ACM Transactions on Graphics (TOG)},
  volume={9},
  number={4},
  year={2017},
  publisher={ACM}
}
```
