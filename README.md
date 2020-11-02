# Video Colorization
This repository contains software that aims to colorize black and white videos. Courtesy of Richard Zhang (https://github.com/richzhang/colorization), we 
have obtained a machine learning algorithm that is able to colorize black and white images without any user input. Our goal is to break down videos into their
individual frames, apply this algorithm on each frame, and acquire a recolorized video as a result. Depending on the progress of the project, we may need to 
utililze segmentation methods in order to better colorize individual components of each frame. 

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
