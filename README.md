# Recurrent Back-Projection Network for Video Super-Resolution (CVPR2019)

Project page: https://alterzero.github.io/projects/RBPN.html

## Dependencies
* Python 3.5
* PyTorch >= 1.0.0
* Pyflow -> https://github.com/pathak22/pyflow

## Dataset
* [Vimeo-90k Dataset](http://toflow.csail.mit.edu)

## HOW TO

#Training

    ```python
    main.py
    ```

#Testing

    ```python
    eval.py
    ```

![RBPN](https://alterzero.github.io/projects/RBPN.png)

## Our other paper for Image Super-Resolution
## [Deep Back-Projection Networks for Super-Resolution (CVPR2018)](https://github.com/alterzero/DBPN-Pytorch)
### Winner (1st) of [NTIRE2018](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Timofte_NTIRE_2018_Challenge_CVPR_2018_paper.pdf) Competition (Track: x8 Bicubic Downsampling)
### Winner of [PIRM2018](https://arxiv.org/pdf/1809.07517.pdf) (1st on Region 2, 3rd on Region 1, and 5th on Region 3)
### Project page: https://alterzero.github.io/projects/DBPN.html


## Citations
If you find this work useful, please consider citing it.
```
@inproceedings{RBPN2019,
  title={Recurrent Back-Projection Network for Video Super-Resolution},
  author={Haris, Muhammad and Shakhnarovich, Greg and Ukita, Norimichi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
