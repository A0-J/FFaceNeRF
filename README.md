## Install Environment via Anaconda (Recommended)
    conda env create -f environment.yml
    conda activate ffacenerf

## Pretrained Model & Dataset

FFaceNeRF requires NeRFFaceEditing checkpoints for initialization

put [pretrained_model](https://drive.google.com/file/d/1N4y3leKEF7rbMVNbpYUYtNnaO4WVDln1/view?usp=drive_link) into networks/NeRFFaceEditing-ffhq-64.pkl

Download [Data](https://drive.google.com/file/d/16ha-UeU2uLZu7YNYPXw-I1yIHyav2E0O/view?usp=drive_link) for training and testing

## Install Dependencies
    pip install opencv-python requests mrcfile tqdm tensorboard imageio matplotlib scikit-learn ninja

## StyleGAN3 & EG3D Setup
    cd ~
    git clone https://github.com/NVlabs/stylegan3.git
    git clone https://github.com/NVlabs/eg3d.git

    # copy dependencies
    cp -r ~/stylegan3/dnnlib ~/FFaceNeRF/dnnlib
    cp ~/eg3d/eg3d/camera_utils.py ~/FFaceNeRF/camera_utils.py

    # set environment path (current session)
    export PYTHONPATH=$PYTHONPATH:/data/users/ksh/stylegan3

    # set environment path permanently
    echo 'export PYTHONPATH=$PYTHONPATH:/data/users/ksh/stylegan3' >> ~/.bashrc
    source ~/.bashrc

    # cp 안될경우
    wget https://raw.githubusercontent.com/NVlabs/stylegan3/main/camera_utils.py

## Install PyTorch (CUDA 12.1)
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

## Training
    python train_ffacenerf.py --mode eyes
    #python train_ffacenerf.py --mode nose
    #python train_ffacenerf.py --mode chin

## Evaluation
    python evaluate.py --network networks/ckpt_eyes_10.pth

## Visualize
    python visualize_results.py --id 70





---

# FFaceNeRF
### [CVPR2025] FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields

### [[Project Page](https://kwanyun.github.io/FFaceNeRF_page/)] [[Paper](https://arxiv.org/abs/2503.08417)]

![teaser](https://github.com/user-attachments/assets/b51980f8-29ae-46ec-a572-6700ae0462ae)

### Be aware current version still need some correction and clean-ups. If there are any suggestion for environment setting, let us know for future users



## :gear: Install Environment via Anaconda (Recommended)
    conda env create -f environment.yml
    conda activate ffacenerf


FFaceNeRF requires NeRRFaceEditing checkpoints for intialization

put [pretrained_model](https://drive.google.com/file/d/1N4y3leKEF7rbMVNbpYUYtNnaO4WVDln1/view?usp=drive_link) into networks/NeRFFaceEditing-ffhq-64.pkl

Download [Data](https://drive.google.com/file/d/16ha-UeU2uLZu7YNYPXw-I1yIHyav2E0O/view?usp=drive_link) for training and testing

## Training requires about 40 minutes on single A6000 GPU
    python train_ffacenerf.py --mode eyes
    #python train_ffacenerf.py --mode nose
    #python train_ffacenerf.py --mode chin


## Testing
    python editing_testset.py --mode eyes --network ckpt_eyes_10.pth --overlap_weight 0.5
    #python editing_testset.py --mode eyes --network ckpt_nose_10.pth --overlap_weight 0.4 --target_image 64



In the original CVPR paper, the test set comprised 22 samples; in this public repository, we expanded it to 41 for the testing.

    python evaluate.py

#### We would like to thank EG3D and NeRFFaceEditing for open-source video interpolation model

## Citation
```
@inproceedings{yun2025ffacenerf,
  title={FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields},
  author={Yun, Kwan and Kim, Chaelin and Shin, Hangyeul and Noh, Junyong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={10825--10835},
  year={2025}
}
```
