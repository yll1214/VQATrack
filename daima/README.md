# VQATrack



## Install the environment
Use the Anaconda
```
conda create -n LiteTrack python=3.9
conda activate LiteTrack
pip install -r requirements.txt
```


## Evaluation

Put the downloaded weights on ```<PROJECT_ROOT>/checkpoints/train/LiteTrack/baseline_base```.

Notably, the modality of target reference (NL, BBOX or NLBBOX) is specified in config ```TEST.MODE```

```


# Evaluation
open airsim

python run-air.py


```

## Run LiteTrack on your own video
Specify the target by bounding box or natural language, which should keep consistent with ```TEST.MODE``` in config.
```
python demo.py baseline_base \
                   <input video path> \
                   <output video path> \
                   <language description of target> \
                   <initial bbox of target: x y w h>
```


## Contact
For questions about our paper or code, please contact [Lingling Yang(yanglingling1214@163.com)

## Acknowledgments
* Thanks for JointNLT and UVLTrack Library, which helps us to quickly implement our ideas.

* We use the implementation of the ViT from the [Timm](https://github.com/huggingface/pytorch-image-models) repo and BERT from the [pytorch_pretrained_bert](https://github.com/Meelfy/pytorch_pretrained_BERT).


