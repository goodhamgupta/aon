# AON-tensorflow

Tensorflow implementation of [AON: Towards Arbitrarily-Oriented Text Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cheng_AON_Towards_Arbitrarily-Oriented_CVPR_2018_paper.pdf) that extracts feature sequences in four directions and combines them into an attention-based decoder to generate character sequence.

# Pretrained Model

The model has been trained to 390000 iterations on a p2.xlarge instance. The training process took about two days. The checkpoints for the model are available [here](https://drive.google.com/file/d/1fol58YSoxrErbkQI-d8zpnOSH4WyigVy/view?usp=sharing)

# Credits

The code in the repository was made by the fantastic Hui Zhang and you can find the original repository [here](https://github.com/huizhang0110/AON). 