# Aplicacao de Aprendizado Semi-Supervisionado (SSL)para classificacao de imagens medicas (CheXNet)

# Utilizacao
O treinamento do modelo e feito pelo scrpt ```run.py```

O treinamento e feito em tres etapas, como descrito no artigo:
- Treinamento do modelo Teacher
- Avaliacao do conjunto nao classificado e criacao de pseudolabels
- Treinamento do modelo Student (pseudolabels + finetune)


## Dataset

[ChestX-ray14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) 

[Download](https://nihcc.app.box.com/v/ChestXray-NIHCC)