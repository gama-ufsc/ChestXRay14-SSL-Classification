from train_binary import train, test
from ssl_prediction import filter_predictions, make_predictions

import os
def get_best_model(path):
    with open(path, 'r') as file:
        text = file.read()
        best_model = max(text.split('\n'), key=lambda line: float(line.split(',')[1]) if line != "" else 0)
        epoch, auroc, model = best_model.split(',')
        print(f"Found best model at {path} : {best_model}")
        return epoch, auroc, model





runsdir = '/media/roberto/external/mestrado_backup/runs k=0.75 strong/'






try:
    teacher = runsdir+'effusion_train_full/densenet_model_weights_epoch18.pth'


    test(CKPT_PATH=teacher ,
          CHOSEN_CLASS=2,
          TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
          RUN_NAME=os.path.join(os.path.dirname(teacher), 'teste_results'),
          )




except Exception as e:
    print(e)
    pass

#
#
# ###############################################################
# ###############################################################
# ###############################################################
# try:
#     teacher = runsdir + 'effusion_train20%_teacher_00/densenet_model_weights_epoch11.pth'
#     pseudo = runsdir + 'effusion_train20%_pseudolabels_00/densenet_model_weights_epoch7.pth'
#     finetune = runsdir + 'effusion_train20%_finetune_00/densenet_model_weights_epoch4.pth'
#
#     test_images = f'./labels/binary_Effusion/test.txt'
#
#     # Run train for 0 epochs to evaluate test set
#
#
#     # test(CKPT_PATH=teacher ,
#     #       CHOSEN_CLASS=2,
#     #       TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#     #       RUN_NAME=os.path.join(os.path.dirname(teacher), 'teste_results'),
#     #       )
#
#     test(CKPT_PATH=pseudo,
#          CHOSEN_CLASS=2,
#          TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#          RUN_NAME=os.path.join(os.path.dirname(pseudo), 'teste_results'),
#          )
#
#     # test(CKPT_PATH=finetune,
#     #      CHOSEN_CLASS=2,
#     #      TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#     #      RUN_NAME=os.path.join(os.path.dirname(finetune), 'teste_results'),
#     #      )
#
#
# except Exception as e:
#     print(e)
#     pass
#
# ###############################################################
# ###############################################################
# ###############################################################
# try:
#     teacher = runsdir + 'effusion_train10%_teacher_00/densenet_model_weights_epoch11.pth'
#     pseudo = runsdir + 'effusion_train10%_pseudolabels_00/densenet_model_weights_epoch9.pth'
#     finetune = runsdir + 'effusion_train10%_finetune_00/densenet_model_weights_epoch4.pth'
#
#     test_images = f'./labels/binary_Effusion/test.txt'
#
#     # Run train for 0 epochs to evaluate test set
#
#
#     test(CKPT_PATH=teacher ,
#           CHOSEN_CLASS=2,
#           TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#           RUN_NAME=os.path.join(os.path.dirname(teacher), 'teste_results'),
#           )
#     test(CKPT_PATH=pseudo,
#          CHOSEN_CLASS=2,
#          TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#          RUN_NAME=os.path.join(os.path.dirname(pseudo), 'teste_results'),
#          )
#     test(CKPT_PATH=finetune,
#          CHOSEN_CLASS=2,
#          TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#          RUN_NAME=os.path.join(os.path.dirname(finetune), 'teste_results'),
#          )
#
#
# except Exception as e:
#     print(e)
#     pass
#
#
# ###############################################################
# ###############################################################
# ###############################################################
# try:
#     teacher = runsdir + 'effusion_train05%_teacher_00/densenet_model_weights_epoch9.pth'
#     pseudo = runsdir + 'effusion_train05%_pseudolabels_00/densenet_model_weights_epoch9.pth'
#     finetune = runsdir + 'effusion_train05%_finetune_00/densenet_model_weights_epoch4.pth'
#
#     test_images = f'./labels/binary_Effusion/test.txt'
#
#     # Run train for 0 epochs to evaluate test set
#
#
#     test(CKPT_PATH=teacher ,
#           CHOSEN_CLASS=2,
#           TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#           RUN_NAME=os.path.join(os.path.dirname(teacher), 'teste_results'),
#           )
#     test(CKPT_PATH=pseudo,
#          CHOSEN_CLASS=2,
#          TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#          RUN_NAME=os.path.join(os.path.dirname(pseudo), 'teste_results'),
#          )
#     test(CKPT_PATH=finetune,
#          CHOSEN_CLASS=2,
#          TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#          RUN_NAME=os.path.join(os.path.dirname(finetune), 'teste_results'),
#          )
#
#
# except Exception as e:
#     print(e)
#     pass
#
#
# ###############################################################
# ###############################################################
# ###############################################################
# try:
#     teacher = runsdir + 'effusion_train02%_teacher_00/densenet_model_weights_epoch11.pth'
#     pseudo = runsdir + 'effusion_train02%_pseudolabels_00/densenet_model_weights_epoch6.pth'
#     finetune = runsdir + 'effusion_train02%_finetune_00/densenet_model_weights_epoch3.pth'
#
#     test_images = f'./labels/binary_Effusion/test.txt'
#
#     # Run train for 0 epochs to evaluate test set
#
#
#     test(CKPT_PATH=teacher ,
#           CHOSEN_CLASS=2,
#           TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#           RUN_NAME=os.path.join(os.path.dirname(teacher), 'teste_results'),
#           )
#     test(CKPT_PATH=pseudo,
#          CHOSEN_CLASS=2,
#          TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#          RUN_NAME=os.path.join(os.path.dirname(pseudo), 'teste_results'),
#          )
#     test(CKPT_PATH=finetune,
#          CHOSEN_CLASS=2,
#          TEST_IMAGE_LIST='./labels/binary_Effusion/test.txt',
#          RUN_NAME=os.path.join(os.path.dirname(finetune), 'teste_results'),
#          )
#
#
# except Exception as e:
#     print(e)
#     pass
