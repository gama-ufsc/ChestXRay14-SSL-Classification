from train_binary import train
from ssl_prediction import filter_predictions, make_predictions


def get_best_model(path):
    with open(path, 'r') as file:
        text = file.read()
        best_model = max(text.split('\n'), key=lambda line: float(line.split(',')[1]) if line != "" else 0)
        epoch, auroc, model = best_model.split(',')
        print(f"Found best model at {path} : {best_model}")
        return epoch, auroc, model




ratio = 10
ratiostr = f"{ratio if len(str(ratio))==2 else '0'+str(ratio)}"
teacher_folder = f'./runs/effusion_train{ratiostr}%_teacher_00/'
pseudolabels_folder = f'./runs/effusion_train{ratiostr}%_pseudolabels_04/'
finetune_folder = f'./runs/effusion_train{ratiostr}%_finetune_04/'

ssl_predictions = f'./labels/binary_Effusion/filtered_predictions/teacher_{ratiostr}%_02/ssl_prediction_{100 - ratio}%.txt'
ssl_filtered = f'./labels/binary_Effusion/filtered_predictions/teacher_{ratiostr}%_02/filtered_{100 - ratio}%.txt'

labeled_images = f'./labels/binary_Effusion/train{ratiostr}%.txt'
unlabeled_images = f'./labels/binary_Effusion/train{100 - ratio}%.txt'

# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST=labeled_images,
#       EPOCHS=25,
#       LR=1e-4,
#       LR_STEP=0.95,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=teacher_folder,
#       LOADER_TYPE='original')
#
# _, _, best_teacher_model = get_best_model(teacher_folder + 'model_auroc.txt')
# make_predictions(model_path=best_teacher_model,
#                  labels_file=unlabeled_images,
#                  output_file=ssl_predictions)
#
# filter_predictions(ssl_predictions,
#                    labels_file=unlabeled_images,
#                    output_file=ssl_filtered,
#                    data_class=2,
#                    k_best=0.75)

train(CKPT_PATH=None,
      CHOSEN_CLASS=2,
      TRAIN_IMAGE_LIST=ssl_filtered,
      EPOCHS=10,
      LR=1e-4,
      LR_STEP=0.90,
      VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
      RUN_NAME=pseudolabels_folder,
      LOADER_TYPE='strong')

_, _, best_pseudolabels_model = get_best_model(pseudolabels_folder + 'model_auroc.txt')
train(CKPT_PATH=best_pseudolabels_model,
      CHOSEN_CLASS=2,
      TRAIN_IMAGE_LIST=labeled_images,
      EPOCHS=20,
      LR=5e-5,
      LR_STEP=0.95,
      VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
      RUN_NAME=finetune_folder,
      LOADER_TYPE='original')




### Run tests with k_best=0.75 and strong augmentations
# for ratio in [2, 5, 10, 20]:
#     try:
#         ratiostr = f"{ratio if len(str(ratio))==2 else '0'+str(ratio)}"
#         teacher_folder = f'./runs/effusion_train{ratiostr}%_teacher_00/'
#         pseudolabels_folder = f'./runs/effusion_train{ratiostr}%_pseudolabels_00/'
#         finetune_folder = f'./runs/effusion_train{ratiostr}%_finetune_00/'
#
#         ssl_predictions = f'./labels/binary_Effusion/filtered_predictions/teacher_{ratiostr}%_00/ssl_prediction_{100 - ratio}%.txt'
#         ssl_filtered = f'./labels/binary_Effusion/filtered_predictions/teacher_{ratiostr}%_00/filtered_{100 - ratio}%.txt'
#
#         labeled_images = f'./labels/binary_Effusion/train{ratiostr}%.txt'
#         unlabeled_images = f'./labels/binary_Effusion/train{100 - ratio}%.txt'
#
#         # train(CKPT_PATH=None,
#         #       CHOSEN_CLASS=2,
#         #       TRAIN_IMAGE_LIST=labeled_images,
#         #       EPOCHS=25,
#         #       LR=1e-4,
#         #       LR_STEP=0.95,
#         #       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#         #       RUN_NAME=teacher_folder,
#         #       LOADER_TYPE='strong')
#
#         # _, _, best_teacher_model = get_best_model(teacher_folder + 'model_auroc.txt')
#         # make_predictions(model_path=best_teacher_model,
#         #                  labels_file=unlabeled_images,
#         #                  output_file=ssl_predictions)
#
#         filter_predictions(ssl_predictions,
#                            labels_file=unlabeled_images,
#                            output_file=ssl_filtered,
#                            data_class=2,
#                            k_best=0.75)
#
#         train(CKPT_PATH=None,
#               CHOSEN_CLASS=2,
#               TRAIN_IMAGE_LIST=ssl_filtered,
#               EPOCHS=10,
#               LR=1e-4,
#               LR_STEP=0.90,
#               VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#               RUN_NAME=pseudolabels_folder,
#               LOADER_TYPE='strong')
#
#         _, _, best_pseudolabels_model = get_best_model(pseudolabels_folder + 'model_auroc.txt')
#         train(CKPT_PATH=best_pseudolabels_model,
#               CHOSEN_CLASS=2,
#               TRAIN_IMAGE_LIST=labeled_images,
#               EPOCHS=15,
#               LR=1e-4,
#               LR_STEP=0.95,
#               VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#               RUN_NAME=finetune_folder,
#               LOADER_TYPE='strong')
#     except Exception as e:
#         print(e)
#         pass

### Test K = 0.25, 0.5, 0.75
# for K in [0.1, 0.25, 0.5, 0.75, 0.9, 1 ]:
#     try:
#         run_label = str(K).replace('.', '')
#         teacher_folder = './runs/effusion_train10%_teacher_00/'
#         pseudolabels_folder = f'./runs/effusion_train10%_pseudolabels_k{run_label}/'
#
#         ssl_predictions = f'./labels/binary_Effusion/filtered_predictions/teacher_10%_00/ssl_prediction_90%.txt'
#         ssl_filtered = f'./labels/binary_Effusion/filtered_predictions/teacher_10%_00/filtered_90%_k{run_label}.txt'
#
#         unlabeled_images = './labels/binary_Effusion/train90%.txt'
#
#         filter_predictions(ssl_predictions,
#                            labels_file=unlabeled_images,
#                            output_file=ssl_filtered,
#                            data_class=2,
#                            k_best=K)
#
#         train(CKPT_PATH=None,
#               CHOSEN_CLASS=2,
#               TRAIN_IMAGE_LIST=ssl_filtered,
#               EPOCHS=10,
#               LR=1e-4,
#               LR_STEP=0.90,
#               VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#               RUN_NAME=pseudolabels_folder,
#               LOADER_TYPE='strong')
#
#
#     except Exception as e:
#         print(f"training failed fork={K}")
#         print(e)
#         pass

###full train
# try:
#     teacher_folder = './runs/effusion_train_full/'
#     labeled_images = './labels/binary_Effusion/train.txt'
#     train(CKPT_PATH=None,
#           CHOSEN_CLASS=2,
#           TRAIN_IMAGE_LIST=labeled_images,
#           EPOCHS=25,
#           LR=1e-4,
#           LR_STEP=0.95,
#           VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#           RUN_NAME=teacher_folder,
#           LOADER_TYPE='strong')
# except Exception as e:
#     print(e)
#     pass

# ### Test for 20%
# try:
#       teacher_folder = './runs/effusion_train20%_teacher_00/'
#       pseudolabels_folder = './runs/effusion_train20%_pseudolabels_00/'
#       finetune_folder = './runs/effusion_train20%_finetune_00/'
#
#       ssl_predictions = './labels/binary_Effusion/filtered_predictions/teacher_20%_00/ssl_prediction_80%.txt'
#       ssl_filtered = './labels/binary_Effusion/filtered_predictions/teacher_20%_00/filtered_80%.txt'
#
#       labeled_images = './labels/binary_Effusion/train20%.txt'
#       unlabeled_images = './labels/binary_Effusion/train80%.txt'
#
#       # train(CKPT_PATH=None,
#       #       CHOSEN_CLASS=2,
#       #       TRAIN_IMAGE_LIST=labeled_images,
#       #       EPOCHS=25,
#       #       LR=1e-4,
#       #       LR_STEP=0.95,
#       #       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       #       RUN_NAME=teacher_folder,
#       #       LOADER_TYPE='strong')
#
#
#       _, _, best_teacher_model = get_best_model(teacher_folder + 'model_auroc.txt')
#       make_predictions(model_path=best_teacher_model,
#                        labels_file=unlabeled_images,
#                        output_file=ssl_predictions)
#
#       filter_predictions(ssl_predictions,
#                          labels_file=unlabeled_images,
#                          output_file=ssl_filtered,
#                          data_class=2)
#
#
#
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST=ssl_filtered,
#             EPOCHS=10,
#             LR=1e-4,
#             LR_STEP=0.90,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=pseudolabels_folder,
#             LOADER_TYPE='strong')
#
#       _, _, best_pseudolabels_model = get_best_model(pseudolabels_folder + 'model_auroc.txt')
#       train(CKPT_PATH=best_pseudolabels_model,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST=labeled_images,
#             EPOCHS=15,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=finetune_folder,
#             LOADER_TYPE='strong')
# except:
#       pass


### Test for 10%
# try:
#       teacher_folder = './runs/effusion_train10%_teacher_00/'
#       pseudolabels_folder = './runs/effusion_train10%_pseudolabels_01/'
#       finetune_folder = './runs/effusion_train10%_finetune_01/'
#
#       ssl_predictions = './labels/binary_Effusion/filtered_predictions/teacher_10%_00/ssl_prediction_90%.txt'
#       ssl_filtered = './labels/binary_Effusion/filtered_predictions/teacher_10%_00/filtered_90%.txt'
#
#       labeled_images = './labels/binary_Effusion/train10%.txt'
#       unlabeled_images = './labels/binary_Effusion/train90%.txt'

# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST=labeled_images,
#       EPOCHS=25,
#       LR=1e-4,
#       LR_STEP=0.95,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=teacher_folder,
#       LOADER_TYPE='strong')


# _, _, best_teacher_model = get_best_model(teacher_folder + 'model_auroc.txt')
# make_predictions(model_path=best_teacher_model,
#                  labels_file=unlabeled_images,
#                  output_file=ssl_predictions)
#
# filter_predictions(ssl_predictions,
#                    labels_file=unlabeled_images,
#                    output_file=ssl_filtered,
#                    data_class=2)
#

#
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST=ssl_filtered,
#             EPOCHS=10,
#             LR=1e-4,
#             LR_STEP=0.90,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=pseudolabels_folder,
#             LOADER_TYPE='strong')
#
#       _, _, best_pseudolabels_model = get_best_model(pseudolabels_folder + 'model_auroc.txt')
#       train(CKPT_PATH=best_pseudolabels_model,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST=labeled_images,
#             EPOCHS=15,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=finetune_folder,
#             LOADER_TYPE='strong')
# except Exception as e:
#       print(e)
#       pass
#
#
# ### Test for 05%
# try:
#       teacher_folder = './runs/effusion_train05%_teacher_00/'
#       pseudolabels_folder = './runs/effusion_train05%_pseudolabels_01/'
#       finetune_folder = './runs/effusion_train05%_finetune_01/'
#
#       ssl_predictions = './labels/binary_Effusion/filtered_predictions/teacher_05%_00/ssl_prediction_95%.txt'
#       ssl_filtered = './labels/binary_Effusion/filtered_predictions/teacher_05%_00/filtered_95%.txt'
#
#       labeled_images = './labels/binary_Effusion/train05%.txt'
#       unlabeled_images = './labels/binary_Effusion/train95%.txt'

# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST=labeled_images,
#       EPOCHS=25,
#       LR=1e-4,
#       LR_STEP=0.95,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=teacher_folder,
#       LOADER_TYPE='strong')


# _, _, best_teacher_model = get_best_model(teacher_folder + 'model_auroc.txt')
# make_predictions(model_path=best_teacher_model,
#                  labels_file=unlabeled_images,
#                  output_file=ssl_predictions)
#
# filter_predictions(ssl_predictions,
#                    labels_file=unlabeled_images,
#                    output_file=ssl_filtered,
#                    data_class=2)
#


#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST=ssl_filtered,
#             EPOCHS=10,
#             LR=1e-4,
#             LR_STEP=0.90,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=pseudolabels_folder,
#             LOADER_TYPE='strong')
#
#       _, _, best_pseudolabels_model = get_best_model(pseudolabels_folder + 'model_auroc.txt')
#       train(CKPT_PATH=best_pseudolabels_model,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST=labeled_images,
#             EPOCHS=15,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=finetune_folder,
#             LOADER_TYPE='strong')
# except Exception as e:
#       print(e)
#       pass


# #### Test for 02%
# teacher_folder = './runs/effusion_train02%_teacher_00/'
# pseudolabels_folder = './runs/effusion_train02%_pseudolabels_00/'
# finetune_folder = './runs/effusion_train02%_finetune_00/'
#
# ssl_predictions = './labels/binary_Effusion/filtered_predictions/teacher_02%_00/ssl_prediction_98%.txt'
# ssl_filtered = './labels/binary_Effusion/filtered_predictions/teacher_02%_00/filtered_98%.txt'
#
# labeled_images = './labels/binary_Effusion/train02%.txt'
# unlabeled_images = './labels/binary_Effusion/train98%.txt'
#
# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST=labeled_images,
#       EPOCHS=25,
#       LR=1e-4,
#       LR_STEP=0.95,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=teacher_folder,
#       LOADER_TYPE='strong')
#
#
# _, _, best_teacher_model = get_best_model(teacher_folder + 'model_auroc.txt')
# make_predictions(model_path=best_teacher_model,
#                  labels_file=unlabeled_images,
#                  output_file=ssl_predictions)
#
# filter_predictions(ssl_predictions,
#                    labels_file=unlabeled_images,
#                    output_file=ssl_filtered,
#                    data_class=2)
#
#
#
# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST=ssl_filtered,
#       EPOCHS=10,
#       LR=1e-4,
#       LR_STEP=0.90,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=pseudolabels_folder,
#       LOADER_TYPE='strong')
#
# _, _, best_pseudolabels_model = get_best_model(pseudolabels_folder + 'model_auroc.txt')
# train(CKPT_PATH=best_pseudolabels_model,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST=labeled_images,
#       EPOCHS=15,
#       LR=1e-4,
#       LR_STEP=0.95,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=finetune_folder,
#       LOADER_TYPE='strong')
#
#


#
#
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train05%.txt',
#             EPOCHS=25,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train05%_teacher_00/',
#             LOADER_TYPE='strong')
# except Exception as e:
#       print(e)
#       print("Erro at strong")
#
#
#
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#             EPOCHS=25,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train10%_teacher_00/',
#             LOADER_TYPE='strong')
# except Exception as e:
#       print(e)
#       print("Erro at strong")
#
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train20%.txt',
#             EPOCHS=25,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train20%_teacher_00/',
#             LOADER_TYPE='strong')
# except Exception as e:
#       print(e)
#       print("Erro at strong")
# #
#
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#             EPOCHS=2,
#             LR=5e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train10%_teacher_evaluate_lr_strong_01/',
#             LOADER_TYPE='strong')
# except Exception as e:
#       print(e)
#       print("Erro at strong")
#
#
#
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#             EPOCHS=2,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train10%_teacher_evaluate_lr_strong_02/',
#             LOADER_TYPE='strong')
# except Exception as e:
#       print(e)
#       print("Erro at strong")
#
#
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#             EPOCHS=2,
#             LR=5e-5,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train10%_teacher_evaluate_lr_strong_03/',
#             LOADER_TYPE='strong')
# except Exception as e:
#       print(e)
#       print("Erro at strong")

"""Evaluate image augmentation for teacher"""
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#             EPOCHS=30,
#             LR=1e-3,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train10%_teacher_evaluate_augmentation_fivecropped/',
#             LOADER_TYPE='five_cropped',
#             TRAIN_BATCH_SIZE=4)
# except Exception as e:
#       print(e)
#       print("Erro at fivecrop")

# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#             EPOCHS=30,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train10%_teacher_evaluate_augmentation_original2/',
#             LOADER_TYPE='original')
# except Exception as e:
#       print(e)
#       print("Erro at original")
#
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#             EPOCHS=30,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train10%_teacher_evaluate_augmentation_original2/',
#             LOADER_TYPE='original')
# except Exception as e:
#       print(e)
#       print("Erro at original")

#
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#             EPOCHS=30,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train10%_teacher_evaluate_augmentation_strong8_90/',
#             LOADER_TYPE='strong')
# except Exception as e:
#       print(e)
#       print("Erro at strong")
#
#
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#             EPOCHS=30,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train10%_teacher_evaluate_augmentation_strong9_90/',
#             LOADER_TYPE='strong')
# except Exception as e:
#       print(e)
#       print("Erro at strong")
#
#
# try:
#       train(CKPT_PATH=None,
#             CHOSEN_CLASS=2,
#             TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#             EPOCHS=30,
#             LR=1e-4,
#             LR_STEP=0.95,
#             VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#             RUN_NAME=f'effusion_train10%_teacher_evaluate_augmentation_original_3/',
#             LOADER_TYPE='original')
# except Exception as e:
#       print(e)
#       print("Erro at strong")

# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=30,
#       LR=1e-4,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_teacher_evaluate_augmentation_fivecropped2/',
#       LOADER_TYPE='five_cropped',
#       TRAIN_BATCH_SIZE=4)


"""Evaluate learning rate for finetuning """
# train(CKPT_PATH='/media/roberto/external/mestrado_backup/runs/effusion_train10%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=2,
#       LR=1e-3,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_finetune_evaluate_lr_00/')

# train(CKPT_PATH='/media/roberto/external/mestrado_backup/runs/effusion_train10%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=2,
#       LR=1e-4,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_finetune_evaluate_lr_01/')


# train(CKPT_PATH='/media/roberto/external/mestrado_backup/runs/effusion_train10%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=2,
#       LR=5e-5,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_finetune_evaluate_lr_02/')


# train(CKPT_PATH='/media/roberto/external/mestrado_backup/runs/effusion_train10%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=2,
#       LR=1e-5,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_finetune_evaluate_lr_03/')

# train(CKPT_PATH='/media/roberto/external/mestrado_backup/runs/effusion_train10%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=2,
#       LR=5e-6,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_finetune_evaluate_lr_04/')


"""Evaluate learning rate for pseudolabels """
# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='labels/binary_Effusion/filtered_predictions/teacher_10%_01/filtered_90%.txt',
#       EPOCHS=2,
#       LR=1e-3,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_pseudo_evaluate_lr_00/')


# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='labels/binary_Effusion/filtered_predictions/teacher_10%_01/filtered_90%.txt',
#       EPOCHS=2,
#       LR=5e-4,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_pseudo_evaluate_lr_01/')

# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='labels/binary_Effusion/filtered_predictions/teacher_10%_01/filtered_90%.txt',
#       EPOCHS=2,
#       LR=1e-4,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_pseudo_evaluate_lr_02/')

# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='labels/binary_Effusion/filtered_predictions/teacher_10%_01/filtered_90%.txt',
#       EPOCHS=2,
#       LR=5e-5,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_pseudo_evaluate_lr_03/')


"""Evaluate learning rate for teacher"""
#
# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=2,
#       LR=1e-3,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_teacher_evaluate_lr_00/')
#
#
# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=2,
#       LR=5e-4,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_teacher_evaluate_lr_01/')
#
#
# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=2,
#       LR=1e-4,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_teacher_evaluate_lr_02/')
#
#
# train(CKPT_PATH=None,
#       CHOSEN_CLASS=2,
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=2,
#       LR=5e-5,
#       LR_STEP=0.9,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_teacher_evaluate_lr_03/')
#


# train(CKPT_PATH='runs/effusion_train02%_student_pseudolabels_00/densenet_model_weights_epoch5.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train02%.txt',
#       EPOCHS_FINETUNE=20,
#       LR_FINETUNIG=1e-4,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train02%_student_finetune_00/')
#


# train(CHOSEN_CLASS=2,
#       TRAIN_PSEUDO_IMAGE_LIST='./labels/binary_Effusion/filtered_predictions/teacher_02%_00/filtered_98%.txt',
#       EPOCHS_PSEUDO=15,
#       LR_PSEUDO=1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train02%_student_pseudolabels_00/')
# #
# filter_predictions('./labels/binary_Effusion/filtered_predictions/teacher_02%_00/ssl_prediction_98%.txt',
#                    labels_file='./labels/binary_Effusion/train98%.txt',
#                    output_file='./labels/binary_Effusion/filtered_predictions/teacher_02%_00/filtered_98%.txt',
#                    data_class=2)


#
# make_predictions(model_path='./runs/effusion_train02%_teacher_00/densenet_model_weights_epoch9.pth',
#                  labels_file='./labels/binary_Effusion/train98%.txt',
#                  output_file='./labels/binary_Effusion/filtered_predictions/teacher_02%_00/ssl_prediction_98%.txt')
#


# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train02%.txt',
#       EPOCHS=25,
#       LR = 1e-4,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train02%_teacher_00/')


# train(CKPT_PATH='runs/effusion_train05%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train05%.txt',
#       EPOCHS_FINETUNE=20,
#       LR_FINETUNIG=1e-4,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train05%_student_finetune_00/')
#
#
# train(CKPT_PATH='runs/effusion_train05%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train05%.txt',
#       EPOCHS_FINETUNE=20,
#       LR_FINETUNIG=1e-4,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train05%_student_finetune_01/')
#
#
# train(CKPT_PATH='runs/effusion_train05%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train05%.txt',
#       EPOCHS_FINETUNE=20,
#       LR_FINETUNIG=1e-4,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train05%_student_finetune_02/')
#
#
#
# train(CHOSEN_CLASS=2,
#       TRAIN_PSEUDO_IMAGE_LIST='./runs/effusion_train05%_teacher_00/pseudo_labels95_filtered.txt',
#       EPOCHS_PSEUDO=15,
#       LR_PSEUDO=1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train05%_student_pseudolabels_00/')
#
#
#
# train(CHOSEN_CLASS=2,
#       TRAIN_PSEUDO_IMAGE_LIST='./runs/effusion_train05%_teacher_00/pseudo_labels95_filtered.txt',
#       EPOCHS_PSEUDO=15,
#       LR_PSEUDO=1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train05%_student_pseudolabels_01/')
#
#
#
# train(CHOSEN_CLASS=2,
#       TRAIN_PSEUDO_IMAGE_LIST='./runs/effusion_train05%_teacher_00/pseudo_labels95_filtered.txt',
#       EPOCHS_PSEUDO=15,
#       LR_PSEUDO=1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train05%_student_pseudolabels_02/')
#
#
#
#
# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train.txt',
#       EPOCHS=25,
#       LR = 1e-4,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_full_train_02/')
#


# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train05%.txt',
#       EPOCHS=25,
#       LR = 1e-4,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train05%_teacher_00/')
#
#
# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train05%.txt',
#       EPOCHS=25,
#       LR = 1e-4,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train05%_teacher_01/')
#
#
# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train05%.txt',
#       EPOCHS=25,
#       LR = 1e-4,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train05%_teacher_02/')
#
#
#
# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train.txt',
#       EPOCHS=25,
#       LR = 5e-4,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_full_train_00/')


#
# train(CKPT_PATH='runs/effusion_train20%_student_pseudolabels_00/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train20%.txt',
#       EPOCHS_FINETUNE=20,
#       LR_FINETUNIG=1e-4,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train20%_student_finetune_03/')
#
#
# train(CKPT_PATH='runs/effusion_train20%_student_pseudolabels_00/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train20%.txt',
#       EPOCHS_FINETUNE=20,
#       LR_FINETUNIG=1e-3,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train20%_student_finetune_04/')
#
#
#
#
#
#
#
# train(CHOSEN_CLASS=2,
#       TRAIN_PSEUDO_IMAGE_LIST='./runs/effusion_train10%_teacher_01/filtered_90%.txt',
#       EPOCHS_PSEUDO=15,
#       LR_PSEUDO=1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_student_pseudolabels_02/')
#
#
#
# train(CHOSEN_CLASS=2,
#       TRAIN_PSEUDO_IMAGE_LIST='./runs/effusion_train10%_teacher_01/filtered_90%.txt',
#       EPOCHS_PSEUDO=15,
#       LR_PSEUDO=1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_student_pseudolabels_03/')
#
#
#
#
#
# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train20%.txt',
#       EPOCHS=25,
#       LR = 1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train20%_teacher_03/')
#
#
#


#
#
# train(CKPT_PATH='runs/effusion_train10%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS_FINETUNE=20,
#       LR_FINETUNIG=1e-4,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_student_finetune_00/')
#
# train(CKPT_PATH='runs/effusion_train10%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS_FINETUNE=20,
#       LR_FINETUNIG=1e-4,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_student_finetune_01/')
#
#
# train(CKPT_PATH='runs/effusion_train10%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS_FINETUNE=20,
#       LR_FINETUNIG=1e-5,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_student_finetune_02/')
#
# train(CKPT_PATH='runs/effusion_train10%_student_pseudolabels_01/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS_FINETUNE=20,
#       LR_FINETUNIG=1e-3,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_student_finetune_03/')

# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=20,
#       LR = 1e-4,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_teacher_02/')

#
# train(CHOSEN_CLASS=2,
#
#       TRAIN_PSEUDO_IMAGE_LIST='./runs/effusion_train10%_teacher_01/filtered_90%.txt',
#       EPOCHS_PSEUDO=15,
#       LR_PSEUDO=1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_student_pseudolabels_00/')
#


# train(CHOSEN_CLASS=2,
#
#       TRAIN_PSEUDO_IMAGE_LIST='./runs/effusion_train10%_teacher_01/filtered_90%.txt',
#       EPOCHS_PSEUDO=15,
#       LR_PSEUDO=1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_student_pseudolabels_03/')
#
# train(CHOSEN_CLASS=2,
#
#       TRAIN_PSEUDO_IMAGE_LIST='./runs/effusion_train10%_teacher_01/filtered_90%.txt',
#       EPOCHS_PSEUDO=15,
#       LR_PSEUDO=1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_student_pseudolabels_03/')


#
# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=20,
#       LR = 1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_teacher_00/')

#
# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train10%.txt',
#       EPOCHS=20,
#       LR = 1e-4,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train10%_teacher_01/')
#
#


# train(CHOSEN_CLASS=2,
#       CKPT_PATH = './runs/effusion_train20%_teacher_00/densenet_model_weights_epoch15.pth'
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train20%.txt',
#       EPOCHS=25,
#       LR = 1e-3,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train20%_teacher_00/')
#
#


# train(CHOSEN_CLASS=2,
#       CKPT_PATH = '',
#       TRAIN_IMAGE_LIST='./labels/binary_Effusion/train.txt',
#       EPOCHS=25,
#       LR = 1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_full_train_01/')
#


# train(CKPT_PATH='runs/effusion_train20%_student_pseudolabels_00/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train20%.txt',
#       EPOCHS_FINETUNE=10,
#       LR_FINETUNIG=1e-5,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train20%_student_finetune_01/')
#
#
# train(CKPT_PATH='runs/effusion_train20%_student_pseudolabels_00/densenet_model_weights_epoch6.pth',
#       CHOSEN_CLASS=2,
#       TRAIN_FINETUNE_IMAGE_LIST='./labels/binary_Effusion/train20%.txt',
#       EPOCHS_FINETUNE=10,
#       LR_FINETUNIG=1e-3,
#
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train20%_student_finetune_02/')
#
# train(CHOSEN_CLASS=2,
#
#       TRAIN_PSEUDO_IMAGE_LIST='./runs/effusion_train20%_teacher_00/pseudo_labels80_filtered.txt',
#       EPOCHS_PSEUDO=10,
#       LR_PSEUDO=1e-3,
#       VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#       RUN_NAME=f'effusion_train20%_student_pseudolabels_01/')
#


#
