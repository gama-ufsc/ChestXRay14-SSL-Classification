from train_binary import train
from ssl_prediction import filter_predictions, make_predictions


def get_best_model(paths_list):
    best_auroc = 0
    best_model = None
    best_epoch = None
    best_path = None
    for path in paths_list:
        with open(path + 'model_auroc.txt', 'r') as file:
            text = file.read()
            best_at_path = max(text.split('\n'), key=lambda line: float(line.split(',')[1]) if line != "" else 0)
            epoch, auroc, model = best_at_path.split(',')
            if float(auroc) > float(best_auroc):
                best_auroc = auroc
                best_model = model
                best_epoch = epoch
                best_path = path
    print(f"Found best model at {best_path} : {best_model}")
    return best_epoch, best_auroc, best_model




## Run tests with k_best=0.75 and optimized augmentations
for ratio in [2, 5, 10, 20]:
    try:
        ratiostr = f"{ratio if len(str(ratio))==2 else '0'+str(ratio)}"
        teacher_folder = f'./runs/effusion_train{ratiostr}%_teacher_00/'
        pseudolabels_folder = f'./runs/effusion_train{ratiostr}%_pseudolabels_00/'
        finetune_folder = f'./runs/effusion_train{ratiostr}%_finetune_00/'

        ssl_predictions = f'./labels/binary_Effusion/filtered_predictions/teacher_{ratiostr}%_00/ssl_prediction_{100 - ratio}%.txt'
        ssl_filtered = f'./labels/binary_Effusion/filtered_predictions/teacher_{ratiostr}%_00/filtered_{100 - ratio}%.txt'

        labeled_images = f'./labels/binary_Effusion/train{ratiostr}%.txt'
        unlabeled_images = f'./labels/binary_Effusion/train{100 - ratio}%.txt'

        train(CKPT_PATH=None,
              CHOSEN_CLASS=2,
              TRAIN_IMAGE_LIST=labeled_images,
              EPOCHS=15,
              LR=1e-4,
              LR_STEP=0.90,
              VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
              RUN_NAME=teacher_folder,
              LOADER_TYPE='strong')

        _, _, best_teacher_model = get_best_model([teacher_folder])
        make_predictions(model_path=best_teacher_model,
                         labels_file=unlabeled_images,
                         output_file=ssl_predictions)

        filter_predictions(ssl_predictions,
                           labels_file=unlabeled_images,
                           output_file=ssl_filtered,
                           data_class=2,
                           k_best=0.75)

        train(CKPT_PATH=None,
              CHOSEN_CLASS=2,
              TRAIN_IMAGE_LIST=ssl_filtered,
              EPOCHS=10,
              LR=1e-4,
              LR_STEP=0.90,
              VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
              RUN_NAME=pseudolabels_folder,
              LOADER_TYPE='strong')

        _, _, best_pseudolabels_model = get_best_model([pseudolabels_folder])
        for loader_type in ['strong', 'original']:  # + strong
            finetune_folder = f'./runs/effusion_train{ratiostr}%_finetune_{loader_type}_00/'

            train(CKPT_PATH=best_pseudolabels_model,
                  CHOSEN_CLASS=2,
                  TRAIN_IMAGE_LIST=labeled_images,
                  EPOCHS=15,
                  LR=5e-5,
                  LR_STEP=0.9,
                  VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
                  RUN_NAME=finetune_folder,
                  LOADER_TYPE=loader_type)
    except Exception as e:
        print(e)
        pass





labeled_images = f'./labels/binary_Effusion/train10%.txt'
ssl_predictions = f'./labels/binary_Effusion/filtered_predictions/teacher_10%_augmentations_tests/ssl_prediction_90%.txt'
ssl_filtered    = f'./labels/binary_Effusion/filtered_predictions/teacher_10%_augmentations_tests/filtered_90%.txt'
unlabeled_images = f'./labels/binary_Effusion/train90%.txt'

# -------------------------- Teste Teacher

# for loader_type in ['strong', 'medium_transforms', 'original', 'random_crop', 'strong_random_crop']:
#     teacher_folder = f'./runs/effusion_train10%_teacher_augmentation_{loader_type}/'
#
#     train(CKPT_PATH=None,
#           CHOSEN_CLASS=2,
#           TRAIN_IMAGE_LIST=labeled_images,
#           EPOCHS=10,
#           LR=1e-4,
#           LR_STEP=0.95,
#           VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#           RUN_NAME=teacher_folder,
#           LOADER_TYPE=loader_type)



# -------------------------- Teste Augmentations
teacher_folders_list = []
for loader_type in ['strong', 'medium_transforms', 'original', 'random_crop', 'strong_random_crop']:
    teacher_folder = f'./runs/effusion_train10%_teacher_augmentation_{loader_type}/'
    teacher_folders_list.append(teacher_folder)



# _, _, best_teacher_model = get_best_model(teacher_folders_list)
# make_predictions(model_path=best_teacher_model,
#                  labels_file=unlabeled_images,
#                  output_file=ssl_predictions)
#
# filter_predictions(ssl_predictions,
#                    labels_file=unlabeled_images,
#                    output_file=ssl_filtered,
#                    data_class=2,
#                    k_best=0.75)

# for loader_type in ['strong', 'medium_transforms', 'original', 'random_crop', 'strong_random_crop']:
#     pseudolabels_folder = f'./runs/effusion_train10%_pseudolabels_augmentation_{loader_type}/'#
#     train(CKPT_PATH=None,
#           CHOSEN_CLASS=2,
#           TRAIN_IMAGE_LIST=ssl_filtered,
#           EPOCHS=6,
#           LR=5e-5,
#           LR_STEP=0.90,
#           VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
#           RUN_NAME=pseudolabels_folder,
#           LOADER_TYPE=loader_type)
# #


## -------------------------- Testes student
pseudolabels_folders_list = []
for loader_type in ['strong', 'medium_transforms', 'original', 'random_crop', 'strong_random_crop']: #strong orignal
    pseudolabels_folder = f'./runs/effusion_train10%_pseudolabels_augmentation_{loader_type}/'
    pseudolabels_folders_list.append(pseudolabels_folder)

_, _, best_pseudolabels_model = get_best_model(pseudolabels_folders_list)
for loader_type in ['strong', 'medium_transforms', 'original']: # + strong
    finetune_folder = f'./runs/effusion_train10%_finetune_augmentation_{loader_type}/'

    train(CKPT_PATH=best_pseudolabels_model,
          CHOSEN_CLASS=2,
          TRAIN_IMAGE_LIST=labeled_images,
          EPOCHS=10,
          LR=5e-5,
          LR_STEP=0.9,
          VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
          RUN_NAME=finetune_folder,
          LOADER_TYPE=loader_type)


