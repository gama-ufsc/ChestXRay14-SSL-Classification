from train_binary import train
from ssl_prediction import filter_predictions, make_predictions


def get_best_model(path):
    with open(path, 'r') as file:
        text = file.read()
        best_model = max(text.split('\n'), key=lambda line: float(line.split(',')[1]) if line != "" else 0)
        epoch, auroc, model = best_model.split(',')
        print(f"Found best model at {path} : {best_model}")
        return epoch, auroc, model




## Setup run configurations
ratio = 10
ratiostr = f"{ratio if len(str(ratio))==2 else '0'+str(ratio)}"
# Define output folders for logs
teacher_folder = f'./runs/effusion_train{ratiostr}%_teacher_00/'
pseudolabels_folder = f'./runs/effusion_train{ratiostr}%_pseudolabels_04/'
finetune_folder = f'./runs/effusion_train{ratiostr}%_finetune_04/'
# Define files for the pseudolabels lists
ssl_predictions = f'./labels/binary_Effusion/filtered_predictions/teacher_{ratiostr}%_02/ssl_prediction_{100 - ratio}%.txt'
ssl_filtered = f'./labels/binary_Effusion/filtered_predictions/teacher_{ratiostr}%_02/filtered_{100 - ratio}%.txt'
# Define what labels are going to be used
labeled_images = f'./labels/binary_Effusion/train{ratiostr}%.txt'
unlabeled_images = f'./labels/binary_Effusion/train{100 - ratio}%.txt'

# Train the teacher model
train(CKPT_PATH=None,
      CHOSEN_CLASS=2,
      TRAIN_IMAGE_LIST=labeled_images,
      EPOCHS=25,
      LR=1e-4,
      LR_STEP=0.95,
      VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
      RUN_NAME=teacher_folder,
      LOADER_TYPE='original')

# Make pseudolabels predictions
_, _, best_teacher_model = get_best_model(teacher_folder + 'model_auroc.txt')
make_predictions(model_path=best_teacher_model,
                 labels_file=unlabeled_images,
                 output_file=ssl_predictions)
# Filter the pseudolabels predictions
filter_predictions(ssl_predictions,
                   labels_file=unlabeled_images,
                   output_file=ssl_filtered,
                   data_class=2,
                   k_best=0.75)

# Train student model with pseudolabels
train(CKPT_PATH=None,
      CHOSEN_CLASS=2,
      TRAIN_IMAGE_LIST=ssl_filtered,
      EPOCHS=10,
      LR=1e-4,
      LR_STEP=0.90,
      VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
      RUN_NAME=pseudolabels_folder,
      LOADER_TYPE='strong')

# Finetune student model with labeled data
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




## Run tests with k_best=0.75 and strong augmentations
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
              EPOCHS=25,
              LR=1e-4,
              LR_STEP=0.95,
              VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
              RUN_NAME=teacher_folder,
              LOADER_TYPE='strong')

        _, _, best_teacher_model = get_best_model(teacher_folder + 'model_auroc.txt')
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

        _, _, best_pseudolabels_model = get_best_model(pseudolabels_folder + 'model_auroc.txt')
        train(CKPT_PATH=best_pseudolabels_model,
              CHOSEN_CLASS=2,
              TRAIN_IMAGE_LIST=labeled_images,
              EPOCHS=15,
              LR=1e-4,
              LR_STEP=0.95,
              VAL_IMAGE_LIST='./labels/binary_Effusion/val.txt',
              RUN_NAME=finetune_folder,
              LOADER_TYPE='strong')
    except Exception as e:
        print(e)
        pass
