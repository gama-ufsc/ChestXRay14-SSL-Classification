import os
import torch
from sklearn.metrics import roc_auc_score
import progressbar


def evaluate_model(model, val_loader, chosen_class, total_time, epoch, writer, logdir):
    with torch.no_grad():
        gt = torch.FloatTensor()
        gt = gt.cuda()
        pred = torch.FloatTensor()
        pred = pred.cuda()

        for i, (inp, target) in progressbar.progressbar(enumerate(val_loader), max_value=val_loader.__len__()):
            target = target[:, chosen_class].reshape([-1, 1])
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)
        writer.add_pr_curve(f'pr_curve/', gt, pred, epoch)
        AUROC = roc_auc_score(gt[:, 0].cpu(), pred[:, 0].cpu())
        print(f'The AUROC at epoch {epoch + 1} is {AUROC:.3f}')

        writer.add_scalar(f'auroc/validation', AUROC, epoch)
        writer.add_scalar(f'training_time', total_time, epoch)

        model_path = os.path.join(logdir + f'densenet_model_weights_epoch{epoch + 1}.pth')
        torch.save(model.state_dict(),
                   model_path
                   )
        writer.flush()

        with open(os.path.join(logdir, 'model_auroc.txt'), 'a+') as file:
            file.write(f"{epoch},{AUROC},{model_path}\r\n")


