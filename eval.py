import os
import argparse
import torch
import numpy as np
from pprint import pprint
from torch.utils.data import DataLoader

from model.OhioModel import OhioModel
from utils.OhioDataset import prepare_personal_data


def run_eval_mape(model, data_loader, opt):
    """ Get MAPE score on test set.
    """
    input_len = opt.left_len
    model.eval()

    mape_sum = 0
    n = 0

    for i, example in enumerate(data_loader):
        batch_size, _, _ = example.shape
        example = example
        output = model.forward(example, input_len)

        y_hat_batch = output[:, -1, 0].detach().numpy() * data_loader.dataset.std[0] + data_loader.dataset.mean[0]
        y_gt_batch = example[:, -1, 0].detach().numpy() * data_loader.dataset.std[0] + data_loader.dataset.mean[0]

        absolute_errors = np.abs(y_hat_batch - y_gt_batch)
        actual_values = y_gt_batch

        mape_sum += np.sum(absolute_errors / actual_values)
        n += len(actual_values)

    mape = (mape_sum / n) * 100

    return mape

def run_eval_rmse(model, data_loader, opt):
    """ Get RMSE score on test set.
    """
    input_len = opt.left_len
    model.eval()

    y_hat = []
    y_gt = []

    for i, example in enumerate(data_loader):
        batch_size, _, _ = example.shape
        example = example
        output = model.forward(example, input_len)

        y_hat_batch = output[:, -1, 0] * data_loader.dataset.std[0] + data_loader.dataset.mean[0]
        y_gt_batch = example[:, -1, 0] * data_loader.dataset.std[0] + data_loader.dataset.mean[0]

        y_hat.extend(y_hat_batch.tolist())
        y_gt.extend(y_gt_batch.tolist())
    y_hat = np.array(y_hat)
    y_gt = np.array(y_gt)

    return np.sqrt(np.mean((y_hat - y_gt) ** 2))


def main(ckpt):
    opt = ckpt["opt"]

    train_csv_path = os.path.join(opt.data_dir, "{}_train.csv".format(opt.patient))
    test_csv_path = os.path.join(opt.data_dir, "{}_test.csv".format(opt.patient))
    train_dataset, test_dataset = prepare_personal_data(train_csv_path, test_csv_path, opt.left_len + opt.missing_len, opt.unimodal)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=False)
    model = OhioModel(
        d_in=test_dataset[0].shape[1],
        num_layers=opt.num_layers,
        d_model=opt.d_model,
        heads=opt.heads,
        d_ff=opt.d_ff,
        dropout=opt.dropout,
        attention_dropout=opt.attention_dropout,
        single_pred=opt.single_pred
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # model.cuda()
    test_score_rmse = run_eval_rmse(model, test_loader, opt)
    test_score_mape = run_eval_mape(model, test_loader, opt)
    print("Patient {} test score RMSE: {:.3f}".format(opt.patient, test_score_rmse))
    print("Patient {} test score MAPE: {:.3f}\n".format(opt.patient, test_score_mape))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts_dir", type=str, default="./pretrained/set1_30",
                             help="The directory (folder) to trained models.")
    eval_opt = parser.parse_args()

    for patient in ["588", "559", "563", "570", "575", "591"]:
        ckpt = torch.load(os.path.join(eval_opt.ckpts_dir, "{}.ckpt".format(patient)), map_location=torch.device('cpu'))
        pprint(vars(ckpt["opt"]), indent=4)
        main(ckpt)