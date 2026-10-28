import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import torch.optim as optim
import torch_dct as dct
import time
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.net import CONSFormer
from utils.opt import Options
from utils.dataloader15 import Data
from utils.metrics import FDE, JPE, APE



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model, batch_data, opt):
    input_seq, output_seq = batch_data
    B, N, _, D = input_seq.shape
    input_ = input_seq.view(-1, 15, input_seq.shape[-1])
    output_ = output_seq.view(output_seq.shape[0] * output_seq.shape[1], -1, input_seq.shape[-1])
    output_seq = output_seq[:, :, :45,  :]


    offset = input_[:, 1:15, :] - input_[:, :14, :]  # dispacement sequence
    src = dct.dct(offset)
    offset_out=output_[:,1:46,:]-output_[:,:45,:]
    src_gt=dct.dct(offset_out)

    rec_ ,rec_gt= model.forward(src,src_gt, N)
    rec = dct.idct(rec_)
    rec_gt=dct.idct(rec_gt)
    results = output_[:, :1, :]
    results_gt = output_[:, :1, :]
    for i in range(1, 46):
        results = torch.cat(
            [results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)],
            dim=1)
        results_gt = torch.cat(
        [results_gt, output_[:, :1, :] + torch.sum(rec_gt[:, :i, :], dim=1, keepdim=True)],
        dim=1)
    results = results[:, 1:, :]  # 3 15 45
    results_gt = results_gt[:, 1:, :] 
    gt_poses1=output_seq
    pred_poses1 = results.reshape(B, N, 45, 15, -1)
    gt_poses1 = gt_poses1.reshape(B, N, 45, 15, -1)
    pred_poses_gt = results_gt.reshape(B, N, 45, 15, -1)
    pd1 = pred_poses1.reshape(B * N, 45, -1)
    gt1 = gt_poses1.reshape(B * N, 45, -1)
    pd1_gt = pred_poses_gt.reshape(B * N, 45, -1)
    loss1 = torch.mean(torch.norm(pd1 - gt1, dim=2))
    loss2 = torch.mean(torch.norm(pd1_gt - gt1, dim=2))
    loss3=torch.mean(torch.norm(pd1 - pd1_gt, dim=2))

    prediction = results.view(B, N, -1, 15, 3)

    gt = output_.view(B, N, -1, 15, 3)[:, :, 1:, ...]

    return prediction, gt, loss1,loss2,loss3, results


def processor(opt):
    device = opt.device

    setup_seed(opt.seed)
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    dataset = Data(dataset='mupots', mode=0, device=device, transform=True, opt=opt)
    test_dataset = Data(dataset='mupots', mode=1, device=device, transform=False, opt=opt)

    print(stamp)
    dataloader = DataLoader(dataset,
                            batch_size=opt.train_batch,
                            shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.test_batch,
                                 shuffle=False, drop_last=True)

    model0 = CONSFormer(input_dim=opt.d_model, d_model=opt.d_model,
                      d_inner=opt.d_inner, n_layers=opt.num_stage,
                      n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, dropout=opt.dropout, device=device,
                      kernel_size=opt.kernel_size, opt=opt).to(device)
    model = torch.nn.DataParallel(model0, device_ids=[0, 1])

    print(">>> training params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))

    Evaluate = True
    save_model = True
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           lr=opt.lr)

    loss_min = 100
    for epoch_i in range(1, opt.epochs + 1):
        with torch.autograd.set_detect_anomaly(True):
            model.train()
        loss_list = []
        test_loss_list = []
        """
        ==================================
           Training Processing
        ==================================
        """
        for _, batch_data in tqdm(enumerate(dataloader)):
            _, _, loss1,loss2,loss3, _ = train(model, batch_data, opt)
            loss_all=loss1*(epoch_i/opt.epochs)+loss2*(1-(epoch_i/opt.epochs))+loss3
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            loss_list.append(loss1.item())

        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch_i
        }

        loss_cur = np.mean(loss_list)
        print('epoch:', epoch_i, 'loss:', loss_cur, "lr: {:.10f} ".format(optimizer.param_groups[0]['lr']))
        if save_model:
            # if (epoch_i + 1) % 5 == 0:
            save_path = os.path.join('checkpointslg', f'epoch_{epoch_i}.model')
            torch.save(checkpoint, save_path)

        frame_idx = [15, 30, 45]
        n = 0
        ape_err_total = np.arange(len(frame_idx), dtype=np.float_)
        jpe_err_total = np.arange(len(frame_idx), dtype=np.float_)
        fde_err_total = np.arange(len(frame_idx), dtype=np.float_)

        if Evaluate:
            with torch.no_grad():
                """
                  ==================================
                     Validating Processing
                  ==================================
                  """
                model.eval()
                print("\033[0:35mEvaluating.....\033[m")
                for _, batch_data in tqdm(enumerate(test_dataloader)):
                    n += 1
                    prediction, gt, test_loss,_,_, _ = train(model, batch_data, opt)
                    test_loss_list.append(test_loss.item())

                    ape_err = APE(gt, prediction, frame_idx)
                    jpe_err = JPE(gt, prediction, frame_idx)
                    fde_err = FDE(gt, prediction, frame_idx)

                    ape_err_total += ape_err
                    jpe_err_total += jpe_err
                    fde_err_total += fde_err

                test_loss_cur = np.mean(test_loss_list)

                if test_loss_cur < loss_min:
                    save_path = os.path.join('checkpointslg', f'best_epoch.model')
                    torch.save(checkpoint, save_path)
                    loss_min = test_loss_cur
                    print(f"Best epoch_{checkpoint['epoch']} model is saved!")

                print("{0: <16} | {1:6d} | {2:6d} | {3:6d} ".format("Lengths", 1000, 2000, 3000))
                print("=== JPE Test Error ===")
                print(
                    "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} ".format("Our", jpe_err_total[0] / n,
                                                                        jpe_err_total[1] / n,
                                                                        jpe_err_total[2] / n))
                print("=== APE Test Error ===")
                print(
                    "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} ".format("Our", ape_err_total[0] / n,
                                                                        ape_err_total[1] / n,
                                                                        ape_err_total[2] / n))
                print("=== FDE Test Error ===")
                print(
                    "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} ".format("Our", fde_err_total[0] / n,
                                                                        fde_err_total[1] / n,
                                                                        fde_err_total[2] / n))


if __name__ == '__main__':
    option = Options().parse()
    processor(option)





