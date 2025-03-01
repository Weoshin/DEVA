import os
import torch
import numpy as np
from tqdm import tqdm
from opts import *
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed
from tensorboardX import SummaryWriter
from models.deva import build_model
from core.metric import MetricsTop
import pickle
import time
opt = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device: {}:{}".format(device, opt.CUDA_VISIBLE_DEVICES))


train_mae, val_mae = [], []


def save_pkl(path, obj):
    pickle_file = open(path, 'wb')
    pickle.dump(obj, pickle_file)
    pickle_file.close()
    print("保存成功")

def load_pkl(path):
    pickle_file = open(path, 'rb')
    obj = pickle.load(pickle_file)
    pickle_file.close()
    print("读取成功")
    return obj

def main():
    opt = parse_opts()
    if opt.seed is not None:
        setup_seed(opt.seed)
    print("seed: {}".format(opt.seed))
    
    log_path = os.path.join(".", "log", opt.project_name)
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    print("log_path :", log_path)

    save_path = os.path.join(opt.models_save_root,  opt.project_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    print("model_save_path :", save_path)

    model = build_model(opt).to(device)

    dataLoader = MMDataLoader(opt)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, opt)
    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)

    writer = SummaryWriter(logdir=log_path)


    for epoch in range(1, opt.n_epochs+1):
        show_list = []
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, writer, metrics)
        evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, writer, save_path, metrics)
        if opt.is_test is not None:
            pre_dict = test(model, dataLoader['test'], optimizer, loss_fn, epoch, writer, metrics)
            show_list.append(pre_dict)
        scheduler_warmup.step()
    writer.close()


def train(model, train_loader, optimizer, loss_fn, epoch, writer, metrics):
    train_pbar = tqdm(enumerate(train_loader))
    losses = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    for cur_iter, data in train_pbar:
        img, audio, text, audio_text, vision_text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['audio_text'].to(device), data['vision_text'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        # output, _, _, _, _, _, _, _, _  = model(img, audio, text, audio_text, vision_text)
        output = model(img, audio, text, audio_text, vision_text)
        loss = loss_fn(output, label)

        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                'loss': '{:.5f}'.format(losses.value_avg),
                                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)
    print('train: ', train_results)
    train_mae.append(train_results['MAE'])

    writer.add_scalar('train/loss', losses.value_avg, epoch)



def evaluate(model, eval_loader, optimizer, loss_fn, epoch, writer, save_path, metrics):
    test_pbar = tqdm(enumerate(eval_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text, audio_text, vision_text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['audio_text'].to(device), data['vision_text'].to(device)
            
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text, audio_text, vision_text)

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)

        writer.add_scalar('evaluate/loss', losses.value_avg, epoch)
        # if epoch == 28:
        #     save_model(save_path, epoch, model, optimizer)


def test(model, test_loader, optimizer, loss_fn, epoch, writer, metrics):

    start_time = time.time()

    test_pbar = tqdm(enumerate(test_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []
    # h_at_list = []
    # h_a_list = []
    # h_fusion_a_list = []
    # h_vt_list = []
    # h_v_list = []
    # h_fusion_v_list = []
    # h_t_list = []
    # h_fusion_t_list = []
    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text, audio_text, vision_text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['audio_text'].to(device), data['vision_text'].to(device)
            raw_text, id, labels = data['raw_text'], data['id'], data['labels']['M'].to(device).view(-1, 1)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            # output, h_at, h_a, h_fusion_a, h_vt, h_v, h_fusion_v, h_t, h_fusion_t = model(img, audio, text, audio_text, vision_text)
            output = model(img, audio, text, audio_text, vision_text)
            print(id)
            print(output.view(1, -1))
            print(label.view(1, -1))
            print("----")

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())
            # h_at_list.append(h_at.cpu())
            # h_a_list.append(h_a.cpu())
            # h_fusion_a_list.append(h_fusion_a.cpu())
            # h_vt_list.append(h_vt.cpu())
            # h_v_list.append(h_v.cpu())
            # h_fusion_v_list.append(h_fusion_v.cpu())
            # h_t_list.append(h_t.cpu())
            # h_fusion_t_list.append(h_fusion_t.cpu())
            # x_list.append(x.cpu())
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")  

        writer.add_scalar('test/loss', losses.value_avg, epoch)

        # present_dict = {'h_at':h_at_list, 'h_a':h_a_list, 'h_fusion_a': h_fusion_a_list, 
        #                 'h_vt':h_vt_list, 'h_v': h_v_list, 'h_fusion_v': h_fusion_v_list, 
        #                 'h_t': h_t_list, 'h_fusion_t':h_fusion_t_list}

        # return present_dict


if __name__ == '__main__':
    main()
