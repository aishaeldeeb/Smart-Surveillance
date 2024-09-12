import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os


def test(dataloader, model, mode, args, device):

    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        pred_vid = torch.zeros(0, device=device) 
        gt_vid = np.empty(shape=(0, 0))

        # dataloader_iter = iter(dataloader)
        # batch_input, batch_label = next(dataloader_iter)
        # print(f"Input shape: {batch_input.shape}")
        # print(f"Label shape: {batch_label.shape}")
        # print(f"Input data type: {batch_input.dtype}")
        # print(f"Label data type: {batch_label.dtype}")

    

        # for i, input, label in enumerate(dataloader):
        count = 0
        for i, (input, label) in enumerate(dataloader):
          
           
            # print(f"Input: {input.shape}")
            # print(f"Label: {label.shape}")
            
            
            input = input.to(device)
          
            input = input.permute(0, 2, 1, 3)
       
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            # print(f"Logits: {sig}")
            # sig = torch.sigmoid(logits)


            #frame level prediction
            pred = torch.cat((pred, sig))

            #video level preditction
            # print(sig)
            
            if (torch.sum(sig > 0.5) / sig.numel()) >= 0.1: # Check if more than 10% of values in sig are 1
                pred_vid = torch.cat((pred_vid,torch.tensor([1.0], device=device)))
            else:
                pred_vid = torch.cat((pred_vid, torch.tensor([0.0], device=device)))

            #append true label
            # gt_vid = np.append(gt_vid, label)
            gt_vid = np.append(gt_vid, label.cpu().numpy())
            count = count + 1
            print(f"video: {count}")

            # print(f"Video prediction size: {len(pred_vid)}")
            # print(f"Video label size: {len(gt_vid)}")




            # #TODO: change this to dowload video based ground truth instead of just one big file for all
            # if mode == "val":
            #     gt = np.load(args.val_gt)
            
            # elif mode == "test":
            #     gt = np.load(args.test_gt)

            


        # download concatenated frame-based ground truth
        if mode == "val":
            gt = np.load(args.val_gt)
            
        elif mode == "test":
            gt = np.load(args.test_gt)

        
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        # np.save('fpr.npy', fpr)
        # np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))


        precision, recall, th = precision_recall_curve(list(gt), pred)

        pr_auc = auc(recall, precision)
        # np.save('precision.npy', precision)
        # np.save('recall.npy', recall)

        pred_vid = list(pred_vid.cpu().detach().numpy())
        # pred_vid = np.repeat(np.array(pred_vid), 16)

        # print(f"Video prediction size: {len(pred_vid)}")
        # print(f"Video label size: {len(gt_vid)}")
        fpr_vid,tpr_vid, threshold_vid = roc_curve(gt_vid, pred_vid)
        rec_auc_vid = auc(fpr_vid, tpr_vid)
        print('video level auc : ' + str(rec_auc_vid))

        precision_vid, recall_vid, th_vid = precision_recall_curve(gt_vid, pred_vid)

        now = datetime.datetime.now()

        output_dir = './output/' + now.strftime('%Y-%m-%d')
        os.makedirs(output_dir, exist_ok=True)

        return rec_auc, fpr, tpr, precision, recall,\
        rec_auc_vid, fpr_vid, tpr_vid, precision_vid, recall_vid
