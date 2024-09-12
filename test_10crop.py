import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# def test(dataloader, model, args, viz, device):
def test(dataloader, model, args, device):

    

    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            
            input = input.to(device)
          
            input = input.permute(0, 2, 1, 3)
       
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

      
        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        # pred = np.repeat(np.array(pred), len(gt))


        if len(gt) == len(pred):
            fpr, tpr, threshold = roc_curve(list(gt), pred)
        
            fpr, tpr, threshold = roc_curve(list(gt), pred)
            # np.save('fpr.npy', fpr)
            # np.save('tpr.npy', tpr)
            rec_auc = auc(fpr, tpr)
            print('auc : ' + str(rec_auc))


            precision, recall, th = precision_recall_curve(list(gt), pred)
            pr_auc = auc(recall, precision)
            # np.save('precision.npy', precision)
            # np.save('recall.npy', recall)
            # Get the current time and date
            now = datetime.datetime.now()

            # Create a directory with date stamp
            output_dir = './output/' + datetime.datetime.now().strftime('%Y-%m-%d')
            os.makedirs(output_dir, exist_ok=True)

            # # Plot ROC curve
            # plt.figure()
            # plt.plot(fpr, tpr)
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('ROC Curve')
            # roc_fig_name = f'{args.model_name}_ROC_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png' # Figure name with timestamp and variable name
            # plt.savefig(output_dir + '/' + roc_fig_name)
            # plt.close()


            # # Plot precision-recall curve
            # plt.figure()
            # plt.plot(recall, precision)
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.title('Precision-Recall Curve')
            # pre_rec_fig_name = f'{args.model_name}_pre_rec_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png' # Figure name with timestamp and variable name
            # plt.savefig( output_dir + '/' + pre_rec_fig_name)
            # plt.close()

            # viz.plot_lines('pr_auc', pr_auc)
            # viz.plot_lines('auc', rec_auc)
            # viz.lines('scores', pred)
            # viz.lines('roc', tpr, fpr)
        else:
            print(f"Error: Number of samples in 'gt': {len(gt)} and 'pred': {len(pred)} arrays are not equal.")
        return rec_auc, fpr, tpr, precision, recall