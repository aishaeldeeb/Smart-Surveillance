from torch.utils.data import DataLoader
import os as os
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
# from test_10crop import test
from test import test
import option
from tqdm import tqdm
from config import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime


if __name__ == '__main__':
    
    
    args = option.parser.parse_args()
    config = Config(args)
   


    train_nloader = DataLoader(Dataset(args, mode="train", is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    

    train_aloader = DataLoader(Dataset(args, mode="train", is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    
    # val_loader = DataLoader(Dataset(args, mode="val", is_normal=False),
    #                            batch_size=1, shuffle=True,
    #                            num_workers=0, pin_memory=False, drop_last=True)
    
    val_dataset = Dataset(args, mode="val", is_normal=False)
    # print("********************")
    # print("validation dataset")
    # for i in range(len(val_dataset)):
    #     input, label = val_dataset[i]
    #     print(input.shape)


    val_loader = DataLoader(val_dataset, 
                        batch_size=1, 
                        shuffle=True, 
                        num_workers=0, 
                        pin_memory=False, 
                        drop_last=True)
    # val_loader = DataLoader(val_dataset, 
    #                         batch_size=args.batch_size, 
    #                         shuffle=True, 
    #                         num_workers=0, 
    #                         pin_memory=False, 
    #                         drop_last=True, 
    #                         collate_fn=val_dataset.collate_fn)
    
    # for batch_idx, (input, label) in enumerate(val_loader):
    #     print(f"Batch {batch_idx + 1}:")
    #     print(f"Input: {input}")
    #     print(f"Label: {label}")

    # test_loader = DataLoader(Dataset(args, mode="test", is_normal=False),
    #                            batch_size=args.batch_size, shuffle=True,
    #                            num_workers=0, pin_memory=False, drop_last=True)
    
    # test_loader = DataLoader(Dataset(args, mode="test", is_normal=False), 
    #                      batch_size=args.batch_size, 
    #                      shuffle=True, 
    #                      num_workers=0, 
    #                      pin_memory=False, 
    #                      drop_last=True, 
    #                      collate_fn=Dataset.collate_fn)
    
    test_dataset = Dataset(args, mode="test", is_normal=False)
    test_loader = DataLoader(test_dataset, 
                        batch_size=1, 
                        shuffle=True, 
                        num_workers=0, 
                        pin_memory=False, 
                        drop_last=True)
    # test_loader = DataLoader(test_dataset, 
    #                         batch_size=args.batch_size, 
    #                         shuffle=True, 
    #                         num_workers=0, 
    #                         pin_memory=False, 
    #                         drop_last=True, 
    #                         collate_fn=test_dataset.collate_fn)

    # dataset = Dataset(args, mode="test", is_normal=True)

    # for i in range(len(dataset)):
    #     input_tensor, label_tensor = dataset[i]
    #     print('Type of input_tensor:', type(input_tensor))
    #     print('Type of label_tensor:', type(label_tensor))
    #     if torch.is_tensor(input_tensor):
    #         print('Input tensor size:', input_tensor.size())
    #     else:
    #         print('input_tensor is not a tensor')
    #     if torch.is_tensor(label_tensor):
    #         print('Label tensor size:', label_tensor.size())
    #     else:
    #         print('label_tensor is not a tensor')



    print(f"Normal training dataloader length: {len(train_nloader)}")
    print(f"Abnormal training dataloader length: {len(train_aloader)}")

    print(f"Validation dataloader length: {len(val_loader)}")

    print(f"Testing dataloader length: {len(test_loader)}")


    


    model = Model(args.feature_size, args.batch_size)


    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    # Load the model 
    if not args.train_from_scratch:
        checkpoint_path = os.path.join('ckpt', args.prev_ckpts)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {checkpoint_path}")

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    val_info = {"epoch": [], "test_AUC": []}
    val_info_vid = {"epoch": [], "test_AUC": []}

    best_auc_val = -1
    best_fpr = 0
    best_tpr = 0
    best_precision = 0
    best_recall = 0
    output_path = '/home/aishaeld/scratch/RTFM/output' 
  
    best_auc_val_vid = -1
    best_fpr_vid = 0
    best_tpr_vid = 0
    best_precision_vid = 0
    best_recall_vid = 0
    print("*****************************")
    print("start training")
    print("*****************************")

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]


        if (step - 1) % len(train_aloader) == 0:
            loadern_iter = iter(train_aloader)

        if (step - 1) % len(train_nloader) == 0:
            loadera_iter = iter(train_nloader)

       


        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device)


        if step % 5 == 0 and step > 200:
        # if step % 5 == 0:
            print("*****************************")
            print("start validation")
            print("*****************************")

            
            auc_val, fpr, tpr, precision, recall,\
            auc_val_vid, fpr_vid, tpr_vid, precision_vid, recall_vid = test(val_loader, model, "val", args, device)
            
            
            # auc_val, fpr, tpr, precision, recall = test(val_loader, model, args, device)

            val_info["epoch"].append(step)
            val_info["test_AUC"].append(auc_val)

            val_info_vid["epoch"].append(step)
            val_info_vid["test_AUC"].append(auc_val_vid)

            if val_info["test_AUC"][-1] > best_auc_val:
                best_auc_val = val_info["test_AUC"][-1]
                best_fpr = fpr
                best_tpr = tpr
                best_precision = precision
                best_recall = recall
                print("********************")

                print(f"Validation dictionary: {val_info}")


                # save_best_record(val_info, os.path.join(output_path, '{}-step-auc_val.txt'.format(step)))

            if val_info_vid["test_AUC"][-1] > best_auc_val:
                best_auc_val_vid = val_info_vid["test_AUC"][-1]
                best_fpr_vid = fpr_vid
                best_tpr_vid = tpr_vid
                best_precision_vid = precision_vid
                best_recall_vid = recall_vid

                torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
                print(val_info_vid)
                save_best_record(val_info_vid, os.path.join(output_path, '{}-step-auc_val_video.txt'.format(step)))
        
    now = datetime.datetime.now()
    output_dir = './output/' + datetime.datetime.now().strftime('%Y-%m-%d')
    os.makedirs(output_dir, exist_ok=True)


    # plot best val ROC curve
    plt.figure()
    plt.plot(best_fpr, best_tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Best Val  ROC Curve')
    roc_fig_name = f'{args.model_name}_best_val_ROC_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png' # Figure name with timestamp and variable name
    plt.savefig(output_dir + '/' + roc_fig_name)
    plt.close()

    plt.figure()
    plt.plot(best_fpr_vid, best_tpr_vid)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Best Video_Based Val  ROC Curve')
    roc_fig_name = f'{args.model_name}_best_val_ROC_video_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png' # Figure name with timestamp and variable name
    plt.savefig(output_dir + '/' + roc_fig_name)
    plt.close()
    


    #Plot best val precision-recall curve
    plt.figure()
    plt.plot(best_recall, best_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Best Val Precision-Recall Curve')
    pre_rec_fig_name = f'{args.model_name}_best_val_pre_rec_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png' # Figure name with timestamp and variable name
    plt.savefig( output_dir + '/' + pre_rec_fig_name)
    plt.close()
    
    plt.figure()
    plt.plot(best_recall_vid, best_precision_vid)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Best Video-based Val Precision-Recall Curve')
    pre_rec_fig_name = f'{args.model_name}_best_val_pre_rec_video_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png' # Figure name with timestamp and variable name
    plt.savefig( output_dir + '/' + pre_rec_fig_name)
    plt.close()

 
    print("*****************************")
    print("Start testing")
    print("*****************************")

    #model inference on test loader and save the results of the test set
    # test_loader_iter = iter(test_loader)
    auc_test, fpr, tpr, precision, recall, \
    auc_test_vid, fpr_vid, tpr_vid, precision_vid, recall_vid = test(test_loader, model, "test", args, device)


    test_info = {"epoch": [1], "test_AUC": []}
    test_info_vid = {"epoch":[1], "test_AUC": []}

    test_info["test_AUC"].append(auc_test)
    test_info_vid["test_AUC"].append(auc_test_vid)

    save_best_record(test_info, os.path.join(output_path, '{}-step-auc_test.txt'.format(step)))
    save_best_record(test_info_vid, os.path.join(output_path, '{}-step-auc_test_video.txt'.format(step)))
    

    # Plot test ROC curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    roc_fig_name = f'{args.model_name}_test_ROC_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png' 
    plt.savefig(output_dir + '/' + roc_fig_name)
    plt.close()

    plt.figure()
    plt.plot(fpr_vid, tpr_vid)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Video_based Test ROC Curve')
    roc_fig_name = f'{args.model_name}_test_ROC_video_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png' 
    plt.savefig(output_dir + '/' + roc_fig_name)
    plt.close()


    #Plot test precision-recall curve
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Precision-Recall Curve')
    pre_rec_fig_name = f'{args.model_name}_test_pre_rec_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png' 
    plt.savefig( output_dir + '/' + pre_rec_fig_name)
    plt.close()

    plt.figure()
    plt.plot(recall_vid, precision_vid)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Video_based Test Precision-Recall Curve')
    pre_rec_fig_name = f'{args.model_name}_test_pre_rec_video_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png' 
    plt.savefig( output_dir + '/' + pre_rec_fig_name)
    plt.close()