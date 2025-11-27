import warnings
warnings.filterwarnings("ignore")

import shutil
import math
import random
from collections import defaultdict
import os
import csv
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import cv2
import matplotlib.pyplot as plt


from datasets.data_utils import get_class_splits, get_datasets, get_pseudo_label_weights
from datasets.transform import get_transform, ContrastiveLearningViewGenerator
from models.modules._MEBin import MEBin
from utils.general_utils import AverageMeter, init_experiment
from utils.cluster_and_log_utils import log_accs_from_preds
from models.modules.load_backbone import load_backbone
from models.loss._distill_loss import DistillLoss
from models.loss._contrastive_loss import info_nce_logits, SupConLoss
from models.modules._classifier import get_params_groups, MultiHead


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AnomalyNCD():
    def __init__(self, args):
        self.args = args


    def train_init(self):
        """
        Initialize the training parameters.
        """

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = get_class_splits(self.args)

        setup_seed(self.args.seed)
        self.args.base_category = self.args.base_data_path.split('/')[-1]
        self.args.num_labeled_classes = len(self.args.train_classes)         
        self.args.num_unlabeled_classes = len(self.args.unlabeled_classes)  
        self.args.image_size = 224
        self.args.feat_dim = 768
        self.args.num_mlp_layers = 3
        self.args.mlp_out_dim = self.args.num_labeled_classes + self.args.num_unlabeled_classes
        self.args.interpolation = 3
        self.args.crop_pct = 0.875 

        init_experiment(self.args, runner_name=[self.args.runner_name])

        torch.backends.cudnn.benchmark = True

        self.model = self.load_model()

        self.train_loader, self.test_loader = self.load_datasets()

        self.args.logger.info('model build')
    

    def load_model(self):
        """
        Load the model consisting of Mask-Guided Vision Transformer (MGViT) and projector 
        """

        # load MGViT model
        MGViT = load_backbone(self.args.pretrained_backbone, mask_layers=self.args.mask_layers)

        # Only the blocks from grad_from_block onward are fine-tuned, while the earlier layers remain frozen.
        for m in MGViT.parameters():
            m.requires_grad = False
        for name, m in MGViT.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= self.args.grad_from_block:
                    m.requires_grad = True
                    
        # load projector
        projector = MultiHead(in_dim=self.args.feat_dim, out_dim=self.args.mlp_out_dim, nlayers=self.args.num_mlp_layers, n_head=self.args.n_head)

        model = nn.Sequential(MGViT, projector).to(self.device)

        return model


    def load_datasets(self):
        """
        load the train loader and test loader
        """
        train_transform, test_transform = get_transform(image_size=self.args.image_size, args=self.args)
        train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=self.args.n_views)

        train_dataset, test_dataset = get_datasets(train_transform, test_transform, self.args)

        # --------------------
        # SAMPLER
        # Sampler which balances labelled and unlabelled examples in each batch
        # --------------------
        label_len = len(train_dataset.labelled_dataset)
        unlabelled_len = len(train_dataset.unlabelled_dataset)
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

        # ✅ 윈도우에서는 멀티프로세싱(worker>0) 쓰면 lambda 때문에 에러 → 0으로 고정
        if os.name == "nt":
            num_workers = 0
        else:
            num_workers = self.args.num_workers

        train_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_size=self.args.batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_dataset,
            num_workers=num_workers,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=False,
        )

        return train_loader, test_loader


    def sub_image_predict(self, epoch, save_name, loss_list):
        """                                                                                        
        Predict cropped sub-images and calculate NMI, ARI, and F1 scores.
        Args:
            epoch: [int]. Current epoch.
            save_name: [str]. Name of the saved file.
            loss_list: [list]. List containing the loss values, one for each head.
        """
        self.model.eval()

        # Find the index of the head with the minimum loss
        min_loss = min(loss_list)
        min_indices = [index for index, value in enumerate(loss_list) if value == min_loss]

        targets, idxs, img_paths = [], [], []
        preds_dict = defaultdict(list)
        mask = np.array([])

        # Iterate over test data loader to process each batch of sub-images
        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            images, label, uq_idx, image_path, masks, mask_path = batch
            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            with torch.no_grad():

                MGViT, projector = self.model
                cls_token = MGViT(images, masks)
                _, logits = projector(cls_token)

                for i in range(self.args.n_head):
                    sof_logit = F.softmax(logits[i], dim=-1)
                    sof_logit[:, :self.args.num_labeled_classes] = 0
                    sof_logit[:, self.args.num_labeled_classes:] /= sof_logit[:, self.args.num_labeled_classes:].sum(dim=-1, keepdim=True)
                    preds_dict[i].extend(sof_logit.argmax(1).cpu().numpy())
                    
                targets.append(label.cpu().numpy())
                idxs.append(uq_idx.cpu().numpy())
                img_paths.extend(image_path)
                mask = np.append(mask, np.array([True if x.item() in range(len(self.args.train_classes)) else False for x in label]))


        targets = np.concatenate(targets)
        idxs = np.concatenate(idxs)
        img_paths = np.array(img_paths)

        result_dict_ls, NMI_ls, ARI_ls, F1_ls= [], [], [], []

        # Evaluate predictions for each head
        for i in range(self.args.n_head):
            preds = np.array(preds_dict[i])
            dic, NMI, ARI, F1 = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                            T=epoch, save_name=save_name,
                                                            args=self.args, idxs=idxs, img_paths=img_paths)
            result_dict_ls.append(dic)
            NMI_ls.append(NMI)
            ARI_ls.append(ARI)
            F1_ls.append(F1)

            # Log results if it's the min loss head
            if i in min_indices:
                self.args.logger.info('NMI {:.4f} | ARI {:.4f} | F1 {:.4f}'.format(NMI, ARI, F1))
        
        return result_dict_ls
    

    def region_merge_predict(self, epoch, save_name, loss_list):
        """                                                                                        
        Weights each sub-image's prediction according to the anomaly area and merges them to obtain the final image prediction, 
        followed by calculating the NMI, ARI, and F1 scores.
        Args:
            epoch: [int]. Current epoch.
            save_name: [str]. Name of the saved file.
            loss_list: [list]. List containing the loss values, one for each head.
        """
        self.model.eval()

        # Find the index of the head with the minimum loss
        min_loss = min(loss_list)
        min_indices = [index for index, value in enumerate(loss_list) if value == min_loss]

        # temperature for area average
        if self.args.dataset == 'mvtec':
            temps = [100]
        else:
            temps = [50]

        masks = np.array([])
        targets = np.array([])
        idxs = np.array([])
        img_paths = np.array([])
        test_dict = defaultdict(lambda: defaultdict(list))

        # Iterate over test data loader to process each batch of sub-images
        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            images, labels, uq_idx, image_path, masks_img, mask_paths = batch                      
            images = images.cuda(non_blocking=True)
            masks_img = masks_img.cuda(non_blocking=True)
        

            with torch.no_grad():
                MGViT, projector = self.model
                cls_token = MGViT(images, masks_img)
                _, logits = projector(cls_token)
                                                                                     
                logits = torch.stack(logits).permute(1,0,2)                                                        
                for label, idx, path, logit, mask_path in zip(labels, uq_idx, image_path, logits, mask_paths):                            
                    last_1 = os.path.basename(path)
                    last_2 = os.path.basename(os.path.dirname(path))
                    split = last_1.split("_crop")[0]
                    name = os.path.join(last_2, split + ".png")

                    test_dict[name]["label"].append(label)
                    test_dict[name]["idx"].append(idx)
                    test_dict[name]["path"].append(path)
                    test_dict[name]["mask_path"].append(mask_path)
                    mask = label in range(len(self.args.train_classes))
                    test_dict[name]["mask"].append(mask)
                    
                    sof_logit = F.softmax(logit / 0.1, dim=-1)

                    if not mask :
                        sof_logit[:, :self.args.num_labeled_classes] = 0
                        sof_logit[:, self.args.num_labeled_classes:] /= sof_logit[:, self.args.num_labeled_classes:].sum(dim=-1, keepdim=True)
                    for i in range(self.args.n_head):
                        test_dict[name]["logit{}".format(i)].append(sof_logit[i])       
                            
        # Compute the weighted predictions based on anomaly area
        for name, data in test_dict.items():
            for i in range(self.args.n_head):
                logit_values = data["logit{}".format(i)]
                with torch.no_grad():
                    if logit_values:
                        if len(logit_values) > 1:
                            pixel_counts_tensor = torch.zeros(len(logit_values)).to('cuda:0')
                            for j, mask_pth in enumerate(data["mask_path"]):
                                gray_image = Image.open(mask_pth).convert('L')
                                gray_array = np.array(gray_image)
                                count_255 = np.sum(gray_array == 255)
                                pixel_counts_tensor[j] = np.sqrt(count_255)
                            logit_values_torch = torch.stack(logit_values)
                            area_average_logits = []
                            for temp_idx, temp in enumerate(temps):
                                pixel_counts_tensor_temp = pixel_counts_tensor / temp
                                sof_counts = F.softmax(pixel_counts_tensor_temp, dim=0).view(-1, 1)
                                area_average_logits.append(torch.sum(logit_values_torch * sof_counts, dim=0))
                        else:
                            area_average_logits = []
                            for idx in range(len(temps)):
                                area_average_logits.append(sum(logit_values) / len(logit_values))

                    area_average_preds = []
                    for id in range(len(temps)):
                        area_average_preds.append(torch.argmax(area_average_logits[id]))
                
                    for id in range(len(temps)):
                        test_dict[name]["area_average_pred{}_temp{}".format(i, id)].append(area_average_preds[id])


        for img_path, data in test_dict.items():
            targets = np.append(targets, data["label"][0].item())
            masks = np.append(masks, data["mask"][0])
            img_paths = np.append(img_paths, img_path)
            idxs = np.append(idxs, data["idx"][0].item())
        
        # Evaluate predictions for each head and temperature
        for i in range(self.args.n_head):
            area_average_preds_ls = [np.array([]) for _ in range(len(temps))]

            for img_path, data in test_dict.items():
                for idk in range(len(temps)):
                    area_average_preds_ls[idk] = np.append(area_average_preds_ls[idk], data["area_average_pred{}_temp{}".format(i, idk)][0].item())
            

            # Calculate performance metrics for area-weighted predictions
            for temp_idx, temp in enumerate(temps):

                result_dict, NMI, ARI, F1 = log_accs_from_preds(
                                                            y_true=targets, y_pred=area_average_preds_ls[temp_idx], mask=masks,
                                                            T=epoch, save_name=save_name,
                                                            args=self.args, idxs=idxs, img_paths=img_paths)
                
                # Log metrics and store results if it's the min loss head
                if i in min_indices:
                    result_dict_area = result_dict
                    head_idx = i
                    self.args.logger.info('NMI {:.4f} | ARI {:.4f} | F1 {:.4f}'.format(NMI, ARI, F1))

                    if epoch + 1 == self.args.epochs:
                        # store NMI, ARI, F1 in results.csv
                        filename = os.path.join('outputs', self.args.runner_name, 'metrics.csv')

                        # if the file does not exist, create it
                        if not os.path.exists(filename):
                            with open(filename, 'w') as file:
                                writer = csv.writer(file)
                                writer.writerow(['category', 'NMI', 'ARI', 'F1'])
                                writer.writerow([self.args.category, NMI, ARI, F1])
                        
                        # else write after the csv
                        else:
                            with open(filename, 'a') as file:
                                writer = csv.writer(file)
                                writer.writerow([self.args.category, NMI, ARI, F1])

        return result_dict_area, head_idx 
    
    
    def binarization(self):
        """
        Use MEBin to binarize the anomaly maps and crop the images and masks.
        """

        dataset_name = self.args.dataset

        # Specify the path of the original_images, anomaly maps and the output.
        origin_image_path = self.args.dataset_path
        anomaly_map_path = self.args.anomaly_map_path
        output_path = self.args.binary_data_path
        crop_output_path = self.args.crop_data_path

        product_name = self.args.category

        for path in [output_path, crop_output_path]:
            ppath = os.path.join(path, product_name)
            if os.path.exists(ppath):
                shutil.rmtree(ppath)
            os.makedirs(ppath)

        # Collect anomaly map paths
        anomaly_map_file_dict = {}
        img_file_list = {}
        anomaly_crop_score_list = {}

        anomaly_type_list = sorted(os.listdir(f"{anomaly_map_path}/{product_name}/"))

        if "combined" in anomaly_type_list:
            anomaly_type_list.remove("combined")

        for anomaly_type in anomaly_type_list:
            if dataset_name.lower() == 'mvtec':
                img_file_path = f"{origin_image_path}/{product_name}/test/{anomaly_type}"
                anomaly_map_file_path = f"{anomaly_map_path}/{product_name}/{anomaly_type}"
            elif dataset_name.lower() == 'mtd':
                img_file_path = f"{origin_image_path}/{anomaly_type}"
                anomaly_map_file_path = f"{anomaly_map_path}/{product_name}/{anomaly_type}"
            else:
                raise NotImplementedError("Dataset not supported")
            tmp_img_file_list = sorted(os.listdir(img_file_path))
            tmp_anomaly_map_file_list = sorted(os.listdir(anomaly_map_file_path))

            anomaly_map_file_dict[anomaly_type] = [os.path.join(anomaly_map_file_path, path) for path in tmp_anomaly_map_file_list]
            img_file_list[anomaly_type] = [os.path.join(img_file_path, path) for path in tmp_img_file_list]
            
        # dict -> list
        anomaly_map_file_list = []
        for anomaly_type in anomaly_type_list:
            anomaly_map_file_list.extend(anomaly_map_file_dict[anomaly_type])

        # Run MEBin
        bin = MEBin(self.args, anomaly_map_file_list)
        binarized_maps_list, est_anomaly_nums_list = bin.binarize_anomaly_maps()

        # list -> dict
        binarized_maps = {}
        est_anomaly_nums = {}
        idx = 0
        for anomaly_type in anomaly_type_list:
            anomaly_type_anomaly_map_file_list = anomaly_map_file_dict[anomaly_type]
            binarized_maps[anomaly_type] = []
            est_anomaly_nums[anomaly_type] = []
            for anomaly_map_file in anomaly_type_anomaly_map_file_list:
                idx = anomaly_map_file_list.index(anomaly_map_file)
                binarized_maps[anomaly_type].append(binarized_maps_list[idx])
                est_anomaly_nums[anomaly_type].append(est_anomaly_nums_list[idx])            

        # save the binarization result
        for anomaly_type in anomaly_type_list:
            anomaly_type_out_path = os.path.join(output_path, product_name, anomaly_type)
            os.makedirs(anomaly_type_out_path, exist_ok=True)
            anomaly_type_binarized_maps = binarized_maps[anomaly_type]
            for i, binarized_map in enumerate(anomaly_type_binarized_maps):
                map_path = os.path.join(anomaly_type_out_path, os.path.basename(anomaly_map_file_dict[anomaly_type][i]))

                # resize
                o_img_path = img_file_list[anomaly_type][i]
                o_img = cv2.imread(o_img_path)
                o_img_shape = o_img.shape
                binarized_map = cv2.resize(binarized_map, (o_img_shape[1], o_img_shape[0]), interpolation=cv2.INTER_NEAREST)

                plt.imsave(map_path, binarized_map, cmap='gray')


        # crop and save the images and masks
        for anomaly_type in anomaly_type_list:
            ano_type_score_list = {}

            save_path = f"{crop_output_path}/{product_name}/images/{anomaly_type}"
            save_mask_path = f"{crop_output_path}/{product_name}/masks/{anomaly_type}"
            for path in [save_path, save_mask_path]:
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path)

            anomaly_type_img_files = img_file_list[anomaly_type]
            anomaly_type_anomap_files = anomaly_map_file_dict[anomaly_type]
            anomaly_type_est_anomaly_nums = est_anomaly_nums[anomaly_type]

            
            for i in range(len(anomaly_type_img_files)):
                image_path = anomaly_type_img_files[i]
                anomaly_map_file_path = anomaly_type_anomap_files[i]

                binary_map_file_path = os.path.join(output_path, product_name, anomaly_type, os.path.basename(anomaly_map_file_path))
                est_ano_num = anomaly_type_est_anomaly_nums[i]

                anomaly_map = cv2.imread(anomaly_map_file_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                binary_map = cv2.imread(binary_map_file_path)
                binary_map = cv2.cvtColor(binary_map, cv2.COLOR_BGR2GRAY)
                sub_images_list, sub_masks_list, anomaly_crop_score = bin.crop_sub_image_mask(image=image, mask=binary_map, anomaly_map=anomaly_map, est_anomaly_num=est_ano_num)
                prefix = os.path.splitext(os.path.basename(image_path))[0]
                for i, img in enumerate(sub_images_list): 
                    img.save(os.path.join(save_path, "{}_crop{}.png".format(prefix, i)))
                for i, img in enumerate(sub_masks_list): 
                    img.save(os.path.join(save_mask_path, "{}_crop{}.png".format(prefix, i)))

                # save the anomaly score for each sub-image
                for i, score in enumerate(anomaly_crop_score):
                    ano_type_score_list["{}_crop{}.png".format(prefix, i)] = anomaly_crop_score[i]/255.0
            
            anomaly_crop_score_list[anomaly_type] = ano_type_score_list

        os.makedirs(f"{crop_output_path}/scores_json", exist_ok=True)

        # dump json file
        with open(f"{crop_output_path}/scores_json/{product_name}.json", "w") as f:
            json.dump(anomaly_crop_score_list, f)

                    

    def MGRL(self, epoch, optimizer, cluster_criterion):
        """
        Mask Guided Representation Learning.
        Args:
            epoch: [int]. Current epoch.
            optimizer: [torch.optim]. Optimizer for the model.
            cluster_criterion: [torch.nn]. pseudo label clustering criterion.
        Returns:
            cluster_loss_head: [list]. List containing the cluster loss values for each head.
            loss_record: [AverageMeter]. Average loss value.
        """
        loss_record = AverageMeter()
        cluster_loss_for_test = defaultdict(list)
        total_loss = 0

        # get anomaly score json file
        json_path = os.path.join(self.args.crop_data_path, 'scores_json', self.args.category+'.json')
        anomaly_score_json = json.load(open(json_path, 'r'))

        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            images, class_labels, image_path, masks, mask_path  = batch               

            # Generate pseudo-label weights for pseudo-label correction.
            sample_weights, mask_lab = get_pseudo_label_weights(image_path, self.args.anomaly_thred, self.args.base_category, anomaly_score_json)

            # copy the sample weights for another view.
            sample_weights = sample_weights * 2
            sample_weights = torch.tensor(sample_weights).cuda(non_blocking=True)

            # to ensure that the training batch contains base and novel samples.
            if torch.any(mask_lab) and not torch.all(mask_lab):
                class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
                images = torch.cat(images, dim=0).cuda(non_blocking=True)
                masks = torch.cat(masks, dim=0).cuda(non_blocking=True)

                MGViT, projector = self.model
                # MGViT extracts class tokens from input images using provided masks.
                student_cls_token = MGViT(images, masks)
                # The extracted class tokens are then passed through the projector.
                student_proj, student_out = projector(student_cls_token)

                teacher_out = [student_out[i].detach() for i in range(self.args.n_head)]

                # unsupervised contrastive loss
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # supervised contrastive loss
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                # classification loss                    
                n_head = len(student_out)
                cls_loss = 0
                cluster_loss = 0
                pstr = ''
                for i in range(n_head):
                    student_out_i = student_out[i]
                    teacher_out_i = teacher_out[i]
                    sup_logits = torch.cat([f[mask_lab] for f in (student_out_i / 0.1).chunk(2)], dim=0)  
                    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0) 
                    cls_loss += nn.CrossEntropyLoss()(sup_logits, sup_labels)
                    
                    # pseudo label classification loss
                    student_out_unlabel = torch.cat([f[~mask_lab] for f in (student_out_i).chunk(2)], dim=0)
                    teacher_out_unlabel = torch.cat([f[~mask_lab] for f in (teacher_out_i).chunk(2)], dim=0)

                    avg_probs = (student_out_unlabel / 0.1).softmax(dim=1).mean(dim=0)
                    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                    head_cluster_loss = cluster_criterion(student_out_unlabel, teacher_out_unlabel, epoch, sample_weights)+ self.args.memax_weight * me_max_loss
                    cluster_loss += head_cluster_loss
                    cluster_loss_for_test[i].append(head_cluster_loss.item())
                    
                cls_loss /= n_head
                cluster_loss /= n_head

                
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
                loss += (1 - self.args.sup_weight) * cluster_loss + self.args.sup_weight * cls_loss
                loss += (1 - self.args.sup_weight) * contrastive_loss + self.args.sup_weight * sup_con_loss
                
                # Train acc
                loss_record.update(loss.item(), class_labels.size(0))
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % self.args.print_freq == 0:
                    self.args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                .format(epoch, batch_idx, len(self.train_loader), loss.item(), pstr))

                
                cluster_loss_head = [0]*self.args.n_head
                for i in range(self.args.n_head):
                    cluster_loss_head[i] = np.mean(cluster_loss_for_test[i])
                    
        return cluster_loss_head, loss_record


    def main(self):
        # Main Element Binarization: generate the binarized results for unlabeled images and apply the Anomaly-Centered Sub-Image Cropping operation.
        self.binarization()
        
        # training the model
        self.train_init()

        if self.args.only_test:
            self.args.checkpoint_path = os.path.join(self.args.checkpoint_path, 'checkpoints/model.pt')
            if os.path.exists(self.args.checkpoint_path):
                checkpoint = torch.load(self.args.checkpoint_path)
                self.model.load_state_dict(checkpoint['model'])
                self.args.logger.info("model loaded from {}, epoch {}, base_cls:{}.".format(self.args.checkpoint_path, checkpoint['epoch'], checkpoint['base_category']))
            else:
                raise ValueError('The checkpoint path does not exist.')
            self.args.logger.info('Predicting for Sub-Image Classification...')
            results_sub_images = self.sub_image_predict(epoch=checkpoint['epoch'], save_name='Sub-image prediction', loss_list=checkpoint['loss_list'])
            self.args.logger.info('Region Merging for Image Classification...')
            results_merge = self.region_merge_predict(epoch=checkpoint['epoch'], save_name='Region merged prediction', loss_list=checkpoint['loss_list'])
        else:
            params_groups = get_params_groups(self.model)
            optimizer = SGD(params_groups, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

            exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.args.epochs,
                    eta_min=self.args.lr * 1e-3,
                )

            cluster_criterion = DistillLoss(
                                warmup_teacher_temp_epochs=self.args.warmup_teacher_temp_epochs,
                                nepochs=self.args.epochs,
                                ncrops=self.args.n_views,
                                warmup_teacher_temp=self.args.warmup_teacher_temp,
                                teacher_temp=self.args.teacher_temp,
                                num_labeled_classes=self.args.num_labeled_classes,
                                num_unlabeled_classes=self.args.num_unlabeled_classes,
                                student_temp=0.1,
                                repeat_times=self.args.repeat_times
                            )

            for epoch in range(self.args.epochs):
                cluster_loss_head, loss_record = self.MGRL(epoch, optimizer, cluster_criterion)

                self.args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

                # Step schedule
                exp_lr_scheduler.step()

                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss_list': cluster_loss_head,
                    'base_category': self.args.base_category,
                    'category': self.args.category,
                    'mask_layers': self.args.mask_layers
                }

                if epoch + 1 == self.args.epochs:
                    # Testing the results after training
                    self.args.logger.info('Predicting for Sub-Image Classification...')
                    results_sub_images = self.sub_image_predict(epoch=epoch, save_name='Sub-image prediction', loss_list=cluster_loss_head)
                    self.args.logger.info('Region Merging for Image Classification...')
                    results_merge = self.region_merge_predict(epoch=epoch, save_name='Region merged prediction', loss_list=cluster_loss_head)
                    # Save the model
                    torch.save(save_dict, self.args.model_path)
                    self.args.logger.info("model saved to {}_epoch{}, base_cls:{}.".format(self.args.model_path, epoch, self.args.num_labeled_classes))