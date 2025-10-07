import os
import logging
import copy
import sys
import h5py
import time

import torch
from Steerable.utils import Metrics, Patchify, Reconstruct, Augment, rotate_image2D

def main(model_path, data_path, batch_size, metric_type, save):

 ################################################################################################################################### 
 ##################################################### Logging #####################################################################
 ###################################################################################################################################
    arguments = copy.deepcopy(locals())
    save = bool(int(save)) 
    log_dir = os.path.join(model_path, 'log/')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    
    # Creating the logger
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"), mode = "w")
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))
    logger.info("\n\n")
    
 ###################################################################################################################################
 ################################################### Loading Model and Datasets ####################################################
 ###################################################################################################################################
    
    # Load the model
    model = Model()
    num_classes = model.num_classes
    device = torch.device("cuda")
    model = model.to(device)
    
    # Datasets
    datasets = get_datasets(data_path)
    with torch.no_grad():
        model(datasets['test'][0][0].unsqueeze(0).to(device))
        if datasets['val'] is not None:
            model.load_state_dict(torch.load(os.path.join(model_path, 'log', 'best_state.pkl')))
        else:
            model.load_state_dict(torch.load(os.path.join(model_path, 'log', 'state.pkl')))
    model.eval()
    logger.info("Model Loaded!")
    logger.info(f"{sum(x.numel() for x in model.parameters())} paramerters in total\n\n")

    if datasets['train'][0][1].ndim == 2:
        function = lambda data, degree : (rotate_image2D(data[0], degree, order=1),
                                          rotate_image2D(data[1], degree, order=0))

    if datasets['train'][0][1].ndim == 3:
        function = lambda data, degree : (rotate_image2D(data[0].permute(0,3,1,2), degree, order=1).permute(0,2,3,1), 
                                          rotate_image2D(data[1].permute(2,0,1), degree, order=0).permute(1,2,0))
    parameters = [0, 180]
    aug_dataset = torch.utils.data.Subset(Augment(datasets['eval_test'], function=function, parameters=parameters, batched=False), list(range(20)))
    data_loader = torch.utils.data.DataLoader(aug_dataset, batch_size = 1)

 ####################################################################################################################################
 ########################################################### Testing Loop ###########################################################
 ####################################################################################################################################
    kernel_size = tuple(datasets['test'][0][0].shape[1:])
    stride = tuple([k // 2 for k in kernel_size])
    image_shape = datasets['eval_val'][0][1].shape

    class ApplyModel:
        def __call__(self, tensor):
            return torch.softmax(model(tensor), dim=1)


    # Patchification-Reconstruction
    with open("output.eval", "r") as f:
        for line in f:
            if "Best Sigma =" in line: 
                sigma = float(line.split("=")[1].strip())
                break
    patchify = Patchify(kernel_size=kernel_size, stride=stride, transform=ApplyModel())
    reconstruct = Reconstruct(kernel_size=kernel_size, image_shape=image_shape, stride=stride, sigma=sigma)

    # Metrics
    metrics = Metrics(num_classes, metric_type)
    total_score_per_class, total_score = torch.zeros(num_classes), 0.0

    # Saving
    if save:
        f = h5py.File(os.path.join(log_dir, 'aug.hdf5'), 'w')
        f.create_dataset('preds', (0,) + image_shape, maxshape=(None,) + image_shape, chunks=True)
        pred_dataset = f['preds']

    logger.info(f"\nTesting:\n")
    logger.info(f'Running evaluation with sigma={sigma:.2f}:')
    for batch_index, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if labels.shape[1:] != reconstruct.image_shape:
            reconstruct = Reconstruct(kernel_size, labels.shape[1:], stride, sigma=sigma)

        with torch.no_grad():
            t0 = time.time()
            probs = reconstruct(patchify(inputs[0], batch_size=batch_size)).unsqueeze(0)
            preds = torch.argmax(probs, dim=1)
            t1 = time.time()

            metrics.add_to_confusion_matrix(preds, labels)
            score_per_class = metrics.macro_per_class(preds, labels)
            score = score_per_class[1:].mean().item()
            total_score_per_class += score_per_class * len(inputs)
            total_score += score * len(inputs)
           
        logger.info(f'Test [{batch_index+1}/{len(data_loader)}] '
                    f'Time : {(t1-t0)*1e3:.1f} ms\t'
                    f'{metric_type.capitalize()} Per Class : {score_per_class}\t'
                    f'{metric_type.capitalize()} : {score:.4f}\t'
                    f'<{metric_type.capitalize()}> : {total_score/(batch_index+1):.4f}')
 
        if save:
            pred_dataset.resize((len(pred_dataset) + len(preds),) + pred_dataset.shape[1:])
            pred_dataset[-len(preds):] = preds.cpu()

    if save:
        f.close()

    # Logging
    macro_score_per_class = total_score_per_class / len(aug_dataset)
    macro_score = total_score / len(aug_dataset)
    micro_score_per_class = metrics.micro_per_class()
    micro_score = micro_score_per_class[1:].mean().item()

    logger.info(f"\nMacro {metric_type.capitalize()} per class = {macro_score_per_class}")
    logger.info(f"Macro {metric_type.capitalize()} = {macro_score:.4f}")
    logger.info(f"\nMicro {metric_type.capitalize()} per class = {micro_score_per_class}")
    logger.info(f"Micro {metric_type.capitalize()} = {micro_score:.4f}\n\n")

    print(f"\nMacro {metric_type.capitalize()} per class = {macro_score_per_class}")
    print(f"Macro {metric_type.capitalize()} = {macro_score:.4f}")
    print(f"\nMicro {metric_type.capitalize()} per class = {micro_score_per_class}")
    print(f"Micro {metric_type.capitalize()} = {micro_score:.4f}")

############################################################################################################################
################################################### Argument Parser ########################################################
############################################################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--metric_type", type=str, default='dice')
    parser.add_argument("--save", type=str, default=0)

    args = parser.parse_args()
    sys.path.append(args.__dict__['model_path'])
    from model import Model, get_datasets

    main(**args.__dict__)
