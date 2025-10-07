import os
import logging
import copy
import sys
import h5py
import time

import torch
from Steerable.utils import Metrics, Patchify, Reconstruct

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

 ####################################################################################################################################
 ########################################################### Testing Loop ###########################################################
 ####################################################################################################################################

    kernel_size = tuple(datasets['test'][0][0].shape[1:])
    stride = tuple([k // 2 for k in kernel_size])
    image_shape = datasets['eval_val'][0][1].shape

    class ApplyModel:
        def __call__(self, tensor):
            return torch.softmax(model(tensor), dim=1)


    def eval(dataset, sigma, save=False, verbose=False):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)

        # Patchification-Reconstruction
        patchify = Patchify(kernel_size=kernel_size, stride=stride, transform=ApplyModel())
        reconstruct = Reconstruct(kernel_size=kernel_size, image_shape=image_shape, stride=stride, sigma=sigma)

        # Metrics
        metrics = Metrics(num_classes, metric_type)
        total_score_per_class, total_score = torch.zeros(num_classes), 0.0

        # Saving
        if save:
            f = h5py.File(os.path.join(log_dir, 'eval.hdf5'), 'w')
            f.create_dataset('probs', (0, num_classes)  + image_shape, maxshape=(None, num_classes) + image_shape, chunks=True)
            prob_dataset = f['probs']

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
                prob_dataset.resize((len(prob_dataset) + len(probs),) + prob_dataset.shape[1:])
                prob_dataset[-len(probs):] = probs.cpu()

        if save:
            f.close()

        # Logging
        macro_score_per_class = total_score_per_class / len(dataset)
        macro_score = total_score / len(dataset)
        micro_score_per_class = metrics.micro_per_class()
        micro_score = micro_score_per_class[1:].mean().item()

        logger.info(f"\nMacro {metric_type.capitalize()} per class = {macro_score_per_class}")
        logger.info(f"Macro {metric_type.capitalize()} = {macro_score:.4f}")
        logger.info(f"\nMicro {metric_type.capitalize()} per class = {micro_score_per_class}")
        logger.info(f"Micro {metric_type.capitalize()} = {micro_score:.4f}\n\n")

        if verbose:
            print(f"Best Sigma = {best_sigma}")
            print(f"\nMacro {metric_type.capitalize()} per class = {macro_score_per_class}")
            print(f"Macro {metric_type.capitalize()} = {macro_score:.4f}")
            print(f"\nMicro {metric_type.capitalize()} per class = {micro_score_per_class}")
            print(f"Micro {metric_type.capitalize()} = {micro_score:.4f}")

        return macro_score

    best_score, best_sigma = 0, 0
    sigma_values = torch.logspace(1, 3, 7).tolist()
    for sigma in sigma_values:
        score = eval(dataset=datasets['eval_val'], sigma=sigma, save=False)
        if score > best_score:
            best_score, best_sigma = score, sigma

    logger.info('Running with best sigma.')
    eval(dataset=datasets['eval_test'], sigma=best_sigma, save=save, verbose=True)


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
