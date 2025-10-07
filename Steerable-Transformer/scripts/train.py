import os
import copy
import time
import h5py
import logging

import torch
from model import Model, get_datasets
from Steerable.utils import Metrics, FocalLoss

def main(data_path, batch_size, rotate, learning_rate, weight_decay, num_epochs, num_workers, lr_decay_rate, lr_decay_schedule, metric_type, save=0):
    #################################################################################################################################
    ########################################################## Logging ##############################################################
    #################################################################################################################################
    arguments = copy.deepcopy(locals())
    
    log_dir = os.path.join('log/')
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

    if num_epochs>0:
        logger.info("%s", repr(arguments))
    else:
        logger.info("%s", repr({k: v for k, v in arguments.items() 
                                     if k not in {'learning_rate', 'weight_decay', 'num_epochs', 
                                                  'lr_decay_rate', 'lr_decay_schedule'}}))
    logger.info("\n\n")
    
    #################################################################################################################################
    ########################################## Loading Model, Datasets, Loss and Optimizer ##########################################
    #################################################################################################################################
    
    # DataLoader
    datasets = get_datasets(data_path, rotate=bool(rotate))
    train_loader = torch.utils.data.DataLoader(dataset=datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = None
    if datasets['val'] is not None:
         val_loader = torch.utils.data.DataLoader(dataset=datasets['val'], batch_size=batch_size, num_workers=num_workers)

    # Load the model
    model = Model()
    num_classes = model.num_classes
    device = torch.device("cuda")
    model = model.to(device)
    model(datasets['train'][0][0].unsqueeze(0).to(device))
    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))        

    # Optimizer
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0, weight_decay = weight_decay)
    def get_learning_rate(epoch):
        return learning_rate * (lr_decay_rate ** (epoch / lr_decay_schedule))

    #################################################################################################################################
    ################################################# Train, Eval and Test Functions ################################################
    #################################################################################################################################
    
    def train_step(inputs, labels):
        model.train()
        
        # Pushing to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward Pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def eval():
        model.eval()

        metrics = Metrics(num_classes, metric_type)
        total_loss, total_score, num_inputs = 0, 0, 0
        
        logger.info('\nValidation :\n')
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            # Pushing to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():

                # Forward Pass
                t0 = time.time()
                outputs = model(inputs)
                t1 = time.time()
                
                # Metrics
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                score = metrics.macro_per_class(preds, labels)[1:].mean().item()
    
                total_loss += loss * len(inputs)
                total_score += score * len(inputs)
                num_inputs += len(inputs)

            # Logging
            logger.info(f"Validation [{batch_idx+1}/{len(val_loader)}] "
                        f"Time : {(t1-t0)*1e3:.1f} ms Loss : {loss:.2f} "
                        f"{metric_type.capitalize()} : {score:.4f} <{metric_type.capitalize()}> : {total_score / num_inputs:.4f}")
            
        return total_loss / len(datasets['val']), total_score / len(datasets['val'])
    
    def test_step(inputs, labels):
        model.eval()

        # Pushing to GPU        
        inputs, labels = inputs.to(device),  labels.to(device)

        with torch.no_grad():
            # Foward Pass
            outputs = model(inputs)
            loss = criterion(outputs, labels).item()
            probs = torch.softmax(outputs, dim=1)

        return probs, loss
    
    #################################################################################################################################
    ######################################################### Training Loop #########################################################
    #################################################################################################################################

    # Metric
    epoch, early_stop, early_stop_after, best_val_loss, best_score = 0, 0, 300, float('inf'), 0
   
    # Training
    logger.info(f"\n\n\nTraining:\n")
    for epoch in range(epoch, num_epochs):
        lr = get_learning_rate(epoch)
        logger.info(f"learning rate = {lr:.2e}, weight decay = {weight_decay}, batch size = {train_loader.batch_size}")
        for p in optimizer.param_groups:
            p['lr'] = lr
        
        total_iteration_loss = 0
        total_iteration_time = 0
        
        for batch_index, (inputs, labels) in enumerate(train_loader):
            ## Train Step
            t0 = time.time()
            iteration_loss = train_step(inputs, labels.long())
            t1 = time.time()

            ## Time
            total_iteration_time = total_iteration_time + (t1-t0)*1000
            avg_iteration_time = total_iteration_time / (batch_index+1)
            
            ## Loss
            total_iteration_loss += iteration_loss
            avg_loss = total_iteration_loss / (batch_index+1)

            ## Logging
            logger.info(f"[{epoch+1}/{num_epochs}:{batch_index+1}/{len(train_loader)}] "
                        f"Time : {(t1-t0)*1e3:.1f} ms <Time> : {avg_iteration_time:.1f} ms\t"
                        f"LOSS={iteration_loss:.2f} <LOSS>={avg_loss:.2f}")
            torch.save(model.state_dict(), os.path.join(log_dir, "state.pkl"))
            
        # Evaluate
        if (epoch+1) % 1 == 0 or epoch == (num_epochs-1) or early_stop == early_stop_after:
            if val_loader is not None:
                ## Validation
                val_loss, score = eval()
                if score>= best_score:
                    best_val_loss, best_score = val_loss, score
                    early_stop = 0
                    torch.save(model.state_dict(), os.path.join(log_dir, "best_state.pkl"))
                else:
                    early_stop += 1

                ## Logging
                logger.info(f"\n\nLoss={val_loss:.4f} Best Loss={best_val_loss:.4f} "
                            f"{metric_type.capitalize()}={score:.4f} Best {metric_type.capitalize()}={best_score:.4f}")
                print(f'epoch {epoch+1}/{num_epochs} '
                      f'avg loss : {avg_loss:.4f} val loss : {val_loss:.4f} score : {score:.4f} {"*" if score==best_score else ""}')

                if early_stop == early_stop_after:
                   print(f"\n\nStopped at epoch {epoch+1}.\n")
                   break
            else :
                print(f'epoch {epoch+1}/{num_epochs} avg loss : {avg_loss:.4f}')
            
        logger.info("\n\n")

    #################################################################################################################################
    ######################################################### Testing Loop ##########################################################
    #################################################################################################################################
    
    # Loading Best Model
    if datasets['val'] is not None:
        model.load_state_dict(torch.load(os.path.join('log', 'best_state.pkl')))
    else:
        model.load_state_dict(torch.load(os.path.join('log', 'state.pkl')))
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size = batch_size, num_workers = num_workers)

    # Metrics
    metrics = Metrics(num_classes, metric_type)
    total_loss, total_score, total_score_per_class, num_inputs = 0, 0, torch.zeros(num_classes), 0

    # Saving
    save = bool(int(save))
    if save:
        f = h5py.File(os.path.join(log_dir, 'probs.hdf5'), 'w')
        prob_dataset = None
 
    # Testing
    logger.info(f"\n\n\nTesting:\n")
    for batch_index, (inputs, labels) in enumerate(test_loader):
        
        t0 = time.time()
        probs, current_loss = test_step(inputs, labels)
        t1 = time.time()
        
        preds = torch.argmax(probs, dim=1).detach().cpu()
        metrics.add_to_confusion_matrix(preds, labels)
        score_per_class = metrics.macro_per_class(preds, labels)
        score = score_per_class[1:].mean().item()
        
        total_score_per_class += score_per_class * len(inputs)
        total_score += score * len(inputs)
        total_loss += current_loss * len(inputs)
        num_inputs += len(inputs)

        if save:
            if prob_dataset is None:
                 f.create_dataset('probs', (0, ) + probs.shape[1:], maxshape=(None,) +  probs.shape[1:], chunks=True)
                 prob_dataset = f['probs']

            prob_dataset.resize((len(prob_dataset) + len(probs),) + probs_dataset.shape[1:])
            prob_dataset[-len(probs):] = probs.cpu()

        logger.info(f'Test [{batch_index+1}/{len(test_loader)}] '
                    f'Time : {(t1-t0)*1e3:.1f} ms  Loss={current_loss:.2f}\t'
                    f'{metric_type.capitalize()} : {score_per_class} <{metric_type.capitalize()}> : {total_score / num_inputs:.4f}')

    if save:   
        f.close()
       
    # Logging 
    avg_loss = total_loss / len(datasets['test'])
    macro_score_per_class = total_score_per_class / len(datasets['test'])
    macro_score = total_score / len(datasets['test'])
    micro_score_per_class = metrics.micro_per_class()
    micro_score = micro_score_per_class[1:].mean().item()
    

    logger.info(f'\n\nTesting Loss = {avg_loss:.4f}')
    logger.info(f"\nMacro {metric_type.capitalize()} per class = {macro_score_per_class}")
    logger.info(f"Macro {metric_type.capitalize()} = {macro_score:.4f}")
    logger.info(f"\nMicro {metric_type.capitalize()} per class = {micro_score_per_class}")
    logger.info(f"Micro {metric_type.capitalize()} = {micro_score:.4f}")

    print(f'\n\nTesting Loss = {avg_loss:.4f}')
    print(f"\nMacro {metric_type.capitalize()} per class = {macro_score_per_class}")
    print(f"Macro {metric_type.capitalize()} = {macro_score:.4f}")
    print(f"\nMicro {metric_type.capitalize()} per class = {micro_score_per_class}")
    print(f"Micro {metric_type.capitalize()} = {micro_score:.4f}")

#################################################################################################################################
######################################################## Argument Parser ########################################################
#################################################################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--rotate", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_decay_rate", type=float, default=0.0)
    parser.add_argument("--lr_decay_schedule", type=int, default=1)
    parser.add_argument("--metric_type", type=str, required=True)
    parser.add_argument("--save", type=int, default=0)

    args = parser.parse_args()
    main(**args.__dict__)
