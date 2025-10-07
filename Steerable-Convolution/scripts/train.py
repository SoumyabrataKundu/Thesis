import torch
import os
import logging
import copy
import sys
import time

def main(model_path, data_path, batch_size, freq_cutoff, interpolation, rotate, learning_rate, weight_decay, num_epochs, num_workers, restore, lr_decay_rate, lr_decay_schedule):
   
 ################################################################################################################################### 
 ##################################################### Logging #####################################################################
 ###################################################################################################################################
    arguments = copy.deepcopy(locals())
    restore = bool(restore)    
    log_dir = os.path.join(model_path, 'log/')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    if restore:
        restore_dir = log_dir
    
    # Creating the logger
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"), mode = "a" if restore else "w")
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))
    logger.info("\n\n")
    torch.backends.cudnn.benchmark = True
    
 ###################################################################################################################################
 ################################## Loading Model, Datasets, Loss and Optimizer ####################################################
 ###################################################################################################################################
    
    # Load the model
    model = Model(freq_cutoff, interpolation)
    device = torch.device("cuda")
    model = model.to(device)

    ## Restoring from previous training
    if restore:
        model.load_state_dict(torch.load(os.path.join(restore_dir, "state.pkl")))
    
    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    
    # Datasets
    datasets = get_datasets(data_path, rotate=bool(rotate))

    ## Dataloaders
    if datasets['train'] is not None:
        train_loader = torch.utils.data.DataLoader(dataset = datasets['train'], batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_loader = None
    if datasets['val'] is not None:
         val_loader = torch.utils.data.DataLoader(dataset = datasets['val'], batch_size = batch_size, num_workers = num_workers)
    
    # Loss
    criterion = torch.nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 0, weight_decay = weight_decay)
    
    def get_learning_rate(epoch):
        return learning_rate * (lr_decay_rate ** (epoch / lr_decay_schedule))


####################################################################################################################################
######################################## Training and Evaluation Function ##########################################################
####################################################################################################################################
    
    def train_step(inputs, labels):
        model.train()
        
        # Pushing to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward Pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Logging
        with torch.no_grad():
            _, predictions = torch.max(outputs, 1)
            accuracy = predictions.eq(labels).sum().item() / len(predictions)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item(), accuracy

    def evaluate(dataloader):
        model.eval()

        total_correct = 0
        total_samples = 0
        logger.info('\n Validation : \n')
        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(dataloader):
                # Pushing to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                t0 = time.time()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                correct = outputs.argmax(1).eq(labels).long().sum().item()
                t1 = time.time()

                total_correct += correct
                total_samples += len(inputs)

                acc = correct / len(inputs)
            
                # Logging
                logger.info(f"Validation [{batch_idx+1}/{len(dataloader)}] Time : {(t1-t0)*1e3:.1f} ms \tACC={acc:.2f}")
            
        return total_correct / total_samples, loss
    
    
####################################################################################################################################
################################################### Training Loop ##################################################################
####################################################################################################################################
    
    dynamics = []
    epoch = 0
    score = best_score = 0
    val_loss = best_val_loss = 100000
    if restore:
        dynamics = torch.load(os.path.join(restore_dir, "dynamics.pkl"))
        epoch = dynamics[-1]['epoch']
        best_score = dynamics[-1]['best_score']
    
    for epoch in range(epoch, num_epochs):
        
        lr = get_learning_rate(epoch)
        logger.info(f"learning rate = {lr}, weight decay = {weight_decay}, batch size = {train_loader.batch_size}")
        for p in optimizer.param_groups:
            p['lr'] = lr
        
        total_iteration_loss = 0
        total_iteration_accuracy = 0
        total_iteration_time = 0
        
        for batch_index, (inputs, labels) in enumerate(train_loader):

            # Train Step
            t0 = time.time()
            iterration_loss, iteration_accuracy = train_step(inputs, labels)
            t1 = time.time()

            total_iteration_time = total_iteration_time + (t1-t0)*1000
            avg_iteration_time = total_iteration_time / (batch_index + 1) 
            ## Loss
            total_iteration_loss += iterration_loss
            avg_loss = total_iteration_loss / (batch_index+1)
            ## Accuracy
            total_iteration_accuracy += iteration_accuracy
            avg_accuracy = total_iteration_accuracy / (batch_index+1)

            # Logging
            logger.info(f"[{epoch+1}/{num_epochs}:{batch_index+1}/{len(train_loader)}] Time : {(t1-t0)*1e3:.1f} ms <Time> : {avg_iteration_time:.1f} ms\
                        LOSS={iterration_loss:.2f} <LOSS>={avg_loss:.2f} \
                        ACC={iteration_accuracy:.2f} <ACC>={avg_accuracy:.2f}")
            
            dynamics.append({
                'epoch': epoch, 'batch_idx': batch_index, 'step': epoch * len(train_loader) + batch_index,
                'learning_rate': lr, 'batch_size': len(labels),
                'loss': iterration_loss, 'correct': iteration_accuracy, 'avg_loss': avg_loss, 'avg_correct': avg_accuracy,
                'best_score': best_score, 'score': score,
            })
            
            torch.save(model.state_dict(), os.path.join(log_dir, "state.pkl"))
            torch.save(dynamics, os.path.join(log_dir, "dynamics.pkl"))
            
        # Evaluate
        if (epoch+1) % 1 == 0 or epoch == (num_epochs-1):
            if val_loader is not None:
                ## Validation
                score, val_loss = evaluate(val_loader)
                if score > best_score or (score == best_score and val_loss <= best_val_loss):
                    best_score = score
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(log_dir, "best_state.pkl"))
            
                logger.info(f"\n\nScore={score*100:.2f}% Best={best_score*100:.2f}%")                   
                print(f'epoch {epoch+1}/{num_epochs} avg loss : {avg_loss:.4f} avg acc : {avg_accuracy*100:.2f} % score : {score*100 :.2f} % {"*" if score==best_score else ""}')

            else :
                print(f'epoch {epoch+1}/{num_epochs} avg loss : {avg_loss:.4f} avg acc : {avg_accuracy*100:.2f} %')
                
                logger.info("\n\n")
            
        logger.info("\n\n")

    ########################################################################################################################
    ############################################### Test Function ##########################################################
    ########################################################################################################################

    def test(inputs, labels):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        n_correct_classwise = torch.zeros(num_classes).to(device)
        n_samples_classwise = torch.zeros(num_classes).to(device)

        for truth, pred in zip(labels, predictions):
            n_correct_classwise[truth] += truth == pred
            n_samples_classwise[truth] += 1

        accuracy_classwise = n_correct_classwise / n_samples_classwise

        return n_correct_classwise, n_samples_classwise

    ###########################################################################################################################
    ################################################### Testing Loop ##########################################################
    ###########################################################################################################################

    num_classes = model.network[-1].__dict__['out_features']
    total_correct_class = torch.zeros(num_classes).to(device)
    total_samples_class = torch.zeros(num_classes).to(device)

    model.eval()
    if datasets['val'] is not None:
        model.load_state_dict(torch.load(os.path.join(model_path, 'log', 'best_state.pkl')))
    else:
        model.load_state_dict(torch.load(os.path.join(model_path, 'log', 'state.pkl')))

    if datasets['test'] is not None:
        test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size = batch_size, num_workers = num_workers)
    
    logger.info(f"\n\n\nTesting:\n")

    for batch_index, (inputs, labels) in enumerate(test_loader):

        t0 = time.time()
        n_correct_classwise, n_samples_classwise = test(inputs, labels)
        acc = torch.sum(n_correct_classwise) / len(inputs)
        total_correct_class += n_correct_classwise
        total_samples_class += n_samples_classwise
        t1 = time.time()

        logger.info(f'Test [{batch_index+1}/{len(test_loader)}] Time : {(t1-t0)*1e3:.1f} ms \t ACC={acc*100:.2f} %')

    total_accuracy = torch.sum(total_correct_class) / torch.sum(total_samples_class)
    logger.info(f'\n\nOverall Accuracy = {total_accuracy.item() * 100:.2f} %\n')
    print(f'\n\nTesting Accuracy = {total_accuracy.item() * 100:.2f} %')

    total_accuracy_class = total_correct_class / total_samples_class
    for i in range(num_classes):
        logger.info(f'Accuracy of class {i} {total_accuracy_class[i] * 100:.2f} %')

############################################################################################################################
################################################### Argument Parser ########################################################
############################################################################################################################


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--freq_cutoff", type=int, required=True)
    parser.add_argument("--interpolation", type=int, required=True)
    parser.add_argument("--rotate", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--restore", type=int, default = 0)
    parser.add_argument("--lr_decay_rate", type=float, default=0.5)
    parser.add_argument("--lr_decay_schedule", type=int, default=20)

    args = parser.parse_args()
    sys.path.append(args.__dict__['model_path'])
    from model import Model, get_datasets

    main(**args.__dict__)
