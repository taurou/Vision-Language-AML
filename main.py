import os
import sys
import logging
from parse_args import parse_arguments
from load_data import build_splits_baseline, build_splits_domain_disentangle, build_splits_clip_disentangle
from load_data_domgen import build_splits_domgen, build_splits_clip_disentangle_domgen
from experiments.baseline import BaselineExperiment
from experiments.domain_disentangle import DomainDisentangleExperiment
from experiments.clip_disentangle import CLIPDisentangleExperiment

def setup_experiment(opt):
    
    if opt['experiment'] == 'baseline':
        experiment = BaselineExperiment(opt)
        if not opt['dom_gen']:
            data = build_splits_baseline(opt)
            test_loader = data[2]
        else:
            data = build_splits_domgen(opt)
            test_loader = data[2]     
        
    elif opt['experiment'] == 'domain_disentangle':
        experiment = DomainDisentangleExperiment(opt)
        if not opt['dom_gen']:
            data = build_splits_domain_disentangle(opt)
            test_loader = data[3]
        else:
            data = build_splits_domgen(opt)
            test_loader = data[2]     

    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment(opt)
        if not opt['dom_gen']:
            data = build_splits_clip_disentangle(opt)
            test_loader = data[3]
        else:
            data = build_splits_clip_disentangle_domgen(opt)
            test_loader = data[2]     

    else:
        raise ValueError('Experiment not yet supported.')
    
    return (experiment, data, test_loader)

def main(opt):
    (experiment, data, test_loader) = setup_experiment(opt)

    # Train 
    if not opt['test']: # Skip training if '--test' flag is set
        iteration = 0
        best_accuracy = 0
        total_train_loss = 0

        # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt) # Log the hyperparameters, we save them in the checkpoint as well

        # Train loops
            if opt['experiment'] == 'baseline' or opt['dom_gen']:
                train_loader, validation_loader, test_loader = data # Unpack data 


                while iteration < opt['max_iterations']:    

                    for data in train_loader:
                        total_train_loss += experiment.train_iteration(data)

                        if iteration % opt['print_every'] == 0:
                            logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                            print(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                        
                        if iteration % opt['validate_every'] == 0:
                            # Run validation
                            val_accuracy, val_loss = experiment.validate(validation_loader)
                            logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                            print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')

                            # We save the best checkpoint based on the validation accuracy
                            if val_accuracy >= best_accuracy:   
                                best_accuracy = val_accuracy
                                experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                            # We also save the last checkpoint
                            experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                        iteration += 1  # One iteration = one batch
                        if iteration > opt['max_iterations']:
                            break

            elif opt['experiment'] != 'baseline' and not opt['dom_gen']:
                source_train_loader, target_train_loader, source_val_loader, test_loader = data
                target_train_loader_iter = iter(target_train_loader)

                while iteration < opt['max_iterations']:
                    for source_data in source_train_loader:
                        try:
                            target_data = next(target_train_loader_iter)
                        except StopIteration:
                            # Restarting the iterator if the source loader is bigger
                            target_train_loader_iter = iter(target_train_loader)
                            target_data = next(target_train_loader_iter)

                        #We compute once with the unlabeled target domain and once with the labeled source domain
                        total_train_loss += experiment.train_iteration(source_data, targetDomain = False)
                        total_train_loss += experiment.train_iteration(target_data, targetDomain = True)

                        if iteration % opt['print_every'] == 0:
                            logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                            print(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')

                        
                        if iteration % opt['validate_every'] == 0:
                            # Run validation
                            val_accuracy, val_loss = experiment.validate(source_val_loader)
                            logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                            print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')

                            if val_accuracy > best_accuracy:
                                best_accuracy = val_accuracy
                                experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                            experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                        iteration += 1
                        if iteration > opt['max_iterations']:
                            break
    

    # Test on BEST checkpoint
    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    test_accuracy, _ = experiment.validate(test_loader)
    logging.info(f'[TEST] Best checkpoint Accuracy: {(100 * test_accuracy):.2f}')
    print(f'[TEST] Best checkpoint Accuracy: {(100 * test_accuracy):.2f}')
    
    # Test on LAST checkpoint
    experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
    test_accuracy, _ = experiment.validate(test_loader)
    logging.info(f'[TEST] Last checkpoint Accuracy: {(100 * test_accuracy):.2f}')
    print(f'[TEST] Last checkpoint Accuracy: {(100 * test_accuracy):.2f}')

if __name__ == '__main__':

    
    opt = parse_arguments() 

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')
    
    

    # Run experiment
    main(opt)
