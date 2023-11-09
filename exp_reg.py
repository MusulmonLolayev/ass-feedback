import numpy as np
import os
import random
from tensorflow.keras.utils import plot_model

import sklearn.metrics as metrics

from utils.reg_utils import (generate_system_features, \
                             generate_system, train_assessor, one_hot_encoding_system, \
                  simple_assessor_reg, plot_regression_exp1, \
                  plot_regression_exp2, constraint_gpu, release_gpu, scaling, \
                  F_NAMES, LOSSES, plot_regression_exp3, rate_system)

import tensorflow as tf
import glob

import pickle

def experiment(
        n_systems: int = 50,
        max_error: int = 10,
        removable_accuracy: int = 0.4,
        target_trunc: int = 100,
        seed: int = 42,
        fit_from_train: bool = True):
    
    RES_DIR = f'./results/regression/' + \
      f'({n_systems}, {max_error}, {removable_accuracy}, {target_trunc}, {seed}, {fit_from_train})'

    # Create folder for results
    # Delete if it exists
    if not os.path.exists(RES_DIR):
        os.mkdir(RES_DIR)

    # setup seed to make sure it must be non zero
    # to reproduce
    if seed:
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # Load dataset
    path = 'datasets/BlogFeedback/blogData_train.csv'

    data = np.loadtxt(path, delimiter=',')
    X_train = data[:, :data.shape[1] - 1]
    y_train = data[:, data.shape[1] - 1]

    # scaling
    sc_denom = X_train.max(axis=0)
    scaling(X_train, sc_denom)

    # Truncating examples by their targets
    if target_trunc:
        cond = y_train < target_trunc
        X_train = X_train[cond]
        y_train = y_train[cond]

    # for seepding up in only test cases (to check code is correct)
    indicies = np.random.randint(0, X_train.shape[0], 3000)
    X_train = X_train[indicies]
    y_train = y_train[indicies]

    n_train, n_features = X_train.shape
    print(f"Number of objects in train: {n_train}")
    print(f"Number of features: {n_features}")

    # systems' groups: 
    # 0 - best groups; 
    # 1 - loss=nan; 
    # 2 - low interval error accuracy
    y_system = np.zeros((n_systems), dtype=int)

    # Generate system features
    x_system = generate_system_features(n_systems)

    # Generate systems: simple NN models
    systems = []
    for i in range(n_systems):
        systems.append(generate_system(x_system[i], n_features))
    
    # Train systems
    print('Training systems')
    for i, model in enumerate(systems):
        print(f"Training model: {i + 1}")
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='mean_absolute_error',
            min_delta=1e-3,
            patience=4,
            restore_best_weights=True)

        history = model.fit(
            x=X_train,
            y=y_train,
            epochs=int(x_system[i, F_NAMES['epoch']]),
            callbacks=[callback],
            batch_size=int(x_system[i, F_NAMES['bs']]),
            shuffle=True,
            verbose=2)
        
        best_loss = np.min(history.history['loss'])
        print(f"""Model best loss ({LOSSES[int(x_system[i, F_NAMES['loss']])]}): {best_loss}, 
        epoch: {np.argmin(history.history['loss'])}""")
        if np.isnan(best_loss):
            y_system[i] = 1
            # no epochs
            x_system[i, F_NAMES['epoch']] = 0
        else:
            # retrieving the best epoch 
            x_system[i, F_NAMES['epoch']] = np.argmin(history.history['mean_absolute_error'])
        
        # model.save(RES_DIR + f'/{i + 1}')
        tf.keras.backend.clear_session()
    
    test_paths = glob.glob(
        str('./datasets/BlogFeedback/tests/*.csv')
    )
    data = np.loadtxt(test_paths[0], delimiter=',')
    X_val = data[:, :data.shape[1] - 1]
    y_val = data[:, data.shape[1] - 1]

    # Truncating examples by their targets
    if target_trunc:
        cond = y_val < target_trunc
        X_val = X_val[cond]
        y_val = y_val[cond]
    # scaling
    scaling(X_val, sc_denom)

    print("Compute system error accuracy")
    # Put systems' interval error accuracy
    for i, model in enumerate(systems):
        print(f"Model: {i + 1}")
        # Checking if model loss is not nan
        if y_system[i] == 0:
            
            # training accuracy
            y_pred = model.predict(X_train, verbose=0)
            x_system[i, F_NAMES['rate']] = rate_system(y_train, y_pred[:, 0], max_error)
            x_system[i, F_NAMES['mse']] = metrics.mean_squared_error(
                y_train, 
                y_pred[:, 0])
            
            # val accuracy
            y_pred = model.predict(X_val, verbose=0)
            x_system[i, F_NAMES['val_rate']] = rate_system(y_val, y_pred[:, 0], max_error)
            x_system[i, F_NAMES['mse_val']] = metrics.mean_squared_error(y_val, y_pred[:, 0])

            # label system targets
            if x_system[i, F_NAMES['val_rate']] < removable_accuracy:
                y_system[i] = 2
        else:
            print(f"Model {i + 1} loss is nan")
    

    # Backup the original systems features, before dropping them
    x_system_backup = x_system.copy()
    # and save it
    # np.save(f'{RES_DIR}/x_system_backup.npy', x_system_backup)
    # np.save(f'{RES_DIR}/y_system_backup.npy', y_system)

    print("Remove systems with loss=nan or low interval error accuracy")
    # Remove systems with loss=nan or low interval error accuracy
    # for avoiding waiting much time to compute
    x_system = x_system[y_system == 0]
    print(f"Initially, num. systems: {n_systems}, Now, {x_system.shape[0]}")

    # save the new x_system
    np.save(f'{RES_DIR}/x_system.npy', x_system)
    
    new_models = []
    for i, model in enumerate(systems):
        if y_system[i] == 0:
          new_models.append(model)
    systems = new_models
    # New system shape
    n_systems = x_system.shape[0]
    print(x_system[:, F_NAMES['val_rate']])

    print(np.max(x_system[:, F_NAMES['val_rate']]))
    print(np.min(x_system[:, F_NAMES['val_rate']]))
    # Get the best and worst accuracies
    best_system_ind = np.argmax(x_system[:, F_NAMES['val_rate']])
    worst_system_ind = np.argmin(x_system[:, F_NAMES['val_rate']])

    print(best_system_ind, worst_system_ind)
    print(x_system[:, F_NAMES['unit']])
    # input('Start training assessor model')

    x_system_ = one_hot_encoding_system(x_system)
    n_system_features = x_system_.shape[1]

    ass_model = simple_assessor_reg(n_features, n_system_features)
    plot_model(ass_model,  to_file='model.png')

    if fit_from_train:
        train_assessor(X_train, y_train, 
                       x_system_, x_system, systems, 
                       ass_model,
                       max_error,
                       random=True)
    
    train_assessor(X_val, y_val, 
                  x_system_, x_system, systems, 
                  ass_model,
                  max_error,
                  random=False)

    # input('Start Experiment 1')

    test_paths.remove(test_paths[0])
    n_test = len(test_paths)
    all_acc = []
    for i, path in enumerate(test_paths):
        print(f"Iter: {i + 1}, Test systems: {path}")
        data = np.loadtxt(path, delimiter=',')
        X_test = data[:, :data.shape[1] - 1]
        y_test = data[:, data.shape[1] - 1]

        # Truncating examples by their targets
        if target_trunc:
            cond = y_test < target_trunc
            X_test = X_test[cond]
            y_test = y_test[cond]

        # scaling
        scaling(X_test, sc_denom)

        acc = np.zeros((n_systems, 5))
        for i, model in enumerate(systems):

            print(f"Model: {i + 1}")

            # Assessor prediction
            x_system_test = np.zeros((X_test.shape[0], 
                                      n_system_features))
            x_system_test[np.arange(X_test.shape[0]), :] = x_system_[i]
            y_pred_error = ass_model.predict(
                [X_test, x_system_test], 
                verbose=0)[:, 0]

            # Model prediction: true error
            y_pred_test = model.predict(X_test, verbose=0)
            y_true_error = y_test - y_pred_test[:, 0]
            
            y_interval_true = np.abs(y_true_error) < max_error
            y_interval_pred = np.abs(y_pred_error) < max_error

            # Compute and store all accuracies
            acc[i, 0] = metrics.accuracy_score(y_interval_true, y_interval_pred)
            acc[i, 1] = metrics.balanced_accuracy_score(y_interval_true, y_interval_pred)
            acc[i, 2] = metrics.f1_score(y_interval_true, y_interval_pred,  zero_division=0.0)
            acc[i, 3] = metrics.precision_score(y_interval_true, y_interval_pred, zero_division=0.0)
            acc[i, 4] = metrics.recall_score(y_interval_true, y_interval_pred, zero_division=0.0)
            
            print(acc[i])
        all_acc.append(acc)
    # save results as raw pickle file
    with open(f'{RES_DIR}/reg_all_acc_exp1.pkl', 'wb') as f:
        pickle.dump(all_acc, f)

    best_tendency = np.zeros((n_test, 5))
    worst_tendency = np.zeros((n_test, 5))

    for i, accuracy in enumerate(all_acc):
        best_tendency[i] = accuracy[best_system_ind]
        worst_tendency[i] = accuracy[worst_system_ind]

    plot_regression_exp1(best_tendency, f'{RES_DIR}/best_reg_exp1')
    plot_regression_exp1(worst_tendency, f'{RES_DIR}/worst_reg_exp1')

    # # save results as raw pickle file
    with open(f'{RES_DIR}/best_reg_exp1.pkl', 'wb') as f:
        pickle.dump(best_tendency, f)
    with open(f'{RES_DIR}/worst_reg_exp1.pkl', 'wb') as f:
        pickle.dump(worst_tendency, f)

    # Producing the second experiment:
    # Select best models for each instance
    i = 0
    system_predictions = []
    assessor_predictions = []

    # Collect predictions
    for i, path in enumerate(test_paths):
        print(f"Iter: {i + 1}, Test systems: {path}")
        data = np.loadtxt(path, delimiter=',')
        X_test = data[:, :data.shape[1] - 1]
        y_test = data[:, data.shape[1] - 1]

        # scaling
        scaling(X_test, sc_denom)

        # Truncating examples by their targets
        if target_trunc:
            cond = y_test < target_trunc
            X_test = X_test[cond]
            y_test = y_test[cond]

        s_predictions = np.zeros((X_test.shape[0], n_systems))
        a_predictions = np.zeros((X_test.shape[0], n_systems))
        input_x_system = np.zeros((X_test.shape[0], n_system_features))

        for j, system in enumerate(systems):
            s_predictions[:, j] = system.predict(X_test, verbose=0)[:, 0]
            
            input_x_system[np.arange(X_test.shape[0])] = x_system_[j]
            a_predictions[:, j] = ass_model.predict([X_test, input_x_system],
                                                    verbose=0)[:, 0]

        system_predictions.append(s_predictions)
        assessor_predictions.append(a_predictions)
    
    # prepare data for ploting
    # Get the best and worst accuracies
    best_system_ind = np.argmax(x_system[:, F_NAMES['val_rate']])
    worst_system_ind = np.argmin(x_system[:, F_NAMES['val_rate']])

    best_errors = []
    best_corrector_errors = []
    worst_errors = []
    worst_corrector_errors = []
    sel_errors = []
    sel_corrector_errors = []
    truncations = []

    for i, path in enumerate(test_paths):
        print(f"Iter: {i + 1}, Test systems: {path}")
        # print(f"Iter: {i + 1}")
        data = np.loadtxt(path, delimiter=',')
        X_test = data[:, :data.shape[1] - 1]
        y_test = data[:, data.shape[1] - 1]
        # scaling
        scaling(X_test, sc_denom)

        # Truncating examples by their targets
        if target_trunc:
            cond = y_test < target_trunc
            X_test = X_test[cond]
            y_test = y_test[cond]

        y_best_system = system_predictions[i][:, best_system_ind]
        y_best_pred_error = assessor_predictions[i][:, best_system_ind]    

        y_worst_system = system_predictions[i][:, worst_system_ind]
        y_worst_pred_error = assessor_predictions[i][:, worst_system_ind]

        ass_selected_system_indicies = np.argmin(
            np.abs(assessor_predictions[i]),
            axis=1)
        y_sel_system = system_predictions[i][np.arange(X_test.shape[0]),
                                            ass_selected_system_indicies]
        y_sel_pred_error = assessor_predictions[i][np.arange(X_test.shape[0]),
                                            ass_selected_system_indicies]        

        y_best_error = np.abs(y_test - y_best_system)
        y_worst_error = np.abs(y_test - y_worst_system)
        y_sel_error = np.abs(y_test - y_sel_system)

        # truncating higher assessor results
        print('Truncations:')
        trun = (np.mean(np.abs(y_best_pred_error) <= max_error),
                np.mean(np.abs(y_worst_pred_error) <= max_error),
                np.mean(np.abs(y_sel_pred_error) <= max_error)
            )
        print('%.4f, %.4f, %.4f' % trun)
        truncations.append(trun)

        y_best_pred_error[np.abs(y_best_pred_error) > 0.75 * max_error] = 0
        y_worst_pred_error[np.abs(y_worst_pred_error) < 0.5 * max_error] = 0
        y_sel_pred_error[np.abs(y_sel_pred_error) > 0.75 * max_error] = 0

        y_best_error_corrector = np.abs(y_test - y_best_system - y_best_pred_error)
        y_worst_error_corrector = np.abs(y_test - y_worst_system - y_worst_pred_error)
        y_sel_error_corrector = np.abs(y_test - y_sel_system - y_sel_pred_error)

        best_errors.append(np.mean(y_best_error))
        best_corrector_errors.append(np.mean(y_best_error_corrector))
        worst_errors.append(np.mean(y_worst_error))
        worst_corrector_errors.append(np.mean(y_worst_error_corrector))
        sel_errors.append(np.mean(y_sel_error))
        sel_corrector_errors.append(np.mean(y_sel_error_corrector))

    exp_2 = {
        'best_errors': best_errors,
        'best_corrector_errors': best_corrector_errors,
        'worst_errors': worst_errors,
        'worst_corrector_errors': worst_corrector_errors,
        'sel_errors': sel_errors,
        'sel_corrector_errors': sel_corrector_errors}
    # save results as raw pickle file
    with open(f'{RES_DIR}/exp2.pkl', 'wb') as f:
        pickle.dump(exp_2, f)
    
    with open(f'{RES_DIR}/test_truncation.pkl', 'wb') as f:
        pickle.dump(truncations, f)
    # Plot
    plot_regression_exp2(
        best_errors,
        best_corrector_errors,
        worst_errors,
        worst_corrector_errors,
        sel_errors,
        sel_corrector_errors,
        RES_DIR)

    # Plot the third exp
    plot_regression_exp3(x_system_backup, y_system, removable_accuracy, RES_DIR)
    

if __name__ == '__main__':
    
    constraint_gpu(tf, 3)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_systems', default=50, type=int)
    parser.add_argument('--max_error', default=10, type=float)
    parser.add_argument('--removable_accuracy', default=0.5, type=float)
    parser.add_argument('--target_trunc', default=100, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    experiment(n_systems=args.n_systems,
               max_error=args.max_error,
               removable_accuracy=args.removable_accuracy,
               target_trunc=args.target_trunc,
               seed=args.seed)
  
    # release_gpu()
    