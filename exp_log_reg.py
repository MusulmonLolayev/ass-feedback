import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
import pickle
import random

from utils.reg_log_utils import (generate_system_features, \
   generate_system, label_log_reg, train_assessor,\
                  simple_assessor, SVMScikit, one_hot_encoding_system,\
                  accuracy_correction, constraint_gpu, release_gpu, \
                  F_NAMES, LOSSES, plot_regression_exp3,
                  load_dataset, scores)

def experiment(
        dataset_name: str = 'rejafada',
        n_systems: int = 50,
        a_acc: int = 0.5,
        b_acc: int = 0.85,
        ass_type: str = 'nn',
        fit_from_train: bool = True,
        seed: int = 42,
        decision_threshold: float = 0.5,
        eps = [0.1, 0.2, 0.3, 0.4, 0.5]):
    
    # Load dataset
    X_train, X_val, X_test, y_train, y_val, y_test = \
      load_dataset(dataset_name, seed)

    # For just speeding testing
    # indicies = np.random.randint(0, X_train.shape[1], 1000)
    # X_train = X_train[:, indicies]
    # X_val = X_val[:, indicies]
    # X_test = X_test[:, indicies]

    # setup seed to make sure it must be non zero
    # to reproduce
    if seed:
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    n_train, n_features = X_train.shape
    print(f"Number of objects in train: {n_train}")
    print(f"Number of features: {n_features}")

    # systems' groups: 
    # 0 - the just best groups; 
    # 1 - loss=nan; 
    # 2 - low error accuracy
    y_system = np.zeros((n_systems), dtype=int)

    # Generate system features
    x_system = generate_system_features(n_systems)
    n_system_features = x_system.shape[1]

    # Generate systems: simple NN models
    systems = []
    for i in range(n_systems):
        systems.append(generate_system(x_system[i], n_features))
    
    # Train systems
    print('*'*20, 'Training systems', '*'*20)
    for i, model in enumerate(systems):
        print(f"Training model: {i + 1}")
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy',
            mode='min',
            min_delta=1e-3,
            patience=3,
            restore_best_weights=True)

        history = model.fit(
            x=X_train,
            y=y_train,
            epochs=int(x_system[i, F_NAMES['epoch']]),
            callbacks=[callback],
            batch_size=int(x_system[i, F_NAMES['bs']]),
            shuffle=True)
        
        best_loss = np.min(history.history['loss'])
        x_system[i, F_NAMES['tr_loss']] = best_loss
        print(f"""Model best loss ({LOSSES[int(x_system[i, F_NAMES['loss']])]}): {best_loss}, 
        epoch: {np.argmin(history.history['loss'])}""")
        if np.isnan(best_loss):
            y_system[i] = 1
            # no epochs
            x_system[i, F_NAMES['epoch']] = np.nan
        else:
            # retrieving the best epoch 
            x_system[i, F_NAMES['epoch']] = np.argmin(history.history['loss'])
        
        tf.keras.backend.clear_session()
    
    print('*'*20, "Compute system error accuracy", '*'*20)
    # Put systems' accuracy
    for i, model in enumerate(systems):
        print(f"Model: {i + 1}")
        # Checking if model loss is not nan
        if y_system[i] == 0:
            # Calculate training accuracy
            y_pred = model.predict(X_train)[:, 0]
            y_pred = label_log_reg(y_pred.copy(), threshold=decision_threshold)
            x_system[i, F_NAMES['tr_acc']] = metrics.balanced_accuracy_score(y_train, y_pred)
            
            # Calculate loss and validation accuracy
            y_pred = model.predict(X_val)
            loss_f = LOSSES[int(x_system[i, F_NAMES['loss']])]()
            x_system[i, F_NAMES['val_loss']] = loss_f(y_val, y_pred)
            
            y_pred = y_pred[:, 0]
            y_pred = label_log_reg(y_pred.copy(), threshold=decision_threshold)
            x_system[i, F_NAMES['val_acc']] = metrics.balanced_accuracy_score(y_val, y_pred)

            # if model accuracy is less than removable accuracy, 
            # them mark system targets
            if x_system[i, F_NAMES['val_acc']] < a_acc or \
              x_system[i, F_NAMES['val_acc']] > b_acc:
                y_system[i] = 2
        else:
            print(f"Model {i + 1} loss is nan")
    print("System training accuracies: ", x_system[:, [F_NAMES['tr_acc'], F_NAMES['val_acc']]])
        
    # Backup the original systems features, before dropping them
    x_system_backup = x_system.copy()
    y_system_backup = y_system.copy()
    # and save it
    np.save('./file-results/log_reg/x_system_backup.npy', x_system_backup)
    np.save('./file-results/log_reg/y_system_backup.npy', y_system)

    print("Remove systems with loss=nan or low interval error accuracy")
    # Remove systems with loss=nan or low interval error accuracy
    # for avoiding waiting much time to compute
    # Filtering systems
    new_models = []
    for i, model in enumerate(systems):
        if y_system[i] == 0:
          new_models.append(model)
    systems = new_models

    x_system = x_system[y_system == 0]
    y_system = y_system[y_system == 0]
    # New system shape
    print(f"Initially, num. systems: {n_systems}, Now, {x_system.shape[0]}")
    n_systems = x_system.shape[0]

    # save a new x_system
    np.save('./file-results/log_reg/x_system.npy', x_system)
    np.save('./file-results/log_reg/y_system.npy', x_system)

    # Coverting categorieal variables via one-hot encoding
    one_hot_system = one_hot_encoding_system(x_system)
    print("Number of system features (one-hot):", one_hot_system.shape[1])
    
    if ass_type == 'nn':
        ass_model = simple_assessor(n_features, one_hot_system.shape[1])
    elif ass_type == 'svm':
        ass_model = SVMScikit()
    else:
        raise NotImplemented
    # Training assessor on the training set
    if fit_from_train:
        train_assessor(X_train, 
                       y_train, 
                       one_hot_system, 
                       systems, 
                       ass_model, 
                       random=True)
    
    # Training assessor on the validation set
    print('='*20, "Training assessor on the validation set", '='*20)
    train_assessor(X_val, 
                   y_val, 
                   one_hot_system, 
                   systems, 
                   ass_model, 
                   random=False, 
                   n_updates=5)
    
    # Producing the experiment 1
    print('*'*20, 'Producing the experiment 1', '*'*20)
    ass_mean_error = np.zeros((n_systems, ))
    for i, model in enumerate(systems):

        print(f"Model: {i + 1}, acc: {x_system[i, F_NAMES['tr_acc']]}")
        print(f"Loss func: {LOSSES[int(x_system[i, F_NAMES['loss']])]}")

        # Assessor prediction
        x_system_test = np.zeros((X_test.shape[0], 
                                  one_hot_system.shape[1]))
        x_system_test[np.arange(X_test.shape[0]), :] = one_hot_system[i]
        y_pred_delta = ass_model.predict(
            [X_test, x_system_test], 
            verbose=0)[:, 0]
        print(sum(np.abs(y_pred_delta) < 0.5) / n_train)
        print(sum(np.abs(y_pred_delta) > 1) / n_train)
        # Model prediction
        model.evaluate(X_test, y_test)
        y_pred_test = model.predict(X_test, verbose=0)
        y_delta_test = y_test - y_pred_test[:, 0]

        print(True in np.isnan(y_pred_delta))
        print(True in np.isnan(y_delta_test))

        ass_model.evaluate([X_test, x_system_test], y_delta_test)
        
        # Compute and store all accuracies
        ass_mean_error[i] = np.abs(y_pred_delta - y_delta_test).mean()
    
    print("Experiment 1:", ass_mean_error)
    print("Min", ass_mean_error.argmin(), ass_mean_error.min())
    print("Max", ass_mean_error.argmax(), ass_mean_error.max())

    # save results as raw pickle file
    with open(f'./file-results/log_reg/{dataset_name}_exp1.pkl', 'wb') as f:
        pickle.dump(ass_mean_error, f)

    # Producing the second experiment:
    # Select best models for each instance
    
    s_predictions = np.zeros((X_test.shape[0], n_systems))
    a_predictions = np.zeros((X_test.shape[0], n_systems))
    input_x_system = np.zeros((X_test.shape[0], one_hot_system.shape[1]))

    for j in range(n_systems):
        system = systems[j]
        s_predictions[:, j] = system.predict(X_test)[:, 0]
        
        input_x_system[np.arange(X_test.shape[0])] = one_hot_system[j]
        a_predictions[:, j] = ass_model.predict([X_test, input_x_system])[:, 0]
    
    # prepare data for ploting
    # Get the best and worst accuracies
    best_system_ind = np.argmax(x_system[:, F_NAMES['val_acc']])
    worst_system_ind = np.argmin(x_system[:, F_NAMES['val_acc']])

    # the best and worst systems
    y_best_pred = s_predictions[:, best_system_ind]
    y_best_pred_error = a_predictions[:, best_system_ind]    
    y_worst_pred = s_predictions[:, worst_system_ind]
    y_worst_pred_error = a_predictions[:, worst_system_ind]

    # selected systems by the assessor for each instance
    ass_selected_system_indicies = np.argmin(
        np.abs(a_predictions),
        axis=1)
    y_sel_pred = s_predictions[np.arange(X_test.shape[0]),
                                        ass_selected_system_indicies]
    y_sel_pred_error = a_predictions[np.arange(X_test.shape[0]),
                                        ass_selected_system_indicies]
    best_accs = np.zeros((len(eps), 5))
    corr_best_accs = np.zeros((len(eps), 5))
    worst_accs = np.zeros((len(eps), 5))
    corr_worst_accs = np.zeros((len(eps), 5))
    sel_accs = np.zeros((len(eps), 5))
    corr_sel_accs = np.zeros((len(eps), 5))

    for i, ep in enumerate(eps):
        y_best_pred_label, y_corr_best_pred_label = \
          accuracy_correction(y_best_pred, y_best_pred_error, ep)
        y_worst_pred_label, y_corr_worst_pred_label = \
          accuracy_correction(y_worst_pred, y_worst_pred_error, ep, is_worst=True)
        y_sel_pred_label, y_sel_worst_pred_label = \
          accuracy_correction(y_sel_pred, y_sel_pred_error, ep)
        
        scores(y_test, y_best_pred_label, best_accs[i])
        scores(y_test, y_corr_best_pred_label, corr_best_accs[i])
        scores(y_test, y_worst_pred_label, worst_accs[i])
        scores(y_test, y_corr_worst_pred_label, corr_worst_accs[i])
        scores(y_test, y_sel_pred_label, sel_accs[i])
        scores(y_test, y_sel_worst_pred_label, corr_sel_accs[i])

    exp_2 = {
        'best_accs': best_accs,
        'corr_best_accs': corr_best_accs,
        'worst_accs': worst_accs,
        'corr_worst_accs': corr_worst_accs,
        'sel_accs': sel_accs,
        'corr_sel_accs': corr_sel_accs}
    print("The second experiment")
    for key, val in exp_2.items():
        print(key, '\n', val)

    # save results as raw pickle file
    with open(f'./file-results/log_reg/{dataset_name}_exp2.pkl', 'wb') as f:
        pickle.dump(exp_2, f)

    # Plot the third exp
    plot_regression_exp3(x_system_backup, y_system_backup, a_acc, b_acc)
    

if __name__ == '__main__':
    
    constraint_gpu(tf, 3)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', 
                        default='rejafada', 
                        type=str,
                        choices=['rejafada', 'mushroom', 'adult', 'spam'])
    parser.add_argument('--n_systems', default=50, type=int)
    parser.add_argument('--a_acc', default=0.5, type=float)
    parser.add_argument('--b_acc', default=0.85, type=float)
    parser.add_argument('--ass_type', default='nn', type=str)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    experiment(dataset_name=args.dataset_name,
              n_systems=args.n_systems,
              a_acc=args.a_acc,
              b_acc=args.b_acc,
              ass_type=args.ass_type,
              seed=args.seed)
  
    # release_gpu()
    