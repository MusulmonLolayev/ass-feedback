# For regression experiment
# python3 ./exp_reg.py --n_systems 60 --max_error 10 --removable_accuracy 0.4 --target_trunc 100 --seed 42

# For regression of one dataset
python3 ./exp_reg_one.py --dataset_name abalone --n_systems 20 --max_error 10 --removable_accuracy 0.4 --target_trunc 100 --seed 42

# For logistic regression experiment
# python3 ./exp_log_reg.py --dataset_name rejafada --n_systems 30 --a_acc 0.55 --b_acc 0.85 --ass_type nn --seed 0

# python3 ./exp_log_reg.py --dataset_name mushroom --n_systems 30 --a_acc 0.55 --b_acc 0.85 --ass_type nn --seed 0

# python3 ./exp_log_reg.py --dataset_name adult --n_systems 30 --a_acc 0.55 --b_acc 0.85 --ass_type nn --seed 42

# python3 ./exp_log_reg.py --dataset_name spam --n_systems 30 --a_acc 0.55 --b_acc 0.85 --ass_type nn --seed 42

# For softmax experiment
python3 ./exp_softmax.py --dataset_name students --n_systems 30 --a_acc 0.55 --b_acc 0.85 --ass_type nn --seed 42