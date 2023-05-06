#!/bin/bash

# Define the parameter values to search

shuffle_values=("True")
device_id_values=(0)
model_values=("resnet50")
optuna_values=("False")
optuna_trials_values=(5)
optimizer_values=("SGD")
smote_k_values=(5)
smote_values=("True" "False")
size_values=("\"(299,299)\"")
bz_values=(8 16)
norm_mean_values=("\"(0.3272, 0.2874, 0.2038)\"" "\"(0.485, 0.456, 0.406)\"")
norm_std_values=("\"(0.0965, 0.1009, 0.1173)\"" "\"(0.229, 0.224, 0.225)\"")
brightness_values=("\"(0.8,2)\"" "\"(0.8,1.5)\"")
noise_std_values=(0.05 0.1)
scheduler_values=("cosineannealing" "steplr")
lr_values=(0.01 0.001 0.1)
momentum_values=(0.9 0.8)
weight_decay_values=(1e-4 1e-5 1e-6)
freeze_num_values=(7 5 3)
epochs_values=(30)
patience_values=(30)
snapshot_ensemble_values=("True" "False")
log_values=("True")
estimators_values=(5 10)
voting_values=("majority" "average")

# Iterate over all combinations of parameter values
for smote in "${smote_values[@]}"; do
  for smote_k in "${smote_k_values[@]}"; do
    for size in "${size_values[@]}"; do
      for bz in "${bz_values[@]}"; do
        for norm_mean in "${norm_mean_values[@]}"; do
          for norm_std in "${norm_std_values[@]}"; do
            for brightness in "${brightness_values[@]}"; do
              for noise_std in "${noise_std_values[@]}"; do
                for shuffle in "${shuffle_values[@]}"; do
                  for device_id in "${device_id_values[@]}"; do
                    for model in "${model_values[@]}"; do
                      for optuna in "${optuna_values[@]}"; do
                        for optuna_trials in "${optuna_trials_values[@]}"; do
                          for optimizer in "${optimizer_values[@]}"; do
                            for scheduler in "${scheduler_values[@]}"; do
                              for lr in "${lr_values[@]}"; do
                                for momentum in "${momentum_values[@]}"; do
                                  for weight_decay in "${weight_decay_values[@]}"; do
                                    for freeze_num in "${freeze_num_values[@]}"; do
                                      for epochs in "${epochs_values[@]}"; do
                                        for patience in "${patience_values[@]}"; do
                                          for snapshot_ensemble in "${snapshot_ensemble_values[@]}"; do
                                            for log in "${log_values[@]}"; do
                                              for estimators in "${estimators_values[@]}"; do
                                                for voting in "${voting_values[@]}"; do
                                                  # Run your command with the current parameter values
                                                  command="python run.py --path data/train \
                                                    --smote $smote \
                                                    --smote_k $smote_k \
                                                    --size $size \
                                                    --bz $bz \
                                                    --norm_mean $norm_mean \
                                                    --norm_std $norm_std \
                                                    --brightness $brightness \
                                                    --noise_std $noise_std \
                                                    --shuffle $shuffle \
                                                    --device_id $device_id \
                                                    --model $model \
                                                    --reuse_model '' \
                                                    --optuna $optuna \
                                                    --optuna_trials $optuna_trials \
                                                    --optimizer $optimizer \
                                                    --scheduler $scheduler \
                                                    --lr $lr \
                                                    --momentum $momentum \
                                                    --weight_decay $weight_decay \
                                                    --freeze_num $freeze_num \
                                                    --epochs $epochs \
                                                    --patience $patience \
                                                    --snapshot_ensemble $snapshot_ensemble \
                                                    --log $log \
                                                    --estimators $estimators \
                                                    --voting $voting"
                                                  echo "Running command: $command"
                                                  # Uncomment the following line to execute the command
                                                  eval "$command"
                                                done
                                              done
                                            done
                                          done
                                        done
                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
