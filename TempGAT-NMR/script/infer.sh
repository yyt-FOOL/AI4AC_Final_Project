data_path="../data/test" # replace to your data path
finetune_model_dir="../../../output_data/TempGAT-NMR/finetune/5cv/F_pretraining_molecular_global__kener__atomdes__unimol_large_atom_regloss_mse_lr_0.0001_bs_16_0.06_5"
save_dir="../../../output_data/TempGAT-NMR/test/5cv/F_pretraining_molecular_global__kener__atomdes__unimol_large_atom_regloss_mse_lr_0.0001_bs_16_0.06_5"
if [ -d "${save_dir}" ]; then
    rm -rf ${save_dir}
    echo "Folder remove at: ${save_dir}"
fi
mkdir -p ${save_dir}
echo "Folder created at: ${save_dir}"
nfolds=5
for fold in $(seq 0 $(($nfolds - 1)))
    do
    echo "Start infering..."
    cv_seed=42
    python ../uninmr_fnmr/infer.py --user-dir  ../uninmr_fnmr ${data_path}   --valid-subset valid \
        --results-path "${save_dir}/${fold}"  --saved-dir $finetune_model_dir \
        --num-workers 8 --ddp-backend=no_c10d --distributed-world-size 1 --batch-size 16 \
        --task uninmr --loss 'atom_regloss_mae' --arch 'unimol_large' \
        --dict-name "dict.txt" \
        --use-gat \
        --path ${finetune_model_dir}/cv_seed_${cv_seed}_fold_${fold}/checkpoint_best.pt \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
        --selected-atom 'F'   --gaussian-kernel   --atom-descriptor 0 --split-mode infer 
    done 2>&1 | tee "${save_dir}/infer.log"

python ../uninmr_fnmr/utils/get_result.py --path $save_dir  --mode cv --dict "${data_path}/dict.txt" 2>&1 | tee "${save_dir}/result.log"
