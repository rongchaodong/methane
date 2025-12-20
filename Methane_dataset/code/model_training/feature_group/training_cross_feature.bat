echo activating conda 'methane'...
call conda activate methane

if %errorlevel% neq 0 (
    echo.
    echo activating conda methane failed!
    goto :eof
)
echo activating done.
echo.

python train_cross_feature.py --type temporal --simulated --data_type seq --epoch 100 --ablation no_intra

python train_cross_feature.py --type temporal --simulated --data_type seq --epoch 100 --ablation no_inter

python train_cross_feature.py --type temporal --data_type seq --epoch 10000 --ablation no_intra

python train_cross_feature.py --type temporal --data_type seq --epoch 10000 --ablation no_inter

call conda deactivate