@echo off
echo activating conda 'methane'...
call conda activate methane

if %errorlevel% neq 0 (
    echo.
    echo activating conda methane failed!
    goto :eof
)
echo activating done.
echo.

for %%m in (lstm pslstm transformer pyraformer itransformer tsmixer timemixer duet patchtst) do (
    echo training %%m ---
    for /L %%i in (1,1,3) do (
        python .\base_model_finetune.py --model %%m
    )
    echo.
)

echo training hierarchy ---
for /L %%i in (1,1,3) do (
    python train_learnable_hierarchical_fluxnet_finetune.py --pretrained
)

call conda deactivate