echo activating conda 'methane'...
call conda activate methane

if %errorlevel% neq 0 (
    echo.
    echo activating conda methane failed!
    goto :eof
)
echo activating done.
echo.

echo training lstm ---
python .\base_model_finetune.py --model lstm
echo.

echo training pslstm ---
python .\base_model_finetune.py --model pslstm
echo.

echo training transformer ---
python .\base_model_finetune.py --model transformer
echo.


echo training pyraformer ---
python .\base_model_finetune.py --model pyraformer
echo.

echo training itransformer ---
python .\base_model_finetune.py --model itransformer
echo.

echo training tsmixer ---
python .\base_model_finetune.py --model tsmixer
echo.

echo training timemixer ---
python .\base_model_finetune.py --model timemixer
echo.

echo training duet ---
python .\base_model_finetune.py --model duet
echo.

echo training patchtst ---
python .\base_model_finetune.py --model patchtst
echo.

call conda deactivate