echo activating conda 'methane'...
call conda activate methane

if %errorlevel% neq 0 (
    echo.
    echo activating conda methane failed!
    goto :eof
)
echo activating done.
echo.

@REM echo training lstm ---
@REM python .\base_model.py --simulated --model lstm --type spatial --epoch 200
@REM echo.

@REM echo training pyraformer ---
@REM python .\base_model.py --simulated --model pyraformer --type spatial --epoch 200
@REM echo.

@REM echo training transformer ---
@REM python .\base_model.py --simulated --model transformer --type spatial --epoch 200
@REM echo.

@REM echo training hybrid ---
@REM python .\base_model.py --simulated --model hybrid --type spatial --epoch 200
@REM echo.

echo training hierarchy ---
python .\train_multi_embedding.py --simulated --type temporal --context 2 --epoch 200
echo.

echo training hierarchy ---
python .\train_multi_embedding.py --simulated --type temporal --context 4 --epoch 200
echo.

echo training hierarchy ---
python .\train_multi_embedding.py --simulated --type temporal --context 6 --epoch 200
echo.


echo training hierarchy ---
python .\train_multi_embedding.py --simulated --type temporal --context 8 --epoch 200
echo.

@REM echo training lstm ---
@REM python .\base_model.py --simulated --model lstm --type temporal --epoch 200
@REM echo.

@REM echo training pyraformer ---
@REM python .\base_model.py --simulated --model pyraformer --type spatial --epoch 200
@REM echo.

@REM echo training transformer ---
@REM python .\base_model.py --simulated --model transformer --type temporal --epoch 200
@REM echo.

@REM echo training hybrid ---
@REM python .\base_model.py --simulated --model hybrid --type temporal --epoch 200
@REM echo.

call conda deactivate