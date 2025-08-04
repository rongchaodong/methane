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
@REM python .\base_model.py --simulated --model lstm --type temporal --epoch 200
@REM echo.

@REM echo training pyraformer ---
@REM python .\base_model.py --simulated --model pyraformer --type temporal
@REM echo.

@REM echo training transformer ---
@REM python .\base_model.py --simulated --model transformer --type temporal
@REM echo.

@REM echo training hybrid ---
@REM python .\base_model.py --simulated --model hybrid --type temporal
@REM echo.

@REM echo training hierachy ---
@REM python .\base_model.py --simulated --model hierachy --model_type seq --type temporal --context 2
@REM echo.

echo training hierachy ---
python .\base_model.py --simulated --model hierachy --model_type seq --type temporal --context 4 --epoch 50
echo.

@REM echo training hierachy ---
@REM python .\base_model.py --simulated --model hierachy --model_type seq --type temporal --context 6
@REM echo.


echo training hierachy ---
python .\base_model.py --simulated --model hierachy --model_type seq --type temporal --context 8 --epoch 50
echo.

call conda deactivate