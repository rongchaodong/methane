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
@REM python .\base_model.py --simulated --model pyraformer --type spatial --epoch 200
@REM echo.

@REM echo training transformer ---
@REM python .\base_model.py --model transformer --type temporal --simulated
@REM echo.

@REM echo training hybrid ---
@REM python .\base_model.py --model hybrid --type temporal --simulated
@REM echo.

@REM echo training hierarchy ---
@REM python .\base_model.py --simulated --model hierarchy --type temporal --context 2 --epoch 200
@REM echo.

@REM echo training hierarchy ---
@REM python .\base_model.py --simulated --model hierarchy --type temporal --context 4 --epoch 200
@REM echo.

@REM echo training hierarchy ---
@REM python .\base_model.py --simulated --model hierarchy --type temporal --context 6 --epoch 200
@REM echo.

@REM echo training hierarchy ---
@REM python .\base_model.py --simulated --model hierarchy --type temporal --context 8 --epoch 200
@REM echo.

@REM echo training hierarchy ---
@REM python .\train_x_y.py --simulated --type temporal --context 2 --epoch 200
@REM echo.

@REM echo training hierarchy ---
@REM python .\train_x_y.py --simulated --type temporal --context 4 --epoch 200
@REM echo.

@REM echo training hierarchy ---
@REM python .\train_x_y.py --simulated --type temporal --context 6 --epoch 200
@REM echo.


@REM echo training hierarchy ---
@REM python .\train_x_y.py --simulated --type temporal --context 8 --epoch 200
@REM echo.

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

echo training hierarchy ---
python train_learnable_hierarchical.py --type temporal --simulated --data_type 1percent --epoch 200 --context_years 2
echo.

echo training hierarchy ---
python train_learnable_hierarchical.py --type temporal --simulated --data_type 1percent --epoch 200 --context_years 4
echo.

echo training hierarchy ---
python train_learnable_hierarchical.py --type temporal --simulated --data_type 1percent --epoch 200 --context_years 6
echo.

echo training hierarchy ---
python train_learnable_hierarchical.py --type temporal --simulated --data_type 1percent --epoch 200 --context_years 8
echo.


@REM echo training hierarchy ---
@REM python train_learnable_hierarchical.py --type temporal --simulated --data_type full --epoch 200 --context_years 2 --batch_size 1024
@REM echo.

@REM echo training hierarchy ---
@REM python train_learnable_hierarchical.py --type temporal --simulated --data_type full --epoch 200 --context_years 4 --batch_size 1024
@REM echo.

@REM echo training hierarchy ---
@REM python train_learnable_hierarchical.py --type temporal --simulated --data_type full --epoch 200 --context_years 6 --batch_size 1024   
@REM echo.

@REM echo training hierarchy ---
@REM python train_learnable_hierarchical.py --type temporal --simulated --data_type full --epoch 200 --context_years 8 --batch_size 1024
@REM echo.

call conda deactivate