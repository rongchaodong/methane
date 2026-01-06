echo activating conda 'methane'...
call conda activate methane

if %errorlevel% neq 0 (
    echo.
    echo activating conda methane failed!
    goto :eof
)
echo activating done.
echo.

echo training lstm embedding ---
python .\train_learnable_hierarchical.py --model lstm_embedding --epoch 200 --context_years 2
echo.

echo training lstm embedding ---
python .\train_learnable_hierarchical.py --model lstm_embedding --epoch 200 --context_years 4
echo.

echo training lstm embedding ---
python .\train_learnable_hierarchical.py --model lstm_embedding --epoch 200 --context_years 6
echo.

echo training lstm embedding ---
python .\train_learnable_hierarchical.py --model lstm_embedding --epoch 200 --context_years 8
echo.


echo training hierarchy ---
python train_learnable_hierarchical.py --epoch 200 --context_years 2
echo.

echo training hierarchy ---
python train_learnable_hierarchical.py --epoch 200 --context_years 4
echo.

echo training hierarchy ---
python train_learnable_hierarchical.py --epoch 200 --context_years 6
echo.

echo training hierarchy ---
python train_learnable_hierarchical.py --epoch 200 --context_years 8
echo.

echo training lstm embedding ---
python .\train_learnable_hierarchical_fluxnet.py --model lstm_embedding --epoch 1000
echo.

echo training hierarchy ---
python train_learnable_hierarchical_fluxnet.py --epoch 1000
echo.

call conda deactivate