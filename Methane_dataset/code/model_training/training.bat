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
python .\base_model.py --simulated --model lstm --epoch 200
echo.

echo training pslstm ---
python .\base_model.py --simulated --model pslstm --epoch 200
echo.

echo training transformer ---
python .\base_model.py  --simulated --model transformer --epoch 200
echo.


echo training pyraformer ---
python .\base_model.py --simulated --model pyraformer --epoch 200
echo.

echo training itransformer ---
python .\base_model.py --simulated --model itransformer --epoch 200
echo.

echo training tsmixer ---
python .\base_model.py --simulated --model tsmixer --epoch 200
echo.

echo training timemixer ---
python .\base_model.py --simulated --model timemixer --epoch 200
echo.

echo training duet ---
python .\base_model.py --simulated --model duet --epoch 200
echo.

echo training patchtst ---
python .\base_model.py --simulated --model patchtst --epoch 200
echo.

echo training hierarchy ---
python train_learnable_hierarchical.py --simulated --epoch 200 --context_years 2
echo.

echo training hierarchy ---
python train_learnable_hierarchical.py --simulated --epoch 200 --context_years 4
echo.

echo training hierarchy ---
python train_learnable_hierarchical.py --simulated --epoch 200 --context_years 6
echo.

echo training hierarchy ---
python train_learnable_hierarchical.py --simulated --epoch 200 --context_years 8
echo.

echo training lstm ---
python .\base_model.py --model lstm --epoch 1000
echo.

echo training pslstm ---
python .\base_model.py --model pslstm --epoch 1000
echo.

echo training transformer ---
python .\base_model.py --model transformer --epoch 1000
echo.

echo training pyraformer ---
python .\base_model.py --model pyraformer --epoch 1000
echo.

echo training itransformer ---
python .\base_model.py --model itransformer --epoch 1000
echo.

echo training tsmixer ---
python .\base_model.py --model tsmixer --epoch 1000
echo.

echo training timemixer ---
python .\base_model.py --model timemixer --epoch 1000
echo.

echo training duet ---
python .\base_model.py --model duet --epoch 1000
echo.

echo training patchtst ---
python .\base_model.py --model patchtst --epoch 1000
echo.


call conda deactivate