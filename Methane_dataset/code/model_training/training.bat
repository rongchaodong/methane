@echo off
setlocal

echo activating conda 'methane'...
call conda activate methane

if %errorlevel% neq 0 (
    echo.
    echo activating conda methane failed!
    goto :eof
)
echo activating done.
echo.

set RUNS=3

for %%m in (lstm pslstm transformer pyraformer itransformer tsmixer timemixer duet patchtst) do (
    echo ------------------------------------------------
    echo training %%m [Simulated] ...
    echo ------------------------------------------------
    
    for /L %%i in (1,1,%RUNS%) do (
        echo Run %%i of %RUNS%...
        python .\base_model.py --simulated --model %%m --epoch 200
    )
    echo.
)

for %%y in (2 4 6 8) do (
    echo ------------------------------------------------
    echo training hierarchy context_years %%y ...
    echo ------------------------------------------------
    
    for /L %%i in (1,1,%RUNS%) do (
        echo Run %%i of %RUNS%...
        python train_learnable_hierarchical.py --simulated --epoch 200 --context_years %%y
    )
    echo.
)

for %%m in (lstm pslstm transformer pyraformer itransformer tsmixer timemixer duet patchtst) do (
    echo ------------------------------------------------
    echo training %%m [Real Data] ...
    echo ------------------------------------------------
    
    for /L %%i in (1,1,%RUNS%) do (
        echo Run %%i of %RUNS%...
        python .\base_model.py --model %%m --epoch 1000
    )
    echo.
)

echo ------------------------------------------------
echo training hierarchy [Fluxnet] ...
echo ------------------------------------------------
for /L %%i in (1,1,%RUNS%) do (
    echo Run %%i of %RUNS%...
    python train_learnable_hierarchical_fluxnet.py --epoch 1000
)

echo.
echo All Done!
call conda deactivate
endlocal
pause