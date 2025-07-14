@echo off
REM --- 设置脚本环境 ---
echo.
echo ===================================================
echo     自动化训练脚本已启动
echo ===================================================
echo.

REM --- 激活 Conda 环境 ---
REM 使用 'call' 来确保激活命令执行后，脚本能继续向下运行
echo [步骤 1/4] 正在激活 Conda 环境 'methane'...
call conda activate methane

REM --- 检查环境是否激活成功 ---
if %errorlevel% neq 0 (
    echo.
    echo 错误：激活 Conda 环境 'methane' 失败！
    echo 请确认您已经正确安装了 Conda 并且环境名称无误。
    goto :eof
)
echo 环境激活成功！
echo.

REM --- 运行第一个 Python 训练脚本 ---
echo [步骤 2/4] --- 开始训练模型 lstm ---
python .\base_model.py --model lstm
echo --- 模型 A 训练完成 ---
echo.

REM --- 运行第二个 Python 训练脚本 ---
echo [步骤 3/4] --- 开始训练模型 pyraformer ---
python .\base_model.py --model pyraformer
echo --- 模型 B 训练完成 ---
echo.

REM --- 运行第二个 Python 训练脚本 ---
echo [步骤 3/4] --- 开始训练模型 transformer ---
python .\base_model.py --model transformer
echo --- 模型 B 训练完成 ---
echo.

REM --- 运行第二个 Python 训练脚本 ---
echo [步骤 3/4] --- 开始训练模型 hybrid ---
python .\base_model.py --model hybrid
echo --- 模型 B 训练完成 ---
echo.

REM --- 结束任务 ---
echo [步骤 4/4] 所有训练任务已完成！
echo.

REM --- 停用 Conda 环境 (好习惯) ---
call conda deactivate