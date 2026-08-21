@echo off
chcp 65001 > nul
echo ========================================================
echo  SO-101 6-DoF Time-Series RNN Closed-Loop Evaluator
echo ========================================================
cd /d "%~dp0"
C:\Users\USER\miniconda3\envs\mujoco_env\python.exe step4_closed_loop_eval.py
pause
