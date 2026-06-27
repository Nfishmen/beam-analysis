@echo off
title Beam Analysis System
cd /d D:eam_analysis
echo ========================================================
echo   Beam Structure Analysis System
echo   Starting server, opening browser...
echo ========================================================
start http://127.0.0.1:5000
python app.py
pause
