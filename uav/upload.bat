@echo off
chcp 65001 >nul
setlocal
set "UAV=%~dp0"
cd /d "%UAV%.."
if not exist school-project (
    echo Cloning school-project...
    git clone https://github.com/ZZZamchi/school-project.git
)
echo Copying uav to school-project\uav...
if not exist school-project\uav mkdir school-project\uav
xcopy /E /I /Y "%UAV%*" school-project\uav\ >nul
cd school-project
git add uav
git status
git commit -m "uav: MM-UAVBench 4-model results"
git push -u origin main
pause
