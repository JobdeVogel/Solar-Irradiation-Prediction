@echo off
echo Checking renderfarm info

rem Jump to T: drive (public fileshare of BK Renderfarm)
call T:

rem Replace your_netid with your actual NetID (folder name) on the R: drive
call cd jdevogel

call conda env list