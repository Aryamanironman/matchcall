@echo off
rem Start the football-commentator web app with the correct environment.
rem NOTE: PYTHONPATH must be cleared or another environment's packages shadow this .venv.
cd /d %~dp0
set PYTHONPATH=
set FISH_VOICE_ID=40f53434437e46a5bcbb5abb39f5079c
.venv\Scripts\python.exe -m uvicorn app.webapp:app --host 127.0.0.1 --port 8000
