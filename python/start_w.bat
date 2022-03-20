
rm -r venv
python -m venv venv
call venv/Scripts/activate
pip install -r requirements.txt
ipython kernel install --user --name=venv

echo.
echo SETUP COMPLETE! Please restart your IDE before proceeding.
echo.