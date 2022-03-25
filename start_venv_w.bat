
pip install virtualenv
rm -r venv
virtualenv venv
call venv/Scripts/activate
pip install -r requirements.txt

echo.
echo SETUP COMPLETE! Please restart your IDE before proceeding.
echo.