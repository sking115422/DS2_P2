
# sudo chmod +x ./start_venv_lm.sh
pip install virtualenv
rm -r venv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "SETUP COMPLETE! Please restart your IDE before proceeding."
echo ""