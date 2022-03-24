
sudo chmod +x start_lm.sh  
rm -r venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
ipython kernel install --user --name=venv

echo ""
echo "SETUP COMPLETE! Please restart your IDE before proceeding."
echo ""