set -e

# create venv
python3 -m venv .venv

# activate venv
source .venv/bin/activate

# upgrade pip
pip install --upgrade pip

# install deps
pip install -r requirements.txt

echo ""
echo "✅ environment created"
echo "to use it now run:"
echo "source .venv/bin/activate"