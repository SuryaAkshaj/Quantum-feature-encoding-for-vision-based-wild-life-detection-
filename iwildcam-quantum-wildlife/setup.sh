#!/usr/bin/env bash
set -e

echo "Creating venv..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Creating folders..."
mkdir -p precomputed models web/uploads web/crops detectors/yolov8

echo "Done. If you want to precompute features run:"
echo "  python src/precompute_features.py"

# Optional steps (run manually in separate terminals):
# 2. Create sample data:
#    python create_sample_data.py
# 3. Train quantum model (2-3 minutes):
#    python -m src.train_vqc
# 4. Start Flask API (Terminal 1):
#    python -m web.app_flask
# 5. Start Streamlit UI (Terminal 2 - new window):
#    streamlit run web/app_streamlit.py
