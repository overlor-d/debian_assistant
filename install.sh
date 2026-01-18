sudo apt install -y python3 python3-venv python3-pip \
                    portaudio19-dev sox ffmpeg

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

clear

echo "[Installation complete]"