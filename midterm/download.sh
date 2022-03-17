sudo apt-get install unrar
mkdir -p data/
wget -P data/ https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt
wget -P data/ https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt
wget -P data/ https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt
gdown --id 1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv
unrar x NWPU-RESISC45.rar data/
rm NWPU-RESISC45.rar
