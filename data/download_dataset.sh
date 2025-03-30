mkdir -p train val
wget -O NCT-CRC-HE-100K.zip https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip
unzip -qq NCT-CRC-HE-100K.zip -d train
rm NCT-CRC-HE-100K.zip

wget -O CRC-VAL-HE-7K.zip https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip
unzip -qq CRC-VAL-HE-7K.zip -d val
rm CRC-VAL-HE-7K.zip

echo "Dataset download and extraction complete."