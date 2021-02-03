# Download the raw and processed scierc dataset and put them in the `data`
# folder.
# Usage: From main project folder, run `bash scripts/data/get_scierc.sh`

out_dir=data/
mkdir $out_dir
cd $out_dir

# Download.
wget https://ai2-s2-cofie.s3-us-west-2.amazonaws.com/data.zip

# Decompress.
unzip  data.zip 

# Clean up.
rm data.zip
