# Download the raw and processed scierc dataset and put them in the `data`
# folder.
# Usage: From main project folder, run `bash scripts/data/get_scierc.sh`

out_dir=data/cofie
mkdir $out_dir

# Download.
wget http://nlp.cs.washington.edu/COFIE/data/COFIE.tar.gz -P $out_dir
wget http://nlp.cs.washington.edu/COFIE/data/COFIE-E.tar.gz -P $out_dir

# Decompress.
tar -xf $out_dir/COFIE.tar.gz -C $out_dir
tar -xf $out_dir/COFIE-E.tar.gz -C $out_dir

# Clean up.
rm $out_dir/*.tar.gz
