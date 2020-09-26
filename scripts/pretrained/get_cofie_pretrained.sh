# Download pretrained COFIE binary and ternary models from AWS bucket and put the result in
# `./pretrained/`.

# Usage: python scripts/pretrained/get_cofie_pretrained.sh.

# binary - relations.
wget --directory-prefix=./pretrained \
    wget https://ai2-s2-cofie.s3-us-west-2.amazonaws.com/models/binary-model.tar.gz

# ternary - relations.
wget --directory-prefix=./pretrained \
    wget https://ai2-s2-cofie.s3-us-west-2.amazonaws.com/models/ternary-model.tar.gz

