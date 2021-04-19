# Download pretrained MECHANIC coarse and granular models from AWS bucket and put the result in
# `./pretrained/`.

# Usage: sh scripts/get_mechanic_pretrained.sh

# coarse and granular relation models
for name in mechanic-coarse mechanic-granular
do
    if [ ! -f pretrained/mechanic-${name}.tar.gz ]
    then
        wget --directory-prefix=./pretrained \
            "https://s3-us-west-2.amazonaws.com/ai2-s2-mechanic/models/${name}.tar.gz"
    fi
done
