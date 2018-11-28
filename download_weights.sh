gdrive_download () {
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
    rm -rf /tmp/cookies.txt
}
tar_download_and_extract() {
    mkdir -p $1
    cd $1
    rm -rf $2*
    gdrive_download $3 $2.tar
    tar -xvf $2.tar
    rm -rf $2.tar
    cd ..
}

tar_download_and_extract logs logs 1eH0oxCr3nLiASft7ZtyOh8vT6rYAkPC6
