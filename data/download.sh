wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P ./
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P ./

unzip ./captions_train-val2014.zip -d ./
rm ./captions_train-val2014.zip
unzip ./train2014.zip -d ./
rm ./train2014.zip 
unzip ./val2014.zip -d ./
rm ./val2014.zip 
