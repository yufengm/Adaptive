
# coding: utf-8

# # Karpathy Split for MS-COCO Dataset
import json
from random import shuffle, seed

seed( 123 ) # Make it reproducible

num_val = 5000
num_test = 5000

val = json.load( open('annotations/captions_val2014.json', 'r') )
train = json.load( open('annotations/captions_train2014.json', 'r') )

# Merge together
imgs = val['images'] + train['images']
annots = val['annotations'] + train['annotations']

shuffle( imgs )

# Split into val, test, train
dataset = {}
dataset[ 'val' ] = imgs[ :num_val ]
dataset[ 'test' ] = imgs[ num_val: num_val + num_test ]
dataset[ 'train' ] = imgs[ num_val + num_test: ]

# Group by image ids
itoa = {}
for a in annots:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)


json_data = {}
info = train['info']
licenses = train['licenses']

split = [ 'val', 'test', 'train' ]

for subset in split:
    
    json_data[ subset ] = { 'type':'caption', 'info':info, 'licenses': licenses,
                           'images':[], 'annotations':[] }
    
    for img in dataset[ subset ]:
        
        img_id = img['id']
        anns = itoa[ img_id ]
        
        json_data[ subset ]['images'].append( img )
        json_data[ subset ]['annotations'].extend( anns )
        
    json.dump( json_data[ subset ], open( 'annotations/karpathy_split_' + subset + '.json', 'w' ) )

