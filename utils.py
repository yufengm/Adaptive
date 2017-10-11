import json
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pickle
from build_vocab import Vocabulary
from torch.autograd import Variable 
from torchvision import transforms, datasets
from coco.pycocotools.coco import COCO
from coco.pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt

# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable( x, volatile=volatile )

# Show multiple images and caption words
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
            
    Adapted from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """
    
    assert(( titles is None ) or (len( images ) == len( titles )))
    
    n_images = len( images )
    if titles is None: 
        titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
        
    fig = plt.figure( figsize=( 15, 15 ) )
    for n, (image, title) in enumerate( zip(images, titles) ):
        
        a = fig.add_subplot( np.ceil( n_images/ float( cols ) ), cols, n+1 )
        if image.ndim == 2:
            plt.gray()
            
        plt.imshow( image )
        a.axis('off')
        a.set_title( title, fontsize=200 )
        
    fig.set_size_inches( np.array( fig.get_size_inches() ) * n_images )
    
    plt.tight_layout( pad=0.4, w_pad=0.5, h_pad=1.0 )
    plt.show()

# MS COCO evaluation data loader
class CocoEvalLoader( datasets.ImageFolder ):

    def __init__( self, root, ann_path, transform=None, target_transform=None, 
                 loader=datasets.folder.default_loader ):
        '''
        Customized COCO loader to get Image ids and Image Filenames
        root: path for images
        ann_path: path for the annotation file (e.g., caption_val2014.json)
        '''
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = json.load( open( ann_path, 'r' ) )['images']


    def __getitem__(self, index):

        filename = self.imgs[ index ]['file_name']
        img_id = self.imgs[ index ]['id']
        
        # Filename for the image
        if 'val' in filename.lower():
            path = os.path.join( self.root, 'val2014' , filename )
        else:
            path = os.path.join( self.root, 'train2014', filename )

        img = self.loader( path )
        if self.transform is not None:
            img = self.transform( img )

        return img, img_id, filename

# MSCOCO Evaluation function
def coco_eval( model, args, epoch ):
    
    '''
    model: trained model to be evaluated
    args: pre-set parameters
    epoch: epoch #, for disp purpose
    '''
    
    model.eval()
    
    # Validation images are required to be resized to 224x224 already
    transform = transforms.Compose([ 
        transforms.Scale( (args.crop_size, args.crop_size) ),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load the vocabulary
    with open( args.vocab_path, 'rb' ) as f:
         vocab = pickle.load( f )
    
    # Wrapper the COCO VAL dataset
    eval_data_loader = torch.utils.data.DataLoader( 
        CocoEvalLoader( args.image_dir, args.caption_val_path, transform ), 
        batch_size = args.eval_size, 
        shuffle = False, num_workers = args.num_workers,
        drop_last = False )  
    
    # Generated captions to be compared with GT
    results = []
    print '---------------------Start evaluation on MS-COCO dataset-----------------------'
    for i, (images, image_ids, _ ) in enumerate( eval_data_loader ):
        
        images = to_var( images )
        generated_captions, _, _ = model.sampler( images )
        
        if torch.cuda.is_available():
            captions = generated_captions.cpu().data.numpy()
        else:
            captions = generated_captions.data.numpy()
        
        # Build caption based on Vocabulary and the '<end>' token
        for image_idx in range( captions.shape[0] ):
            
            sampled_ids = captions[ image_idx ]
            sampled_caption = []
            
            for word_id in sampled_ids:
                
                word = vocab.idx2word[ word_id ]
                if word == '<end>':
                    break
                else:
                    sampled_caption.append( word )
            
            sentence = ' '.join( sampled_caption )
            
            temp = { 'image_id': int( image_ids[ image_idx ] ), 'caption': sentence }
            results.append( temp )
        
        # Disp evaluation process
        if (i+1) % 10 == 0:
            print '[%d/%d]'%( (i+1),len( eval_data_loader ) ) 
            
            
    print '------------------------Caption Generated-------------------------------------'
            
    # Evaluate the results based on the COCO API
    resFile = 'results/mixed-' + str( epoch ) + '.json'
    json.dump( results, open( resFile , 'w' ) )
    
    annFile = args.caption_val_path
    coco = COCO( annFile )
    cocoRes = coco.loadRes( resFile )
    
    cocoEval = COCOEvalCap( coco, cocoRes )
    cocoEval.params['image_id'] = cocoRes.getImgIds() 
    cocoEval.evaluate()
    
    # Get CIDEr score for validation evaluation
    cider = 0.
    print '-----------Evaluation performance on MS-COCO validation dataset for Epoch %d----------'%( epoch )
    for metric, score in cocoEval.eval.items():
        
        print '%s: %.4f'%( metric, score )
        if metric == 'CIDEr':
            cider = score
            
    return cider
