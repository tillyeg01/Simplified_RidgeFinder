######################
######################
## Isolating Tracks ##
######################
######################
import numpy as np
from scipy import ndimage as nd



def BinaryMap(image,threshold):
    """
    Takes an input image and turns it into a binary image for values
    above a set threshold
    
    Parameters
    ----------
    image: ARRAY
         The input image that will be turned into a binary
    
    threshold : NUMBER
        Standard deviation threshold for excluding noise. 
        
    Returns
    -------
    A binary image of the same size as the input image
    """
    (x,y)=image.shape
    outim = np.zeros((x,y))
    outim = image>threshold
    
    return outim

from math import nan

def Single_Track_Isolation(img, thresh=0, size=128):
    '''
    This function isolates tracks from a given
    image and returns a new "postage stamp" 
    image. 

    Parameters
    ----------
    img : Numpy Array
        Original large image with a track in it.
    thresh : Float
        Threshold above which it will look for
        the tracks. Uncalled value = 0
    size : Int
        Size of the postage stamp to be returned.
        Uncalled value = 128
    Returns
    -------
    NewIm : Numpy Array
        A newly cropped size x size image

    '''
        
    XsYs = BinaryMap(img,thresh) #Creates a binary map of the image above a given threshold
    labelled_array, numOfLabels = nd.label(XsYs, structure = np.ones((3,3),int))
    ## numOfLabels is self explanatory.
    ## labelled_array is similar to XsYs except instead of everything being 0s and 1s,
    ## each "clump" of pixels is set to its label. So, if you have 2 "objects" in your
    ## image, one will be a group of pixels of value 1 and the other will be a group of 
    ## pixels of value 2. To visualize, uncomment the following code:
    
    #plt.imshow(labelled_array)
    
    ## Here, we actually look for the size of the "objects" found
    values, counts = np.unique(labelled_array, return_counts=True)
    values = values[1:]
    counts = counts[1:]
    
    larg_obj_ind = np.where(counts == max(counts))[0] ## Find the largest object found
    filtarray = np.where(labelled_array==values[larg_obj_ind],1,0) ## Masks for the largest object in the image
    filtarray1 = np.asarray(np.nonzero(filtarray)) ## Gets the actual indices of the object we've isolated
    
    (maxx,maxy) = filtarray.shape
    if maxx in filtarray1 or maxy in filtarray1 or 0 in filtarray1:
        raise Exception("Track on Edge of Image.")
    else:
        filtarray2 = np.pad(filtarray, pad_width=int(size/2),mode='constant',constant_values=0) ## Pads the edges with 0
        img = np.pad(img, pad_width=int(size/2),mode='constant',constant_values=0) ## Pads the edges with 0
        filtarray1 = np.asarray(np.nonzero(filtarray2)) ## Gets the actual indices of the object we've isolated
        ## Determine the midpoint of your object
        
        maxxind = np.where(filtarray1[0]==max(filtarray1[0]))[0]
        minxind = np.where(filtarray1[0]==min(filtarray1[0]))[0]
        maxyind = np.where(filtarray1[1]==max(filtarray1[1]))[0]
        minyind = np.where(filtarray1[1]==min(filtarray1[1]))[0]
        
        maxx = filtarray1[0][maxxind[0]]
        minx = filtarray1[0][minxind[0]]
        maxy = filtarray1[1][maxyind[0]]
        miny = filtarray1[1][minyind[0]]
        
        
        midx = int((maxx-minx)/2)+minx
        midy = int((maxy-miny)/2)+miny
        
        
        NewIm = img[int(midx-(size)/2):int(midx+(size)/2),int(midy-(size)/2):int(midy+(size)/2)]
        
        ### TO see part of this code in action, follow this link https://colab.research.google.com/drive/1dQ-ZBuNqACIlVM8t5MT8CE6-u8YEvCDP?usp=sharing
        ### You shouldn't have to run it at all, but if you do, DON'T run it past the def GetGalaxyInformation block. It takes a super long
        ### time to run. Note: this works just like a jupyter notebook for the most part. You'll have to scroll down to get to the stuff
        ### that impliments this code. 
        
        return NewIm
    
#%%

# import glob
# import io
# from astropy.io import fits

# directoryDD = 'C:\\Users\\tilly\\Documents\\Simulations\\Tims Tracks\\Tiffs\\data\\data\\DD\\'
# directoryDDM = 'C:\\Users\\tilly\\Documents\\Simulations\\Tims Tracks\\Tiffs\\data\\data\\DD_migdal\\'


# nameDD = glob.glob(directoryDD+'*.tiff')
# nameDDM = glob.glob(directoryDDM+'*.tiff')

# for ii,name in enumerate(nameDD):
#     img = io.imread(name)
#     img = Single_Track_Isolation(img)