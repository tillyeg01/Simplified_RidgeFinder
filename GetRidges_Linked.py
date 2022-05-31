import RFFunctions as RF
from astropy.io import fits
import glob
import pandas as pd
import os
import numpy as np
import re



def getridges(directory,SIGMA,lthresh,uthresh,minlen,logim=False,linked_lines=True,linkthresh=0):
    '''
    This finds and extracts the ridge lines from all images of tracks in a
    given directory.
    
    Parameters
    ----------
    directory : str
        Location of the image files to be analysed.
    SIGMA : float
        Sigma for derivative determination (somehow relate to track width).
    lthresh : float
        Lower threshold for the ridgefinding algorithm.
        This excludes tracks whose hessian eigenvalues fall below lthresh.
    uthresh : float
        Upper threshold for the ridgefinding algorithm.
        This excludes tracks whose hessian eigenvalues exceed uthresh.
    minlen : int
        Minimum track length accepted.
    logim : BOOL, optional
        Do you want the RidgeFinder to operate on the image in log space?
        The default is False.
    linked_lines : BOOL, optional
        Do you also want the linked ridges? If yes, any ridges with endpoints
        within linkthresh will be stitched together. This will return both the
        unlinked and linked dataframes. Otherwise, it will return the unlinked
        dataframe and an empty list. The default is True.
    linkthresh : int, optional
        The maximum distance between endpoints allowed for linking ridges.
        The default is 0.

    Returns
    -------
    Ridges : Dataframe
        This contains all of the unlinked ridgelines found in the directory.
        The first line of each column contains the image number and the track's
        start coordinates for ID purposes. Each row contains the x y coordinates
        of the ridge. 
    Linked_Ridges: Dataframe
        If linked_lines = True
        This contains all of the unlinked ridgelines found in the directory.
        The first line of each column contains the image number and the track's
        start coordinates for ID purposes. Each row contains the x y coordinates
        of the ridge. 
        If linked_lines = False
        This will just return an empty list.

    '''
    ##Get Raw images##
    name = list()
    imnumb = list()
    for np_name in glob.glob(directory+'*.fits'):
            name.append(np_name)
            imnumb.append(os.path.basename(np_name))
    Num_trax=len(name)
    (Nx,Ny) = fits.getdata(name[0]).shape
    Trackimages = np.zeros((Num_trax,Nx,Ny))
    for i in range(Num_trax):
        Trackimages[i]=np.array(fits.getdata(name[i]))   
    
    ##Initialize the dataframes
    Ridges = pd.DataFrame()
    Linked_Ridges = pd.DataFrame()
    
    for j,Im in enumerate(Trackimages):
        
        ## Applies Logarithmic scaling if the user so desires
        if logim:
            img = Im
            c = 255/np.log(1+np.max(img))
            I = c *(np.log(img+1))
        else:
            I = Im  
        
        
        ##Run the Ridgefinder##
        px, py, nx, ny, eigvals, valid = points_out = RF.find_points(I, sigma=SIGMA, l_thresh = lthresh, u_thresh=uthresh)
        lines, junctions = RF.compose_lines_from_points(points_out)
        link_lines = RF.linklines(lines,minlen,linkthresh) ## Link ridges close to one another

        ##Cycle through the tracks to determine the statistics of each one##
        for i, line in enumerate(lines):
            if len(line[1]) > minlen:
                x = px[line[1], line[0]]
                y = py[line[1], line[0]]
                
                
                ##Get each track's identifying information
                Track_ID = str(j)+":_" + str(round(x[0],1))+"_"+str(round(y[0],1)) #Unique ID for track
                Im_ID = imnumb[j] #Image ID from filename (not the same as Im_num
                tru_Im_num = int(re.findall('\d+',Im_ID)[0]) #Extracts the number from the filename
                
                           
                new_points,der_points = RF.getspline(x,y) #Get spline of ridgline for initial direction calculation 
                ridgecoord=[str(m)+' '+str(n) for m,n in zip(new_points[0],new_points[1])]
                Ridgescoord = pd.DataFrame({'Image: '+str(tru_Im_num)+'; Track: '+str(Track_ID):ridgecoord})
                Ridges = pd.concat([Ridges,Ridgescoord],axis=1)
                
        
            
        
        
        for k, line1 in enumerate(link_lines):
            try:
                if len(line1[1]) > minlen:
                        x1 = px[line1[1], line1[0]]
                        y1 = py[line1[1], line1[0]]
                        Track_ID = str(j)+":_" + str(round(x1[0],1))+"_"+str(round(y1[0],1)) #Unique ID for track
                        # Im_num = str(j) #Unique ID for image
                        Im_ID = imnumb[j] #Image ID from filename (not the same as Im_num
                        tru_Im_num = int(re.findall('\d+',Im_ID)[0]) #Extracts the number from the filename       
                        
                        ##Get the splinefit for the image
                        new_points,der_points = RF.getspline(x1,y1)   
                        # Track_length =np.hypot(np.diff(x1), np.diff(y1)).sum()*176/10**3 #Determine tracklength
                        
                        Lridgecoord=[str(m)+' '+str(n) for m,n in zip(new_points[0],new_points[1])]
                        LRidgescoord = pd.DataFrame({'Image: '+str(tru_Im_num)+'; Track: '+str(Track_ID):Lridgecoord})
                        Linked_Ridges = pd.concat([Linked_Ridges,LRidgescoord],axis=1)
            except Exception as e:                
                print("For Im "+str(j)+" linked track "+str(k))
                print(e)
        
    if linked_lines == True:
          return Ridges, Linked_Ridges
    elif linked_lines == False:
          a = []
          return Ridges , a
      
if __name__=='__main__':
    
    
    directory = "C:\\Users\\tilly\\Documents\\Simulations\\50torrCF4_5.204keV\\5.204 keV\\"
    SIGMA = 3.6 #sigma for derivative determination ~> Related to track width
    lthresh = 0.3 #tracks with a response lower than this are rejected (0 accepts all)
    uthresh = 0 #tracks with a response higher than this are rejected (0 accepts all)
    minlen = 11 #minimum track length accepted
    linkthresh = 11 #maximum distance to be linked
    logim = False
    
    Ridges, Linked_Ridges = getridges(directory,SIGMA,lthresh,uthresh,minlen,logim=logim,linked_lines=True, linkthresh=linkthresh)
