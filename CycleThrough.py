import RFFunctions as RF
from astropy.io import fits
import glob
import os
import numpy as np
import matplotlib.pyplot as plt



def er_plot(ii,directory):
    ##This just creates a plot of the image sans ridges. 
    ##It serves as a placeholder for if the ridgefinder doesn't work for some reason.
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
    
    img = Trackimages[ii]
    c = 255/np.log(1+np.max(img))
    img2 = c *(np.log(img+1))
    plt.title(str(imnumb[ii]))
    plt.imshow(img2,cmap='magma')
    plt.pause(0.1)

def make_plot(kk,directory,sigma, lt, ut, minlen, linkthresh, logim = False):
    '''
    Create a figure containing the image of the particle track and its
    ridgeline. Two plots are produced in this figure. One is in linear space
    and the other is in log space. On both, the unlinked and linked ridges are
    plotted. The linked are plotted as a thin white line overlayed on the 
    colored points of the unlinked lines.
    
    This can be iterated through.
    
    Parameters
    ----------
    kk : int
        Image number to be looked at.
    directory : str
        Location of the image files to be analysed.
    sigma : float
        Sigma for derivative determination (somehow relate to track width).
    lt : float
        Lower threshold for the ridgefinding algorithm.
        This excludes tracks whose hessian eigenvalues fall below lthresh.
    ut : float
        Upper threshold for the ridgefinding algorithm.
        This excludes tracks whose hessian eigenvalues exceed uthresh.
    minlen : int
        Minimum track length accepted.
    linkthresh : int
        The maximum distance between endpoints allowed for linking ridges.
    logim : BOOL, optional
        Do you want the RidgeFinder to operate on the image in log space?
        The default is False.

    Returns
    -------
    None.

    '''
    ##This will create a plot of tracks and ridge for a single image. 
    ##It's meant to be used in cycling through plots but can be run alone.
    
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
    
    ################################################################
    ##Initialize matplotlib
    plt.close()
    plt.rc('font', size=10)

    

    ################################################################
    #These are the Parameters to change to work with the RF        #
    ################################################################
    I=kk #Image number to inspect
    SIGMA = sigma #sigma for derivative determination ~> Related to track width
    lthresh = lt #tracks with a response lower than this are rejected (0 accepts all)
    uthresh = ut #tracks with a response higher than this are rejected (0 accepts all)
    minlen = minlen #minimum track length accepted
    linkthresh = linkthresh #maximum distance to be linked
    # thresh = 0.0001

    img = Trackimages[I]
    if logim:
        c = 255/np.log(1+np.max(img))
        img2 = c *(np.log(img+1))
    else:
        img2 = img  
    
    c = 255/np.log(1+np.max(img))
    img3 = c *(np.log(img+1))
    
    ##Run Ridgefinder
    px, py, nx, ny, eigvals, valid = points_out = RF.find_points(img2, sigma=SIGMA, l_thresh = lthresh, u_thresh=uthresh)
    lines_before, junctions = RF.compose_lines_from_points(points_out)
    
    ##Link the Ridges
    nlines = RF.linklines(lines_before,minlen,linkthresh)
    lines=lines_before
    
    ##Set some plot properties
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle(str(imnumb[I]))
    
    ax1.imshow(img2, cmap="magma")
    ax1.set_title('Linear Scaling')
    ax2.imshow(img3, cmap="magma")
    ax2.set_title('Logarithmic Scaling')
    lim =RF.get_lines_bounding_box(lines)
    
    
    ##Run through all ridges found in image
    
    ##Create and plot the splinefit for all unlinked ridgepoints
    for i, line in enumerate(lines):
        if len(line[1]) > minlen:
                
                ax1.set_xlim(lim)
                ax1.set_ylim(lim)
                ax2.set_xlim(lim)
                ax2.set_ylim(lim)
                
                x = px[line[1], line[0]]
                y = py[line[1], line[0]]
                
                ##Get the splinefit for the image
                new_points,der_points = RF.getspline(x,y)    
                ax1.plot(new_points[1],new_points[0],'o')
                ax2.plot(new_points[1],new_points[0],'o')


    ##Create and plot the splinefit for all linked ridgepoints            
    for i, line in enumerate(nlines):
        if len(line[1]) > minlen:
            
                x = px[line[1], line[0]]
                y = py[line[1], line[0]]   
                
                ##Get the splinefit for the image
                new_points,der_points = RF.getspline(x,y)    
                ax1.plot(y,x,'-',color = 'white') 
                ax2.plot(y,x,'-',color = 'white') 

                
    plt.show()
    plt.pause(0.1) ##This command allows the cycling to actually happen

def cycleimages(directory,sigma, lt, ut, minlen, linkthresh, logim = False):
    '''
    Cycle through figures containing the image of the particle track and its
    ridgeline. Two plots are produced in each figure. One is in linear space
    and the other is in log space. On both, the unlinked and linked ridges are
    plotted. The linked are plotted as a thin white line overlayed on the 
    colored points of the unlinked lines.
    
    Parameters
    ----------
    kk : int
        Image number to be looked at.
    directory : str
        Location of the image files to be analysed.
    sigma : float
        Sigma for derivative determination (somehow relate to track width).
    lt : float
        Lower threshold for the ridgefinding algorithm.
        This excludes tracks whose hessian eigenvalues fall below lthresh.
    ut : float
        Upper threshold for the ridgefinding algorithm.
        This excludes tracks whose hessian eigenvalues exceed uthresh.
    minlen : int
        Minimum track length accepted.
    linkthresh : int
        The maximum distance between endpoints allowed for linking ridges.
    logim : BOOL, optional
        Do you want the RidgeFinder to operate on the image in log space?
        The default is False.

    Returns
    -------
    None.

    '''
    ##############################################################
    ##This gets all of the images to be analyzed and stores them##
    ##############################################################
    ##Get Raw images##
    Num_trax=len(glob.glob(directory+'*.fits'))
    
    for kk in range(Num_trax):
        try:
            make_plot(kk,directory,sigma, lt, ut, minlen, linkthresh, logim = False)
        except Exception as e:
            print(e)
            er_plot(kk,directory)
        _ = input("Press [enter] to continue.")
    
    
if __name__=='__main__':
    
    
    directory = "C:\\Users\\tilly\\Documents\\Simulations\\50torrCF4_5.204keV\\5.204 keV\\"
    SIGMA = 3.6 #sigma for derivative determination ~> Related to track width
    lthresh = 0.3 #tracks with a response lower than this are rejected (0 accepts all)
    uthresh = 0 #tracks with a response higher than this are rejected (0 accepts all)
    minlen = 11 #minimum track length accepted
    linkthresh = 11 #maximum distance to be linked
    logim = False
    
    cycleimages(directory,SIGMA,lthresh,uthresh,minlen,linkthresh,logim)

