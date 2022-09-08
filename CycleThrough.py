import RFFunctions as RF
from astropy.io import fits
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io



def er_plot(ii,directory):
    ##This just creates a plot of the image sans ridges. 
    ##It serves as a placeholder for if the ridgefinder doesn't work for some reason.
    ##Get Raw images##
    plt.close()
    name1= glob.glob(directory+'*.fits')
    name2= glob.glob(directory+'*.fit')
    name = np.concatenate((name1,name2)).tolist()
    imnumb = [os.path.basename(i) for i in name] 
    
    Num_trax=len(name)
    (Nx,Ny) = fits.getdata(name[0]).shape
    Trackimages = np.zeros((Num_trax,Nx,Ny))
    
    for i in range(Num_trax):
        Trackimages[i]=np.array(fits.getdata(name[i]))
    
    img = Trackimages[ii]
    
    ## Uncomment this to see the pic in log space ##
    # c = 255/np.log(1+np.max(img))
    # img2 = c *(np.log(img+1))
    plt.title(str(imnumb[ii]))
    plt.imshow(img,cmap='jet')
    plt.show()
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
      
    name1= glob.glob(directory+'*.fits')
    name2= glob.glob(directory+'*.fit')
    name = np.concatenate((name1,name2)).tolist()
    imnumb = [os.path.basename(i) for i in name] 
    
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
    
    # img2 = img2[np.isfinite(img2)]
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
                try:
                    new_points,der_points = RF.getspline(x,y,ss=2)    
                    ax1.plot(new_points[1],new_points[0],'.')
                    ax2.plot(new_points[1],new_points[0],'.')
                except Exception as e: print(e)

    ##Create and plot the splinefit for all linked ridgepoints            
    for i, line in enumerate(nlines):
        if len(line[1]) > minlen:
            
                x = px[line[1], line[0]]
                y = py[line[1], line[0]]   
                
                ##Get the splinefit for the image
                try:
                    new_points1,der_points1 = RF.getspline(x,y,ss=2)    
                    ax1.plot(new_points1[1],new_points1[0],'-',color='white')
                    ax2.plot(new_points1[1],new_points1[0],'-',color='white')
                except Exception as e: print(e)

                
    plt.show()
    plt.pause(0.1) ##This command allows the cycling to actually happen

def make_plot_tiff(kk,directory,sigma, lt, ut, minlen, linkthresh, logim = False):
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
      
    name1= glob.glob(directory+'*.tiff')
    name2= glob.glob(directory+'*.tif')
    name = np.concatenate((name1,name2)).tolist()

    img = io.imread(name[kk])
    imnumb = os.path.basename(name[kk]) ## Get image name
    img = img/20 ## Scale the image
    
    
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

    ## Logarithmically scales the image 
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
    fig.suptitle(str(imnumb))
    
    ax1.imshow(img, cmap="magma")
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
                try:
                    new_points,der_points = RF.getspline(x,y,ss=1)    
                    ax1.plot(new_points[1],new_points[0],'.')
                    ax2.plot(new_points[1],new_points[0],'.')
                except Exception as e: print(e)

    ##Create and plot the splinefit for all linked ridgepoints            
    for i, line in enumerate(nlines):
        if len(line[1]) > minlen:
            
                x = px[line[1], line[0]]
                y = py[line[1], line[0]]   
                
                ##Get the splinefit for the image
                try:
                    new_points1,der_points1 = RF.getspline(x,y,ss=2)    
                    ax1.plot(new_points1[1],new_points1[0],'-',color='white')
                    ax2.plot(new_points1[1],new_points1[0],'-',color='white')
                except Exception as e: print(e)

                
    plt.show()
    plt.pause(0.1) ##This command allows the cycling to actually happen    


def make_plot_2(kk,directory,sigma, lt, ut, minlen, linkthresh, logim = False):
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
      
    # name1= glob.glob(directory+'*light.fits')
    # name2= glob.glob(directory+'*light.fit')
    name1= glob.glob(directory+'*.fits')
    name2= glob.glob(directory+'*.fit')
    name = np.concatenate((name1,name2)).tolist()
    imnumb = [os.path.basename(i) for i in name] 
    
    Num_trax=len(name)
    (Nx,Ny) = fits.getdata(name[0]).shape
    Trackimages = np.zeros((Num_trax,Nx,Ny))
    
    for i in range(Num_trax):
        Trackimages[i]=np.array(fits.getdata(name[i]))
    
    ################################################################
    ##Initialize matplotlib
    plt.close()
    plt.rc('font', size=15)

    

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
    
    # c = 255/np.log(1+np.max(img))
    # img3 = c *(np.log(img+1))
    
    ##Run Ridgefinder
    px, py, nx, ny, eigvals, valid = points_out = RF.find_points(img2, sigma=SIGMA, l_thresh = lthresh, u_thresh=uthresh)
    lines_before, junctions = RF.compose_lines_from_points(points_out)
    
    ##Link the Ridges
    nlines = RF.linklines(lines_before,minlen,linkthresh)
    lines=lines_before
    
    ##Set some plot properties
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle(str(imnumb[I]))
    
    ax1.imshow(Trackimages[I], cmap="jet")
    ax1.set_title('Ridge and Initial Direction')
    # ax2.imshow(img3, cmap="magma")
    ax2.set_title('Bragg Curve')
    lim =RF.get_lines_bounding_box(lines)
    
    
    ##Run through all ridges found in image
    
    ##Create and plot the splinefit for all unlinked ridgepoints
    for i, line in enumerate(lines):
        if len(line[1]) > minlen:
                
                ax1.set_xlim(lim)
                ax1.set_ylim(lim)
                # ax2.set_xlim(lim)
                # ax2.set_ylim(lim)
                
                x = px[line[1], line[0]]
                y = py[line[1], line[0]]
                
                ##Get the splinefit for the image
                new_points,der_points = RF.getspline(x,y,ss=2)    
                # ax1.plot(new_points[1],new_points[0],'.')
                # ax2.plot(new_points[1],new_points[0],'.')


    ##Create and plot the splinefit for all linked ridgepoints            
    for i, line in enumerate(nlines):
        if len(line[1]) > minlen:
            
                x = px[line[1], line[0]]
                y = py[line[1], line[0]]   
                
                
                
                
                ##Get the splinefit for the image
                new_points1,der_points1 = RF.getspline(x,y,ss=2)
                Bragg = RF.simplebragg(new_points1[0],new_points1[1],img)
                peak = np.where(Bragg == max(Bragg))[0][0]
                if peak < len(Bragg)/2:
                #Init Dir is at the end of the line list, so we reverse it
                    new_points1[1]=new_points1[1][::-1]
                    new_points1[0]=new_points1[0][::-1]
                else:
                    new_points1[1]=new_points1[1]
                    new_points1[0]=new_points1[0]
                
                ax1.plot(new_points1[1],new_points1[0],'-',color='white')
                # ax2.plot(new_points1[1],new_points1[0],'-',color='white')
                # Bragg = RF.simplebragg(new_points1[0],new_points1[1],img)
                Bragg = RF.InitialBragg(new_points1[0],new_points1[1],img)
                indir,(x0,y0) = RF.initdir_simdat(new_points1[1],new_points1[0],pix=1)
                slope = abs(np.tan(indir))
                leng = 10
                if indir >= -np.pi and indir < -np.pi/2:
                    dy = -leng/(np.sqrt(1+slope**2))
                    dx = slope*dy
                elif indir >= -np.pi/2 and indir < 0:
                    dy = leng/(np.sqrt(1+slope**2))
                    dx = -slope*dy
                elif indir >= 0 and indir < np.pi/2:
                    dy = leng/(np.sqrt(1+slope**2))
                    dx = slope*dy
                else:
                    dy = -leng/(np.sqrt(1+slope**2))
                    dx = -slope*dy
                colors = ["red","orange","green","blue","purple"]
                ax1.arrow(x0,y0,dx,dy,head_width=3, head_length=3,label=str(i),ec=colors[i],fc=colors[i])
                ax1.annotate(str(round(indir*180/np.pi,2)),(x0,y0),color = "white")
                ax2.plot(Bragg,label=str(i),color=colors[i])
                ax2.set_xlabel("Length along Track")
                ax2.set_ylabel("Pixel Intensity")
                # indir = indir*180/np.pi
                ax1.legend()
                ax2.legend()
                print(indir)

                
    plt.show()
    plt.pause(0.1) ##This command allows the cycling to actually happen

def make_plot_2a(kk,track,directory,sigma, lt, ut, minlen, linkthresh, logim = False, leng=15, pt = 10, stx= 140, sty= 70):
    '''
    Create a figure containing the image with particle track, its
    ridgeline and initial direction and a plot with the Bragg curve 
    along the ridgeline.
    
    This can be iterated through.
    
    Parameters
    ----------
    kk : int
        Image number to be looked at.
    track : int
        Track to be looked at. 
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
    leng : int
        Length of the initial direction vector.
    pt : int
        Location for the initial direction vector to be calculated from and 
        drawn on.
    stx sty : int
        X and Y coordinate for the 1mm bar to be drawn (note, you'll have to 
        change the length of it in the fxn itself. It's automatically set to
        6.6 pix)
    
    Returns
    -------
    None.

    '''
    ##This will create a plot of tracks and ridge for a single image. 
    ##It's meant to be used in cycling through plots but can be run alone.
    
    ##Get Raw images##
      
    # name1= glob.glob(directory+'*light.fits')
    # name2= glob.glob(directory+'*light.fit')
    name1= glob.glob(directory+'*.fits')
    name2= glob.glob(directory+'*.fit')
    name = np.concatenate((name1,name2)).tolist()
    imnumb = os.path.basename(name[kk]) 
    
    # Num_trax=len(name)
    # (Nx,Ny) = fits.getdata(name[0]).shape
    # Trackimages = np.zeros((Num_trax,Nx,Ny))
    I=kk
    img = np.array(fits.getdata(name[I]))
    # for ii in range(Num_trax):
    #     Trackimages[ii]=np.array(fits.getdata(name[ii]))
    
    ################################################################
    ##Initialize matplotlib
    plt.close()
    plt.rc('font', size=15)

    

    ################################################################
    #These are the Parameters to change to work with the RF        #
    ################################################################
     #Image number to inspect
    SIGMA = sigma #sigma for derivative determination ~> Related to track width
    lthresh = lt #tracks with a response lower than this are rejected (0 accepts all)
    uthresh = ut #tracks with a response higher than this are rejected (0 accepts all)
    minlen = minlen #minimum track length accepted
    linkthresh = linkthresh #maximum distance to be linked
    # thresh = 0.0001

    
    if logim:
        c = 255/np.log(1+np.max(img))
        img2 = c *(np.log(img+1))
    else:
        img2 = img  
    
    # c = 255/np.log(1+np.max(img))
    # img3 = c *(np.log(img+1))
    
    ##Run Ridgefinder
    px, py, nx, ny, eigvals, valid = points_out = RF.find_points(img2, sigma=SIGMA, l_thresh = lthresh, u_thresh=uthresh)
    lines_before, junctions = RF.compose_lines_from_points(points_out)
    
    ##Link the Ridges
    nlines = RF.linklines(lines_before,minlen,linkthresh)
    lines=lines_before
    
    ##Set some plot properties
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle(str(imnumb))
    
    ax1.imshow(img, cmap="jet")
    ax1.set_title('Trajectory and Initial Direction')
    # ax2.imshow(img3, cmap="magma")
    ax2.set_title('Bragg Curve along Trajectory')
    lim =RF.get_lines_bounding_box(lines)
    
    ##Create and plot the splinefit for all linked ridgepoints            
    line = nlines[track]
    if len(line[1]) > minlen:
        
            x = px[line[1], line[0]]
            y = py[line[1], line[0]]   
            
            
            
            
            ##Get the splinefit for the image
            new_points1,der_points1 = RF.getspline(x,y,ss=2)
            Bragg = RF.simplebragg(new_points1[0],new_points1[1],img)
            peak = np.where(Bragg == max(Bragg))[0][0]
            if peak < len(Bragg)/2:
            #Init Dir is at the end of the line list, so we reverse it
                new_points1[1]=new_points1[1][::-1]
                new_points1[0]=new_points1[0][::-1]
            else:
                new_points1[1]=new_points1[1]
                new_points1[0]=new_points1[0]
                
            Track_length =np.hypot(np.diff(x), np.diff(y)).sum() #Determine tracklength
            # Track_length_micm = Track_length*pscale
            
            print("Track Length is : ", Track_length)
            
            
            ax1.plot(new_points1[1],new_points1[0],'-',color='white',linewidth=2)
            # ax2.plot(new_points1[1],new_points1[0],'-',color='white')
            Bragg = RF.simplebragg(new_points1[0],new_points1[1],img)
            Bragg2 = RF.InitialBragg(new_points1[0],new_points1[1],img,d=2)
            
            Bxscale = np.linspace(0,Track_length, len(Bragg2))
            
            indir,(x0,y0) = RF.initdir_simdat(new_points1[1],new_points1[0],pix=1)
            slope = abs(np.tan(indir))
            # leng = 15
            if indir >= -np.pi and indir < -np.pi/2:
                dy = -leng/(np.sqrt(1+slope**2))
                dx = slope*dy
            elif indir >= -np.pi/2 and indir < 0:
                dy = leng/(np.sqrt(1+slope**2))
                dx = -slope*dy
            elif indir >= 0 and indir < np.pi/2:
                dy = leng/(np.sqrt(1+slope**2))
                dx = slope*dy
            else:
                dy = -leng/(np.sqrt(1+slope**2))
                dx = -slope*dy
            colors = ["red","yellow","orange","green","blue","purple"]
            
            
            
            ax1.arrow(new_points1[1][pt],new_points1[0][pt],dx,dy,linewidth=2,head_width=2, head_length=2,ec=colors[1],fc=colors[1])
            # ax1.annotate(str(round(indir*180/np.pi,2)),(x0,y0-2),color = "white")
            # ax1.hlines(y0-13,x0-9.3,x0,color="red",linewidth=3)
            
            ax1.annotate(str(round(indir*180/np.pi,2)),(x0+2,y0+2),color = "white")
            ax1.hlines(sty ,stx,stx+6.6,color="yellow",linewidth=3)
            ax1.annotate("1 mm",(stx,sty+1),color = "yellow")
            # ax2.plot(Bragg)#,label=str(i),color=colors[i])
            ax2.plot(Bxscale,Bragg2)
            ax2.annotate("Track Length is:\n"+str(round(Track_length,2))+" pix",(0,max(Bragg2)*0.95),color='black')
            ax2.set_xlabel("Length along Track (pix)")
            ax2.set_ylabel("Pixel Intensity")
            # indir = indir*180/np.pi
            # ax1.legend()
            # ax2.legend()
            # print(indir)

                
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
    Num_trax=len(glob.glob(directory+'*.fits'))+len(glob.glob(directory+'*.fit'))
    # print(directory)
    for kk in range(Num_trax):
        try:
            make_plot(kk,directory,sigma, lt, ut, minlen, linkthresh, logim = False)
        except Exception as e:
            print(e)
            er_plot(kk,directory)
        _ = input("Press [enter] to continue.")

def make_plot_0(img,sigma, lt, ut, minlen, linkthresh, logim = False):
    '''
    Create a figure containing the image of the particle track and its
    ridgeline. Two plots are produced in this figure. One is in linear space
    and the other is in log space. On both, the unlinked and linked ridges are
    plotted. The linked are plotted as a thin white line overlayed on the 
    colored points of the unlinked lines.
    
    This can be iterated through.
    
    Parameters
    ----------
    img: numpy array
        The image for analysis.
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
    xx,yy : numpy array
        The points on the last ridge that was found.

    '''

    ################################################################
    ##Initialize matplotlib
    plt.close()
    plt.rc('font', size=10)

    

    ################################################################
    #These are the Parameters to change to work with the RF        #
    ################################################################
    
    SIGMA = sigma #sigma for derivative determination ~> Related to track width
    lthresh = lt #tracks with a response lower than this are rejected (0 accepts all)
    uthresh = ut #tracks with a response higher than this are rejected (0 accepts all)
    minlen = minlen #minimum track length accepted
    linkthresh = linkthresh #maximum distance to be linked
    # thresh = 0.0001

    
    
    if logim:
        c = 255/np.log(1+np.max(img))
        img2 = c *(np.log(img+1))
    else:
        img2 = img  
    
    c = 255/np.log(1+np.max(img))
    img3 = c *(np.log(img+1))
    
    # img2 = img2[np.isfinite(img2)]
    ##Run Ridgefinder
    px, py, nx, ny, eigvals, valid = points_out = RF.find_points(img2, sigma=SIGMA, l_thresh = lthresh, u_thresh=uthresh)
    lines_before, junctions = RF.compose_lines_from_points(points_out)
    
    ##Link the Ridges
    nlines = RF.linklines(lines_before,minlen,linkthresh)
    lines=lines_before
    
    ##Set some plot properties
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8), dpi=80, facecolor='w', edgecolor='k')
    
    
    ax1.imshow(img, cmap="magma")
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
                try:
                    new_points,der_points = RF.getspline(x,y,ss=1)    
                    ax1.plot(new_points[1],new_points[0],'.')
                    ax2.plot(new_points[1],new_points[0],'.')
                except Exception as e: print(e)

    ##Create and plot the splinefit for all linked ridgepoints            
    for i, line in enumerate(nlines):
        if len(line[1]) > minlen:
            
                x = px[line[1], line[0]]
                y = py[line[1], line[0]]   
                
                ##Get the splinefit for the image
                try:
                    new_points1,der_points1 = RF.getspline(x,y,ss=2)    
                    ax1.plot(new_points1[1],new_points1[0],'-',color='white')
                    ax2.plot(new_points1[1],new_points1[0],'-',color='white')
                    
                    xx = new_points1[0]
                    yy = new_points1[1]
                except Exception as e: print(e)

    
    plt.show()
    plt.pause(0.1) ##This command allows the cycling to actually happen
    return xx,yy
    