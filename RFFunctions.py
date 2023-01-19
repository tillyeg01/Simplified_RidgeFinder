import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix
from skimage.filters import gaussian as gaussian_filter
import scipy.ndimage
from scipy import ndimage as nd
from scipy.optimize import curve_fit
import scipy.interpolate as interpolate
from scipy.stats import norm
import plotly.graph_objects as go

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#This first section is the Ridgefinder algorithm as developed by C. Steger and adapted for#
#python by T. Neep.                                                                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_derivatives(image, sigma):
    # We actually do the gaussian filter twice, (the additional call is in in
    # hessian_matrix) - not time critical but potential place to optimize
    hrr, hrc, hcc = hessian_matrix(image, sigma=sigma)
    hx, hy = np.gradient(gaussian_filter(image, sigma=sigma))
    return hx, hy, hrr, hrc, hcc


def find_points(image, sigma, l_thresh=0, u_thresh=0, *, dark=True):
    """The first step of the Steger algorithm is to find the points in the image
    that we construct lines from.

    This will return px and py, which are the sub-pixel positions of the line;
    nx and ny, which are the x and y components of the eigenvector corresponding
    to the maximum absolute eigenvalue.
    """

    # Get first and second derivatives
    hx, hy, hrr, hrc, hcc = get_derivatives(image, sigma=sigma)

    # Compute all eigenvalues and vectors
    stack = np.stack([[hrr, hrc], [hrc, hcc]]).T
    stack = np.swapaxes(stack, 0, 1)
    # I expect there is a better/faster way of doing this but I'm not sure
    eigvals, eigvecs = np.linalg.eigh(stack.reshape(-1, 2, 2))

    eigvals = eigvals.reshape(*image.shape, 2)
    eigvecs = eigvecs.reshape(*image.shape, 2, 2)

    # Find the absolute maximum eigenvalue and corresponding eigenvector
    max_ev = np.argmax(np.abs(eigvals), axis=-1)[..., np.newaxis]
    eigvals = np.take_along_axis(eigvals, max_ev, axis=-1).squeeze()
    eigvecs = np.take_along_axis(eigvecs, max_ev[..., np.newaxis], axis=-1,).squeeze()

    ny = eigvecs[..., 0]
    nx = eigvecs[..., 1]

    # Dark lines need inverting
    if dark:
        eigvals *= -1

    # Ignore warnings about divide by zero. Ok to have NaNs here
    with np.errstate(invalid="ignore"):
        t = -(hx * nx + hy * ny) / (hcc * nx ** 2 + 2 * hrc * nx * ny + hrr * ny ** 2)

    px = t * nx
    py = t * ny

    # The paper suggests px and py should both have |p| < 0.5 but it doesn't
    # seem to be a hard rule. It seems having a slightly looser requirement can
    # sometimes help. To be investigated
    in_range = (np.abs(px) < 0.5) & (np.abs(py) < 0.5)

    px += np.arange(image.shape[0])[np.newaxis].T
    py += np.arange(image.shape[1])
    
    
    # Restrict which ridgelines will be kept based on thresholds applied to the 
    # eigenvalues  
    if l_thresh != 0 and u_thresh != 0:
        valid = (eigvals >= l_thresh) & (eigvals <= u_thresh) & in_range
    elif l_thresh != 0:
        valid = (eigvals >= l_thresh) & in_range
    elif u_thresh != 0:
        valid = (eigvals > 0) & (eigvals <= u_thresh) & in_range
    else:
        valid = (eigvals > 0) & in_range
    

    return px, py, nx, ny, eigvals, valid


OCTANTS = np.array(
    [
        [[0, 1], [-1, 1], [-1, 0]],  # NE
        [[-1, 1], [-1, 0], [-1, -1]],  # N
        [[-1, 0], [-1, -1], [0, -1]],  # NW
        [[-1, -1], [0, -1], [1, -1]],  # W
        [[0, -1], [1, -1], [1, 0]],  # SW
        [[1, -1], [1, 0], [1, 1]],  # S
        [[1, 0], [1, 1], [0, 1]],  # SE
        [[1, 1], [0, 1], [-1, 1]],  # E
    ]
)


def find_next_points(row, col, octant, mirror=False):
    """Find the three next points to query. At this point there is no
    requirement that they are within the image bounds, that is done
    seperately"""
    return np.array([row, col]) + OCTANTS[octant] * (-1 if mirror else 1)


def check_next_points(points, shape):
    """Checks whether or not points go out of bounds"""
    less_than_zero = np.any(points < 0, axis=1)
    out_of_range = (points[:, 0] >= shape[0]) | (points[:, 1] >= shape[1])
    return points[~(out_of_range | less_than_zero)]


def _debug_lines(image, line, next_points):
    """Useful for checking algorithm works as expected"""
    import matplotlib.pyplot as plt

    plt.imshow(image, cmap="gray_r")
    tmp_line = np.array(line).T
    plt.plot(py[tmp_line[1], tmp_line[0]], px[tmp_line[1], tmp_line[0]], ".-")
    plt.plot(next_points[:, 1], next_points[:, 0], "or")
    plt.show()


def compose_lines_from_points(points_output):
    """Join the points found in `find_points` into lines (and junctions)"""
    px, py, nx, ny, eigvals, valid = points_output

    alphas = np.arctan2(ny, nx)
    bins = np.arange(-np.pi + np.pi / 8, np.pi, np.pi / 4)
    # Octants goes from 0 - 7 anti-clockwise starting at SW
    # i.e. SW, W, SE, E, NE, N, NW, W
    octants = np.digitize(alphas, bins, right=True)
    octants[octants == 0] = 8
    octants -= 1

    assert not np.any(octants < 0)
    assert not np.any(octants > 7)

    active = np.where(valid, 0, -1)
    eigvals_ = eigvals.copy()
    eigvals_[active != 0] = -1

    lines = []
    junctions = []

    def iter_start(eigvals):
        sorted_eigs = np.argsort(eigvals, axis=None)[: np.sum(~valid) : -1]
        for val in sorted_eigs:
            row, col = np.array(np.unravel_index(val, eigvals.shape))
            if active[row, col] == 0:
                yield row, col

    for row, col in iter_start(eigvals_):
        line = [(col, row)]

        # We have two directions to travel along the line
        for iteration in range(2):

            # If iteration == 1 then flip the line
            if iteration == 1:
                col, row = line[0]
                line = line[::-1]

            while True:

                active[row, col] = len(lines) + 1

                alpha_1 = alphas[row, col]
                p1 = (px[row, col], py[row, col])

                octant = (octants[row, col]) % 8

                next_points = find_next_points(row, col, octant, mirror=bool(iteration))
                next_points = check_next_points(next_points, alphas.shape)

                # If no valid next points then stop the line
                if not len(next_points):
                    break

                indices = next_points.T[0], next_points.T[1]

                new_alphas = alphas[indices]
                new_ps = (px[indices], py[indices])

                d = np.hypot(new_ps[0] - p1[0], new_ps[1] - p1[1])

                def norm_angle(x):
                    x[x >= np.pi] -= 2 * np.pi
                    x[x < -np.pi] += 2 * np.pi
                    x = np.abs(x)
                    x[x > np.pi / 2] = np.pi - x[x > np.pi / 2]
                    return x

                beta = norm_angle(new_alphas - alpha_1)
                # c is a constant defined in the paper and set to 1
                c = 1

                vals = d + c * beta
                vals[active[indices] < 0] = np.inf

                if np.isinf(vals).all():
                    # print(iteration)
                    # _debug_lines(image, line, next_points)
                    break

                row, col = next_points[np.argmin(vals)]

                # Append point as col (x) and row (y)
                line.append((col, row))

                # Now check if that point was already on a line, if so stop this
                # line
                if active[row, col] > 0:
                    junctions.append((col, row))
                    break

                # We can have a situation where the next alpha point is flipped
                # by np.pi radians. Here we check if this is the case and if so
                # flip to match the previous point we checked.
                dalpha = alphas[row, col] - alpha_1
                if dalpha > np.pi:
                    dalpha -= 2 * np.pi
                if dalpha <= -np.pi:
                    dalpha += 2 * np.pi

                if np.abs(dalpha) > np.pi / 2:
                    alphas[row, col] = np.sign(alphas[row, col]) * (
                        np.abs(alphas[row, col]) - np.pi
                    )
                    octants[row, col] = (octants[row, col] + 4) % 8

        if len(line) > 1:
            lines.append(np.array(line).T)

    return lines, np.array(junctions).T

def get_lines_bounding_box(lines):
    xmin = min(min(line[0]) for line in lines)
    xmax = max(max(line[0]) for line in lines)
    ymin = min(min(line[1]) for line in lines)
    ymax = max(max(line[1]) for line in lines)
    return min(xmin, ymin) - 5, max(xmax, ymax) + 5


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#This next section contains all of the algorithms for determining initial direction.      #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def initdir_simdat(x,y,pix = 1, show = False):
    """
    Assumes that the track is oriented corectly.
    This uses the angle between the first point of the line 
    and the point "pix" away.

    Parameters
    ----------
    x : LIST
        The x coordinates of the ridgeline.
    y : LIST
        The y coordinates of the ridgeline.
    pix: INT
        How many pixels to include in finding the initial direction.
    show: BOOL
        Set True to help visualize the initial angle.

    Returns
    -------
    initdir : FLOAT
        Initial angle of the track in radians.
    (xs[0],ys[0]): POINT
        First point of the ridgeline where the initial direction is measured from. 

    """
    xs = x.copy()
    ys = y.copy()
    

    # diffx = xs[pix]-xs[0]
    # diffy = ys[pix]-ys[0]
    
    pt=10
    
    diffx = xs[pt]-xs[pt-1]
    diffy = ys[pt]-ys[pt-1]
    
    initdir = np.arctan2(diffx,diffy)
    
    if show:
        angle2 = initdir*180/np.pi
        slope = np.tan(initdir)
        dx = 1/(np.sqrt(1+slope**2))
        dy = slope*dx
        plt.arrow(y[0],x[0],dx,-dy,head_width=1, head_length=1,ec='white',fc="yellow")
        plt.annotate(str(round(angle2,2)),(y[0],x[0]),color = "white")
    return initdir, (xs[0],ys[0])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#These are outdated, but I left them in just in case I come back to them later.           #
#Note: the documentation on them may also be wrong...                                     #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def initdir_diff(x,y,Bragg,pix = 6, show = False):
    """
    Uses the Bragg curve to determine head/tail relations and then from there
    uses the x,y coordinates of the ridgeline to get initial angular information
    for an Electron Track. This uses the angle between the first point of the line 
    and the point "pix" away.

    Parameters
    ----------
    x : LIST
        The x coordinates of the ridgeline.
    y : LIST
        The y coordinates of the ridgeline.
    Bragg : LIST
        Bragg curve of the track along the ridgeline.
    pix: INT
        How many pixels to include in finding the initial direction.

    Returns
    -------
    initdir : FLOAT
        Initial angle of the track in radians.

    """
    #Find the peak of the Bragg curve and use this information to determine head/tail
    peak = np.where(Bragg == max(Bragg))[0][0]
    xs = np.array(x).astype('float64')
    ys = np.array(y).astype('float64')
    
    arrowlen = 3
    
    #Determine which end of the list contains the head of the track and find
    #the initial direction of the track from there. 
    if peak < len(Bragg)/2:
    #Init Dir is at the end of the line list, so we reverse it
        xs=xs[::-1]
        ys=ys[::-1]
      
    diffx = xs[pix]-xs[0]
    diffy = ys[pix]-ys[0]
    
    initdir = np.arctan2(diffx,diffy)
    
    if show:
        angle2 = angle*180/np.pi
        slope = np.tan(initdir)
        dx = arrowlen/(np.sqrt(1+slope**2))
        dy = slope*dx
        plt.arrow(y[0],x[0],dx,-dy,head_width=1, head_length=1,ec='white',fc="yellow")
        plt.annotate(str(round(angle2,2)),(y[0],x[0]),color = "white")
    return initdir, (xs[0],ys[0])




def initdir_avg(x,y,Bragg,pix = 6, show = False):
    """
    Uses the Bragg curve to determine head/tail relations and then from there
    uses the x,y coordinates of the ridgeline to get initial angular information
    for an Electron Track. This algorithm takes the average of the angles between
    each of the points from 0 to "pix" in the ridge.

    Parameters
    ----------
    x : LIST
        The x coordinates of the ridgeline.
    y : LIST
        The y coordinates of the ridgeline.
    Bragg : LIST
        Bragg curve of the track along the ridgeline.
    pix: INT
        How many pixels to include in finding the initial direction.

    Returns
    -------
    initdir : FLOAT
        Initial angle of the track in radians.

    """
    #Find the peak of the Bragg curve and use this information to determine head/tail
    peak = np.where(Bragg == max(Bragg))[0][0]
    xs = np.array(x).astype('float64')
    ys = np.array(y).astype('float64')
    
    if peak < len(Bragg)/2:
    #Init Dir is at the end of the line list, so we reverse it
        xs=xs[::-1]
        ys=ys[::-1]
    
    y1 = np.append(ys,0)
    x1 = np.append(xs,0)

    y2 = np.insert(ys,0, 0)
    x2 = np.insert(xs,0, 0)
        
    arrowlen = 3
    
    diffx = (x2-x1)[1:len(x2-x1)-1]
    diffy = -(y2-y1)[1:len(y2-y1)-1]
    
    angle = -np.arctan2(diffx,diffy)
    
    initdir = np.mean(angle[0:pix],dtype=np.float64)

    if show:
        for i in range(len(angle)):
            slope1 = np.tan(angle[i])
            dx = 1/(np.sqrt(1+slope1**2))
            dy = slope1*dx
            plt.arrow(ys[i],xs[i],dx,dy,head_width=0.3, head_length=0.3,ec='white')
        slope = np.tan(initdir)
        dx = arrowlen/(np.sqrt(1+slope**2))
        dy = slope*dx
        
        plt.arrow(y[0],x[0],dx,-dy,head_width=1, head_length=1,ec='white',fc="yellow",label = str(i))
        plt.annotate(str(round(initdir*180/np.pi,2)),(y[0],x[0]),color = "white")
        plt.legend()
     
        
    return initdir, (xs[0],ys[0])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#This next section contains all of the random functions that I've created for the analysis#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    
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

def getclumps(img, thresh,minmask = 20,maxmask=10000):
    ##WIP--WIP--WIP##
    ##Finds patches of an image above a given threshold for isolating tracks##
    XsYs = BinaryMap(img,thresh) #Creates a binary map of the image above a given threshold
    labelled_array, numOfLabels = nd.label(XsYs, structure = np.ones((3,3),int))
    values, counts = np.unique(labelled_array, return_counts=True)
    counts_mask1 = counts > minmask #Excludes track that are too short
    counts_mask2 = counts < minmask #Excludes overlapping tracks
    counts_mask = counts_mask1*counts_mask2 
    filtered = filter(None, counts_mask*values) 
    clumpy_label = []
    for i in filtered:
        clumpy_label.append(i)
    return clumpy_label,labelled_array

def gaussian(x, amp, mu, sigma):
    return amp*np.exp(-np.power(x - mu, 2)/(2*np.power(sigma, 2)))

def plotgaus(A,mean,sigma):
    ##For plotting a gaussian on a histogram##
    xs = np.linspace(-180,180,num=360)
    ys = gaussian(xs,A,mean,sigma)
    plt.plot(xs,ys)

def InitialBragg(x,y,image,d=5,show = False):
    """
    Takes an input ridgeline and image and calculates a Bragg curve using 
    tangential profiles to the ridgeline.
    
    Parameters
    ----------
    x,y: LISTS
         The ridgeline for the track to be analyzed.
    
    imgae : ARRAY
        Image with ridgelines to be analyzed. 
        
    Returns
    -------
    A list containing the Bragg curve along with the parameters and the
    covariance of the gaussian fit to it.
    
   
    """
    
    y1 = np.append(y,0)
    x1 = np.append(x,0)

    y2 = np.insert(y,0, 0)
    x2 = np.insert(x,0, 0)

    diffx = (x1-x2)[1:len(x1-x2)-1]
    diffy = -(y1-y2)[1:len(y1-y2)-1]

    angle = np.arctan2(diffx,diffy)
    NextBragg = []
    
    for i, theta in enumerate(angle):

        slope = np.tan(np.pi/2+theta)
        
        dx = d/(np.sqrt(1+slope**2))
        dy = slope*dx
        init = (x[i]+dx,y[i]+dy)
        fin = (x[i]-dx,y[i]-dy)
        xblank = np.linspace(init[0],fin[0],num=2*d)
        yblank = np.linspace(init[1],fin[1],num=2*d)
        z = image
        zi = scipy.ndimage.map_coordinates(z, np.vstack((xblank,yblank)))
        NextBragg.append(np.sum(zi))
        amp = max(zi)
        mean,std=norm.fit(zi)
        xx = np.linspace(0,len(zi),num=len(zi))
        # pars, cov = curve_fit(f=gaussian, xdata=xx, ydata=zi, p0=[amp, len(zi)/2, 4], bounds=(0, np.inf))
        
        if show:
            plt.figure(1)
            plt.plot([init[1],fin[1]],[init[0],fin[0]],'o-')        
            plt.show()
            
    return NextBragg#,pars,cov

def simplebragg(x,y,img):
    ##Create a first-order Bragg curve using the intensity at each point
    ##along the ridgeline
    xint = [int(m) for m in x]
    yint = [int(n) for n in y]
    xint = np.array(xint)
    yint = np.array(yint)
    SimpleBragg = img[xint,yint]
    return SimpleBragg

### OLD CODE ###
# def getspline(x,y,ss=0,Nmult=3):
#     ##Find a splinefit of the ridgeline
    
#     N = len(x)*Nmult
#     xx = np.linspace(0, 1, N)
#     tck, u = interpolate.splprep([x,y],s=ss)
#     new_points = interpolate.splev(xx,tck)
#     ##Get the derivative of this point
#     der_points = interpolate.splev(xx,tck,der=1)
#     return new_points, der_points

def getspline(x,y,ss=0,Nmult=3):
    ##Find a splinefit of the ridgeline
    N = len(x)*Nmult
    xx = np.linspace(0, 1, N)
   
    #INCLUDED HERE IS A FIX TO A BUG IN SCIPY - see: https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs
    bad = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) == 0)
   
    for i in range(len(bad)):
        np.delete(x,bad[i])
        np.delete(x,bad[i]+1)
        np.delete(y,bad[i])
        np.delete(y,bad[i]+1)

    #end of fix

    tck, u = interpolate.splprep([x,y],s=ss)
    new_points = interpolate.splev(xx,tck)
    ##Get the derivative of this point
    der_points = interpolate.splev(xx,tck,der=1)
    return new_points, der_points



        
        
def circ_hist(hist,title,color="blue"):
    ##Create a circular histogram from an existing histogram##
    fig = go.Figure(go.Barpolar(
        r=hist[0],
        theta=hist[1],
        marker_color = color,
        marker_line_color = color
        
    ))

    fig.update_layout(
        title = {
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig.show()
            
        
        

def openandconvert(file):
    ##Takes the degrad output and converts it to readable lists##
    tracks = []
    rawfile = open(file,"r")
    filecontents = rawfile.readlines()
    trackdata = filecontents[1::2]
    trackiddat = filecontents[0::2]
    trackids = []
    trackdat =[]
    for i in range(len(trackdata)):
        tracks.append(np.fromstring(trackdata[i],dtype=float,sep=" "))
        trackids.append(np.fromstring(trackiddat[i],dtype=float,sep=" ")[0])
        trackdat.append(np.fromstring(trackiddat[i],dtype=float,sep=" "))
    return tracks,trackids,trackdat        	




def plotsplinetans(x,y,dx,dy,ll=1,figno=1):
    ##Plot the tangents along the whole splinefit##
    for i, xx in enumerate(x):
        norm = np.sqrt(dx[i]**2+dy[i]**2)
        p1 = (x[i]-ll*dx[i]/norm,x[i]+ll*dx[i]/norm)
        p2 = (y[i]-ll*dy[i]/norm,y[i]+ll*dy[i]/norm)
        plt.figure(figno)
        plt.plot(p1,p2,'-')
    
        
        
        
def track_start(x,y,image,lenrej=True):
    ##Eliminates the ridgeline points at the beginning of the track
    ##that fall on pixels that are zero
    xpop = []
    ypop = []
    xrej =[]
    yrej = []
    for i, xx in enumerate(x):
        if image[int(x[i]+0.5),int(y[i]+0.5)]==0:
            xpop.append(i)
            ypop.append(i)
            xrej.append(x[i])
            yrej.append(y[i])
        else:
            break
    len_rej = np.hypot(np.diff(xrej), np.diff(yrej)).sum()*176/10**3
    xnew = np.delete(x,xpop)
    ynew = np.delete(y,ypop)
    if lenrej:
        return xnew,ynew,len_rej
    else:
        return xnew,ynew

def track_obscure(x,y,radius, pixscale=176, origin = [128,128]):
    #This is to simulate how much track is actually visible after the first
    #"radius" of the track is hidden by the NR
    #Note, this is from the origin of the image (only use this with sim data)
    #note: radius in mm; pixscale in micrometer/pix
    rad = radius*10**3/pixscale
    
    xpop = []
    ypop = []
    for i, xx in enumerate(x):
        dis = dist(origin, (x[i],y[i]))
        if dis < rad:
            xpop.append(i)
            ypop.append(i)
        else:
            break
    xnew = np.delete(x,xpop)
    ynew = np.delete(y,ypop)
    return xnew,ynew

def length_inside(x,y,radius, pixscale=176, origin = [128,128]):
    #This is to determine the length of track obscured when the 
    #beginning of the track is hidden by the NR
    #Note, this is from the origin of the image (only use this with sim data)
    #note: radius in mm; pixscale in micrometer/pix
    rad = radius*10**3/pixscale
    
    xins = []
    yins = []
    for i, xx in enumerate(x):
        dis = dist(origin, (x[i],y[i]))
        if dis < rad:
            xins.append(x[i])
            yins.append(y[i])
        else:
            break
    leninside = np.hypot(np.diff(xins),np.diff(yins)).sum()
    return leninside
        

def dist(p1,p2):
    #Easily calculates distance between two points
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


##Histogram Analysis##
def mean_h(val, freq):
    return np.average(val, weights = freq)

def var_h(val, freq):
    dev = freq * (val - mean_h(val, freq)) ** 2
    return dev.sum() / freq.sum()



def var_unweight(val,freq):
    dev = (val-mean_h(val,freq))**2
    var=dev.sum()
    return var


def moment_h(val, freq, n):
    n = (freq * (val - mean_h(val, freq)) ** n) / freq.sum()
    d = var_h(val, freq) ** (n / 2)
    return n / d        

def hist_stats(n,bins):
    mids = 0.5*(bins[1:]+bins[:-1])
    mu = np.average(mids, weights=n)
    var = np.average((mids-mu)**2,weights=n)
    amp = n[int(mu)]#max(n)
    return mu, var, amp

# def checkstats(data,pri=False):
#     for i in range(3,360,3):
#             n,bins,other = plt.hist(res_spid,bins=i,range=[-180,180])

#             mu,var,amp = hist_stats(n,bins)

#             sigma = np.sqrt(var)
#             plt.vlines(mu,0,10,label='Mean',color='yellow')
#             if pri:
#                 print(i)
#                 print('mean = ',mu,' sigma = ',sigma, ' FWHM = ',2.355*sigma)
#             means.append(mu)
#             fwhms.append(2.355*sigma)
#     plt.close()
#     plt.figure(1)
#     plt.plot(means,'.-')
#     plt.plot(fwhms,'.-')
#     plt.show()

def fallsbetween(hist,ang1,ang2):
    ##Determines how many values fall between two angles using a histogram##
    count = 0
    for i in range(len(hist[1])):
        if hist[1][i] >=ang1 and hist[1][i]<=ang2:
            count += hist[0][i]
    frac = count/np.sum(hist[0])
    return count, frac

def create_circular_mask(h, w, rad,pixscale = 176, center=[128,128]):
    radius = rad*10**3/pixscale
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    return mask


##WIP---WIP----WIP###  
import itertools
          
def linklines(lines,minlen,thresh):
    ##Make sure we're only looking at RLs that are longer than the minlen
    lines = [line for i,line in enumerate(lines) if len(line[0])>minlen]
    while True:
        ##Make sure there's more than one RL that we're looking at
        if len(lines) <=1:
            break
        
        dislist=[]
        indlist=[]
        
        ##Look for the two RLs that are closest to one another
        ##If they're close enough together, concatenate them and replace the old two RLs with the new one
        for line1, line2 in itertools.combinations(enumerate(lines),2):
            f1 = line1[1][0][0],line1[1][1][0] #first point of line1
            e1 = line1[1][0][-1],line1[1][1][-1] #last point of line1
            f2 = line2[1][0][0],line2[1][1][0] #first point of line2
            e2 = line2[1][0][-1],line2[1][1][-1] #last point of line2
            indlist.append([line1[0],line2[0]]) #Indices of the two tracks being compared
            dislist.append([dist(f1,f2),dist(f1,e2),dist(e1,f2),dist(e1,e2)]) #4 distances between the 4 endpoints
        mins = [min(dislist[i]) for i in range(len(dislist))] #Finds the smallest distance between endpoints of all RLs (i.e. the two closest RLs)
        mindex = mins.index(min(mins)) #Index in dislist of closest RLs
        
        ##Make sure our RLs are close enough together to be linking them
        if mins[mindex]>=thresh:
            break
        
        pindex = dislist[mindex].index(min(dislist[mindex])) #Number in range 0-3 corresponding to which endpoints are closest (to correctly orient the tracks)
        l1_ind,l2_ind = indlist[mindex] #Which two ridgelines are actually closest to each other
        l1 = lines[l1_ind]
        l2 = lines[l2_ind]
        if pindex == 0:
            l1[0] = l1[0][::-1] #Make sure line orientation is correct
            l1[1] = l1[1][::-1] #Make sure line orientation is correct
            newline = np.concatenate((l1,l2),axis=1) #Combine the two lines
        if pindex == 1:
            newline = np.concatenate((l2,l1),axis=1) #Combine the two lines
        if pindex == 2:
            newline = np.concatenate((l1,l2),axis=1) #Combine the two lines
        if pindex == 3:
            l2[0] = l2[0][::-1] #Make sure line orientation is correct
            l2[1] = l2[1][::-1] #Make sure line orientation is correct
            newline = np.concatenate((l1,l2),axis=1) #Combine the two lines
        lines[l1_ind]=newline
        lines.pop(l2_ind)
    return lines


        