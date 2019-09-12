import numpy as np
import skimage as sk
import pandas as pd
import mpu
import os
from math import degrees, radians, pi, atan2, sin, cos
from scipy import ndimage
pd.options.mode.chained_assignment = None # suppress waring messages for in-place dataframe edits

def getcosine(testpoint, center_of_fluor):
    dotproduct = testpoint[0]*center_of_fluor[0] + testpoint[1]*center_of_fluor[1]
    magnitude = (testpoint[0]**2 + testpoint[1]**2)**0.5 * (center_of_fluor[0]**2 + center_of_fluor[1]**2)**0.5
    result = dotproduct / magnitude
    return result

def fluor_polarity_3d(fluor_chan, bmask, cell_tracks):
    """
    Calculates polarity of a fluorescence signal within an object at each time
    point across a time series. A DataFrame containing tracking info for these
    objects (a unique label for each tracked object and the x-y position at
    each time point) needs to be included as this function does not incorporate
    any tracking algorithms.
  
    Parameters
    ----------
    fluor_chan: ndarray
        A 3d image stack (t, x, y) of the fluorescence signal to be measured.
        
    bmask: ndarray
        A 3d image stack (t, x, y) containing binary masks of the objects in
        'fluor_chan'. The shapes of 'fluor_chan' and 'bmask' must be identical.
        
    cell_tracks: DataFrame
        This DataFrame summarizes the x-y positions of previously-tracked
        objects that each have a unique label. EXACTLY four columns need to be
        present and named as follows:        
        'Time_s' contains the timepoints of each x-y frame in a 3d time series.
        'Object_id' contains a unique label for each object that was tracked
        in time.
        'X' contains the x coordinate of the geometric center of each object
        at each timepoint.
        'Y' contains the y coordinate of the geometric center of each object
        at each timepoint.
  
    Returns
    -------
    output: DataFrame 
        This DataFrame contains the original 'Time_s' and 'Object_id' columns,
        with the further addition of 'Distance_polarity_score' and
        'Angular_polarity_score' columns, indicating the polarity measurement
        of each object at each timepoint.
        
    """
    assert type(bmask) is np.ndarray, "Binary masks are not a numpy array!"
    assert type(fluor_chan) is np.ndarray, "Fluorescence images are not a numpy array!"
    assert fluor_chan.shape == bmask.shape, "Fluorescence image and binary mask are different dimensions!"
    assert type(cell_tracks) is pd.DataFrame, "'cell tracks' need to be formatted as a pandas DataFrame!"
    assert 'Time_s' in cell_tracks, "'cell_tracks' is missing 'Time_s' column!"
    assert 'Object_id' in cell_tracks, "'cell_tracks' is missing 'Object_id' column!"
    assert 'X' in cell_tracks and 'Y' in cell_tracks, "'cell_tracks' is missing 'X' and/or 'Y' column(s)!"
    assert len(cell_tracks.columns) == 4, "'cell_tracks' must contain EXACTLY five columns labeled 'Time_s', 'Object_id, 'X', and 'Y'!"
    
    #reformat input DataFrame so order of columns is correct for numpy arrays that are created later
    sortdata = pd.DataFrame(columns=[])
    sortdata['Time_s'] = cell_tracks['Time_s']; sortdata['Object_id'] = cell_tracks['Object_id']
    sortdata['X'] = cell_tracks['X']; sortdata['Y'] = cell_tracks['Y']
    cell_tracks = None; cell_tracks = sortdata; sortdata = None
           
    time_intv = cell_tracks.loc[1, 'Time_s'] - cell_tracks.loc[0, 'Time_s'] # for determining time interval between each frame
    final_table = pd.DataFrame(columns=[])    
    img_labels = np.zeros(bmask.shape, dtype=int)

    for x, frame in enumerate(bmask):
        img_labels[x] = sk.measure.label(frame)       

    areascol = []; labelscol = []; objs = []; intlocscol = []; cenlocscol = []; xlist = []; major_axes_col = []

    # label objects in binary mask and get x-y positions for the geometric center
    # and weighted fluorescence intensity center for each labeled object
    for x, frame in enumerate(img_labels):
        areas = [r.area for r in sk.measure.regionprops(frame)]
        labels = [r.label for r in sk.measure.regionprops(frame)]
        major_axes = [r.major_axis_length for r in sk.measure.regionprops(frame)]
        intlocs = [list(r.weighted_centroid) for r in sk.measure.regionprops(
            frame, intensity_image=fluor_chan[x,:,:])]
        cenlocs = [list(r.weighted_centroid) for r in sk.measure.regionprops(
            frame, intensity_image=bmask[x,:,:])]
        areascol.append(areas); labelscol.append(labels); intlocscol.append(intlocs); cenlocscol.append(cenlocs); major_axes_col.append(major_axes)
        y = 0
        while y < np.amax(frame):
            xlist.append(x)
            y += 1

    # make a numpy array from all the lists generated in preceding 'for' loop
    flatarea = mpu.datastructures.flatten(areascol)
    flatlabel = mpu.datastructures.flatten(labelscol)
    flatmajor_axes = mpu.datastructures.flatten(major_axes_col)
    flatcoords = mpu.datastructures.flatten(intlocscol)
    flatcoords = np.reshape(np.asarray(flatcoords), (len(flatcoords)//2, 2))
    flatcencoords = mpu.datastructures.flatten(cenlocscol)
    flatcencoords = np.reshape(np.asarray(flatcencoords), (len(flatcencoords)//2, 2))
    objs.append(xlist); objs.append(flatlabel); objs.append(flatarea); objs.append(flatmajor_axes)
    objs = np.transpose(np.asarray(objs))
    objs = np.concatenate((objs,flatcoords, flatcencoords),axis = 1)
    areascol = None; labelscol = None; intlocscol = None; cenlocscol = None

    # normalize distances between geometric and fluorescence center to origin to make later calcuations easier
    absx = objs[:,4] - objs[:,6]
    absy = objs[:,5] - objs[:,7]
    absx = np.reshape(np.asarray(absx), (len(absx), 1))
    absy = np.reshape(np.asarray(absy), (len(absy), 1))

    objs = np.concatenate((objs, absx, absy), axis = 1)
    flatlabel = None; flatarea = None; flatcencoords = None; flatcoords = None; absx = None; absy = None

    collection = pd.DataFrame(objs, columns=[
        'Timepoint', 'Reg_Props_Obj_Num', 'Area', 'Major_axis_length', 'Y_intensity', 'X_intensity',
        'Y_center', 'X_center', 'Y_adj', 'X_adj'])
    collection['Timepoint'] = (collection['Timepoint'] * time_intv).astype(int)
    collection['Reg_Props_Obj_Num'] = collection['Reg_Props_Obj_Num'].astype(int)
    collection['Distance_polarity_score'] = ((collection['X_adj'] ** 2 + collection['Y_adj'] ** 2) ** 0.5) / (collection['Major_axis_length'] / 2)
    objs = None

    polarity_scores_final = []
    assert len(collection.columns) == 11
    for x, frame in enumerate(img_labels):
        pointslist = []; weightedpointlist = []; obj_intensitylist = []

        # find all x-y positions where there is an object present
        for index, item in np.ndenumerate(frame):
            if item > 0:
                nextpoint = [item, index]
                pointslist.append(nextpoint)

        pointslist.sort()
        subcollection = (collection[collection['Timepoint'] == (x * time_intv)]).values

        # find the total intensity of each object in the current image frame
        z = 1
        while z <= np.amax(frame):
            obj_intensity = ndimage.sum(fluor_chan[x,:,:], img_labels[x,:,:], index=z)
            obj_intensitylist.append(obj_intensity)
            z += 1

        # for each point in object, find the consine between its vector and the "polarity" vector
        for y, item in enumerate(pointslist):
            objnum = item[0]; xypos = item[1]
            center = (subcollection[(objnum - 1),6], subcollection[(objnum - 1),7])
            fluorcenter = (subcollection[(objnum - 1),4], subcollection[(objnum - 1),5])
            adjxypoint = np.subtract(xypos, center)
            adjxyfluor = np.subtract(fluorcenter, center)
            cosine = getcosine(adjxypoint, adjxyfluor)
            pointintensity = fluor_chan[x,xypos[0], xypos[1]]
            weightedpoint = cosine * pointintensity
            weightedpointlist.append(weightedpoint)    

        weightedpointlist = np.asanyarray(weightedpointlist).astype(int)
        sumweightedpoints = 0
        finalweights = []

        # this sums together the values for all the individual points of a given object
        for y, item in enumerate(weightedpointlist):
            if y + 1 == len(weightedpointlist):
                sumweightedpoints = sumweightedpoints + weightedpointlist[y]
                finalweights.append(sumweightedpoints)
                sumweightedpoints = 0
            elif pointslist[y][0] - pointslist[y + 1][0] == 0:
                sumweightedpoints = sumweightedpoints + weightedpointlist[y]
            elif pointslist[y][0] - pointslist[y + 1][0] == -1:
                sumweightedpoints = sumweightedpoints + weightedpointlist[y]
                finalweights.append(sumweightedpoints)
                sumweightedpoints = 0

        polarity_scores = np.asanyarray(finalweights) / np.asarray(obj_intensitylist)
        polarity_scores_final.append(list(polarity_scores))

    polarity_scores_final = mpu.datastructures.flatten(polarity_scores_final)
    polarity_scores_final = np.transpose(np.asarray(polarity_scores_final))
    collection['Angular_polarity_score'] = polarity_scores_final

    # Below for loop matches values from the 'polarity scores array' to those in the DataFrame
    # containing the CellProfiler tracks. This is needed since polarity scores are calculated for
    # every object, even ones that are not present in all timepoints, merge, or split.
    xy_coords = np.zeros((len(collection), len(collection.columns)), dtype=float)
    for indx, row in cell_tracks.iterrows():
        time_idx = row['Time_s']
        x_idx = row['X']
        lookup = (collection['Timepoint'] == time_idx) & (abs(collection['X_center'] - x_idx) < 0.001)
        extract = collection[lookup]
        # Below 'if' statement for catching cases where => 2 rows have very similar x-coords.
        # If true, also use y-coords for further discrimination
        if extract.shape[0] > 1:
            extract = None; lookup = None
            y_idx = row['Y']
            lookup = (collection['Timepoint'] == time_idx) & (
                abs(collection['X_center'] - x_idx) < 0.001) & (abs(collection['Y_center'] - y_idx) < 0.001)
            extract = collection[lookup]
        extract = extract.values
        xy_coords[indx,:] = extract

    # check these columns labels VERY CAREFULLY with the 'xy_coords' numpy array!
    new_coords = pd.DataFrame({'X_intensity_center':xy_coords[:,5], 'Y_intensity_center':xy_coords[:,4], 'X_object_center':xy_coords[:,7],
                               'Y_object_center':xy_coords[:,6], 'Distance_polarity_score':xy_coords[:,10], 'Angular_polarity_score':xy_coords[:,11]})
    cell_polarity_scores = cell_tracks.join(new_coords)
    output = pd.DataFrame(columns=[])
    output['Object_id'] = cell_polarity_scores['Object_id'].astype(int)
    output['Time_s'] = cell_polarity_scores['Time_s']
    output['Distance_polarity_score'] = cell_polarity_scores['Distance_polarity_score']
    output['Angular_polarity_score'] = cell_polarity_scores['Angular_polarity_score']
        
    return output