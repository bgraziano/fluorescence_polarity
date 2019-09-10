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

def fluor_polarity(fluor_chan, bmask, cell_tracks):
    assert type(bmask) is np.ndarray, "Binary masks are not a numpy array!"
    assert type(fluor_chan) is np.ndarray, "Fluorescence images are not a numpy array!"
    assert fluor_chan.shape == bmask.shape, "Fluorescence image and binary mask are different dimensions!"
    assert type(cell_tracks) is pd.DataFrame, "'cell tracks' need to be formatted as a pandas DataFrame!"
    assert 'Time_s' in cell_tracks, "'cell_tracks' is missing 'Time_s' column!"
    assert 'Cell_id' in cell_tracks, "'cell_tracks' is missing 'Cell_id' column!"
    assert 'X' in cell_tracks and 'Y' in cell_tracks, "'cell_tracks' is missing 'X' and/or 'Y' column(s)!"    
           
    time_intv = cell_tracks.loc[1, 'Time_s'] - cell_tracks.loc[0, 'Time_s'] # for determining time interval between each frame
    final_table = pd.DataFrame(columns=[])    
    img_labels = np.empty(bmask.shape, dtype=int)

    for x, frame in enumerate(bmask):
        img_labels[x] = sk.measure.label(frame)       

    areascol = []; labelscol = []; objs = []; intlocscol = []; cenlocscol = []; xlist = []

    # label objects in binary mask and get x-y positions for the geometric center
    # and weighted fluorescence intensity center for each labeled object
    for x, frame in enumerate(img_labels):
        areas = [r.area for r in sk.measure.regionprops(frame)]
        labels = [r.label for r in sk.measure.regionprops(frame)]
        intlocs = [list(r.weighted_centroid) for r in sk.measure.regionprops(
            frame, intensity_image=fluor_chan[x,:,:])]
        cenlocs = [list(r.weighted_centroid) for r in sk.measure.regionprops(
            frame, intensity_image=bmask[x,:,:])]
        areascol.append(areas); labelscol.append(labels); intlocscol.append(intlocs); cenlocscol.append(cenlocs)
        y = 0
        while y < np.amax(frame): # dumb way to mark all labeled objects at a given timepoint
            xlist.append(x)
            y += 1

    # make a numpy array from all the lists generated in preceding 'for' loop
    flatarea = mpu.datastructures.flatten(areascol)
    flatlabel = mpu.datastructures.flatten(labelscol)
    flatcoords = mpu.datastructures.flatten(intlocscol)
    flatcoords = np.reshape(np.asarray(flatcoords), (len(flatcoords)//2, 2))
    flatcencoords = mpu.datastructures.flatten(cenlocscol)
    flatcencoords = np.reshape(np.asarray(flatcencoords), (len(flatcencoords)//2, 2))
    objs.append(xlist); objs.append(flatlabel); objs.append(flatarea)
    objs = np.transpose(np.asarray(objs))
    objs = np.concatenate((objs,flatcoords, flatcencoords),axis = 1)

    # normalize distances between geometric and fluorescence center to origin to make later calcuations easier
    absx = objs[:,3] - objs[:,5]
    absy = objs[:,4] - objs[:,6]
    absx = np.reshape(np.asarray(absx), (len(absx), 1))
    absy = np.reshape(np.asarray(absy), (len(absy), 1))

    objs = np.concatenate((objs, absx, absy), axis = 1)
    flatlabel = None; flatarea = None; flatcencoords = None; flatcoords = None; absx = None; absy = None

    collection = pd.DataFrame(objs, columns=[
        'Timepoint', 'Reg_Props_Obj_Num', 'Area', 'Y_intensity', 'X_intensity',
        'Y_center', 'X_center', 'Y_adj', 'X_adj'])
    collection['Timepoint'] = (collection['Timepoint'] * time_intv).astype(int)
    collection['Reg_Props_Obj_Num'] = collection['Reg_Props_Obj_Num'].astype(int)
    objs = None

    polarity_scores_final = []
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
            center = (subcollection[(objnum - 1),5], subcollection[(objnum - 1),6])
            fluorcenter = (subcollection[(objnum - 1),3], subcollection[(objnum - 1),4])
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
    collection['Polarity_scores'] = polarity_scores_final

    # Below for loop matches values from the 'polarity scores array' to those in the DataFrame
    # containing the CellProfiler tracks. This is needed since polarity scores are calculated for
    # every object, even ones that are not present in all timepoints, merge, or split.
    xy_coords = np.zeros((len(collection), 8), dtype=float)
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
        extract.drop(columns = ['Timepoint', 'Area'], inplace=True)
        extract = extract.values
        xy_coords[indx,:] = extract

    new_coords = pd.DataFrame({'X_intensity_center':xy_coords[:,2], 'Y_intensity_center':xy_coords[:,1], 'X_object_center':xy_coords[:,4],
                               'Y_object_center':xy_coords[:,3], 'Angular_polarity_score':xy_coords[:,7]})
    cell_polarity_scores = cell_tracks.join(new_coords)
    output = pd.DataFrame(columns=['Cell_id', 'Time_s', 'Angular_polarity_score'])
    output['Cell_id'] = cell_polarity_scores['Cell_id'].astype(int)
    output['Time_s'] = cell_polarity_scores['Time_s']
    output['Angular_polarity_score'] = cell_polarity_scores['Angular_polarity_score']
        
    return output