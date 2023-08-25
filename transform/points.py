import numpy as np

def data_to_array(points, normals):
    point_list = np.empty((len(points), 3))
    normal_list = np.empty((len(normals), 3))
    
    for i, (point, normal) in enumerate(zip(points, normals)):
        if point is None:
            point_list[i, :] = [np.nan, np.nan, np.nan]
            normal_list[i, :] = [np.nan, np.nan, np.nan]
        else:
            point_list[i, :] = [point.X, point.Y, point.Z]
            normal_list[i, :] = [normal.X, normal.Y, normal.Z]
    
    irradiance = np.repeat([0.0], len(point_list)).T
    array = np.concatenate((point_list, normal_list), axis=1)
    array = np.column_stack((array, irradiance))
    
    return array