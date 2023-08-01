def set_array_values(array, points=None, normals=None, irradiance=None, pointmap=None):
    if points == None and normals == None and pointmap == None:
        array[:, 6] = irradiance
    elif pointmap == None:
        if points != None: 
            array[:,0:3] = [[point.X, point.Y, point.Z] for point in points]
        if normals != None:
            array[:,3:6] = [[normal.X, normal.Y, normal.Z] for normal in points]
        if irradiance != None:
            array[:, 6] = irradiance
    else:
        if points != None:
            points = iter(points)
        if normals != None:
            normals = iter(normals)
        
        for i, (bool, value) in enumerate(zip(pointmap, irradiance)):
            if bool:
                if points != None:
                    point = next(points)
                    array[i, 0:3] = [point.X, point.Y, point.Z]
                if normals != None:
                    normal = next(normals) 
                    array[i, 3:6] = [normal.X, normal.Y, normal.Z]
                if irradiance != None:
                    array[i, 6] = value

    return array
        