import Rhino.Geometry as rg
import System
import json

def polyline_to_points(polyline):
    return [(point.X, point.Y, point.Z) for point in polyline]

def points_to_polyline(points):
    points = System.Array[rg.Point3d](points)
    return rg.Polyline(points)

def serialize(data):
    # data = [[rg.Polyline()], [...]]
        
    for i, outlines in enumerate(data):
        for j, polyline in enumerate(outlines):
            data[i][j] = polyline_to_points(polyline)
    
    return json.dumps(data)

def deserialize(data):
    data = json.loads(data)
    for i, outlines in enumerate(data):
        for j, polyline in enumerate(outlines):
            points = [rg.Point3d(point[0],point[1],point[2]) for point in polyline]
            data[i][j] = points_to_polyline(points)

    return data
    