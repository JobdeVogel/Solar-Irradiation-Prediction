import Rhino.Geometry as rg
import System
import json

def polyline_to_points(polyline):
    # Transform a polyline to points as tuples
    return [(point.X, point.Y, point.Z) for point in polyline]

def points_to_polyline(points):
    # Transform points as tuples to polylines
    points = System.Array[rg.Point3d](points)
    return rg.Polyline(points)

def serialize(data):
    """Serialize polyline data assuming the structure [[rg.Polyline], [...]]
    Serializing can be used to make sure data is not changed in-place

    Args:
        data (list[list[rg.Polyline]]): polylines to serialize

    Returns:
        json string (str): serialized polylines in JSON format 
    """
    
    # Iterate over all polylines in the data
    for i, outlines in enumerate(data):
        for j, polyline in enumerate(outlines):
            # Transform the polyline data too points
            data[i][j] = polyline_to_points(polyline)
    
    return json.dumps(data)

def deserialize(data):
    """Deserialize polyline data assuming the structure [[rg.Polyline], [...]]

    Args:
        data (JSON str): JSON string containing polyline data

    Returns:
        data (list[list[rg.Polyline]]): polylines to serializee
    """
    
    # Load the JSON data
    data = json.loads(data)
    for i, outlines in enumerate(data):
        for j, polyline in enumerate(outlines):
            # Extract the points for the outline
            points = [rg.Point3d(point[0],point[1],point[2]) for point in polyline]
            
            # Transform the points to an outline
            data[i][j] = points_to_polyline(points)

    return data