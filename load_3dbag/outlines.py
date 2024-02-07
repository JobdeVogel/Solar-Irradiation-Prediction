"""
This module contains all functions regarding flat polylines, curves and rectangles
"""

import Rhino.Geometry as rg
import System
import sys

from parameters.params import TRANSLATE_TO_ORIGIN, FSI, _SPLIT_TOLERANCE, MIN_AREA

# Get the x,y domain of a bbox
def get_domain(bbox):
    """Get the x and y domain from a boundingbox

    Args:
        bbox (rg.BoundingBox): boundingbox

    Returns:
        domains: two tuples containing minima and maxima for x and y
    """
    x = []
    y = []
    
    for point in bbox.GetCorners():
        x.append(point.X)
        y.append(point.Y)
    
    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)
    
    return (min_x, max_x), (min_y, max_y)

# Divide main domain into sub domains
def divide_domain(domain, size, min_coverage):
    """Divide a domain in subdomains

    Args:
        domain (tuple): domain in shape min, max
        size (float): size of the subdomains height and width
        min_coverage (float): minimum overlap between subdomains

    Returns:
        sub_domains (list[tuple]): divided domains
    """
    # Extract the minimum and maximum from the original domain
    min_x, max_x = domain[0]
    min_y, max_y = domain[1]
    
    # Compute the distance between minima and maxima for the domain
    x_size = abs(min_x) + abs(max_x)
    y_size = abs(min_y) + abs(max_y)
    
    # Compute the step size
    step_size = size * min_coverage / 100
    
    # Compute number of steps in x and y direction
    x_steps = (x_size // step_size)
    y_steps = (y_size // step_size)
    
    # Store the new subdivide domains
    sub_domains = []
    for i in range(int(x_steps)):
        for j in range(int(y_steps)):
            # Compute a new domain
            x_domain = (min_x + i * step_size, min_x + i * step_size + size)
            y_domain = (min_y + j * step_size, min_y + j * step_size + size)
            
            # Append the new domain to the returned domains
            sub_domains.append((x_domain, y_domain))
    
    return sub_domains

def generate_outlines_from_bbox(bbox, size, min_coverage):
    """Generate a set of patch outlines that are used to indicate which meshes
    should be extracted per sample

    Args:
        bbox (rg.BoundingBox): boundingbox of the entire mesh
        size (float): width and height of the patches
        min_coverage (float): factor indicating how much area may be repeated in sequential patches

    Returns:
        outlines (list[rg.Rectangle3d]): oulines of bbox as rectangles 
    """
    
    #Get the domain of a boundingbox
    main_domain = get_domain(bbox)
    
    #Divide the main domain to subdomains
    sub_domains = divide_domain(main_domain, size, min_coverage)
    
    #Generate a rectangle based on the subdomains
    outlines = []
    for domain in sub_domains:
        corner_0 = rg.Point3d(domain[0][0], domain[1][0], 0)
        corner_1 = rg.Point3d(domain[0][1], domain[1][1], 0)
        outlines.append(rg.Rectangle3d(rg.Plane.WorldXY, corner_0, corner_1))

    return outlines

def includes_ground_outline(ground_outline, building_outline):
    """Check if a building outline is inside a ground outline rectangle

    Args:
        ground_outline (rg.Rectange3d): Rectangle outline for the ground patch
        building_outline (rg.Polyline): Polyline outline for a building

    Returns:
        int: -1 if outside, 1 if inside, 0 i intersecting, else None
    """
    # Boolean indicating if points are outside and inside
    outside = False
    inside = False
    
    # Base relation is unset
    relation = rg.PointContainment.Unset
    
    # Transform the ground outline polyline to Nurbscurve
    curve = ground_outline.ToNurbsCurve()
    
    # For each point in the polyline
    for point in building_outline:
        # Compute the relation with the ground outline
        relation = curve.Contains(point)
        
        # Set the relations
        if relation == rg.PointContainment.Inside:
            inside = True
        elif relation == rg.PointContainment.Outside:
            outside = True
        
        if outside and inside:
            return 0
    
    # If not intersecting, check the new relation
    if outside and not inside:
        return -1
    elif inside and not outside:
        return 1
    else:
        return None

def extract_building_outlines(wall_meshes, roof_meshes, tolerance=_SPLIT_TOLERANCE, logger=False):
    """Extract polyline building outlines from wall and roof meshes

    Args:
        wall_meshes (list[rg.Mesh]): list of wall meshes
        roof_meshes (list[rg.Mesh]): list of roof meshes
        tolerance (float, optional): Tolerance for adding naked edges to the outlines list. Defaults to _SPLIT_TOLERANCE.

    Returns:
        building_outlines (list[list[rg.Polyline]]): building outlines
        building_heights (list[float]): heights individual buildings
    """
    
    # Store the building outlines and heihgts
    building_outlines = []
    building_heights = []
    
    # Iterate over both walls and roofs
    for i, (wall, roof) in enumerate(zip(wall_meshes, roof_meshes)):
        # Store temporary outlines for a specific building
        outlines = []
        
        # Get the naked edges from the wall
        lines = wall.GetNakedEdges()
        
        # If the extractionn succeeded
        if lines != None:       
            # Iterate over the naked lines
            for naked in lines:
                # Check the height of the naked line
                height = naked.CenterPoint().Z
                
                # If the height is lower than the tolerance, add the naked line to the outlines
                if height < tolerance:
                    outlines.append(naked)
            
            # Reduce the number of segments in the outlines
            for outline in outlines:
                outline.ReduceSegments(tolerance)
            
            # Check if the building is not floating above the ground
            if len(outlines) != 0:
                # If not, append the outlines and the building height
                building_outlines.append(outlines)
                try:
                    building_heights.append(roof.Faces.GetFaceCenter(0).Z)
                except:
                    building_heights.append(max([vertex.Z for vertex in wall.Vertices]))
            else:
                if logger:
                    print('Mesh ' + str(i) + ' does not have naked edges with height lower than tolerance ' + str(tolerance) + '. This building if floating above the ground.')
        else:
            if logger:
                print('Mesh ' + str(i) + ' naked edges extraction failed. This mesh is most likely closed.')
    
    return building_outlines, building_heights

# Find which polyline is the outer polyline of a mesh surface
def find_outer_polyline(polylines):
    """Given multiple polylines, find which polyline has the longest length, and is most likely  the outer polyline
    
    # ! IMPROVE: This assumption is theoratically not always correct

    Args:
        polylines (list[rg.Polyline]): list of polylines

    Returns:
        outer + inner (list[rg.Polyline]): ordered list of polylines
    """
    # Store the lengths of the polylines and which polylines are inner polylines
    outers = []
    inners = []
    
    curves = [polyline.ToNurbsCurve() for polyline in polylines]
    
    for inner in polylines:
        
        inside_any = False
        for outer in curves:
            is_inside = []
            for point in inner:
                
                if outer.Contains(point, rg.Plane.WorldXY) == rg.PointContainment.Inside:
                    is_inside.append(True)
                else:
                    is_inside.append(False)
         
            if all(is_inside):
                # The inner is inside this outside
                inside_any = True
                break
        if inside_any:
            inners.append(inner)
        else:
            outers.append(inner)

    return outers, inners

def find_segments(segments, base_curve):
    """Find which segments from a segmented curve should be kept and which
    should be deleted.
    Inputs:
        segments: The segments as polylines
        base_curve: The curve used for containment
    Output:
        polylines: the segments that should be kept"""
    
    #Store the polylines that should be kept    
    polylines = []
    for segment in segments:
        #Transform the segment from rg.Polyline to rg.NurbsCurve
        segment = segment.ToNurbsCurve()
        
        #Get 
        domain = segment.Domain
        midpoint_parameter = domain.Mid
        
        midpoint = segment.PointAt(midpoint_parameter)
        
        if base_curve.Contains(midpoint, rg.Plane.WorldXY) == rg.PointContainment.Inside:
            polylines.append(segment)
        elif base_curve.Contains(midpoint, rg.Plane.WorldXY) == rg.PointContainment.Coincident:
            polylines.append(segment)

    return polylines
    
def cut_polyline(ground_outline, building_outline, tolerance=_SPLIT_TOLERANCE, min_area=MIN_AREA, logger=False):
    # Transform polylines to curves
    ground_curve = ground_outline.ToNurbsCurve()
    building_curve = building_outline.ToNurbsCurve()
    
    # Compute intersection between building and ground outline curve
    intersection_events = rg.Intersect.Intersection.CurveCurve(building_curve, ground_curve, tolerance, tolerance)
    
    # Store the parameter lengths for building and ground curve
    parameters_ground = []
    parameters_building = []
    for event in intersection_events:
        if event.IsPoint:
            
            # Compute the points for the building intersection
            parameter_building = building_curve.ClosestPoint(event.PointA)[1]
            parameters_building.append(parameter_building)

            # Compute the points for the ground intersection
            parameter_ground = ground_curve.ClosestPoint(event.PointB)[1]
            parameters_ground.append(parameter_ground)
    
    parameters_building = System.Array[System.Double](parameters_building)
    parameters_ground = System.Array[System.Double](parameters_ground)
    
    # Compute the segments for the ground and building outlines
    ground_segments = ground_curve.Split(parameters_ground)
    building_segments = building_curve.Split(parameters_building)
    
    # Get the inner building segment
    building_segments = find_segments(building_segments, ground_curve)

    # Get the ground segment
    ground_polylines = [segment.TryGetPolyline()[1] for segment in ground_segments]
    ground_segments = find_segments(ground_polylines, building_curve)

    new_building_curves = System.Array[rg.NurbsCurve](building_segments + ground_segments)
    joined_building_curve = rg.Curve.JoinCurves(new_building_curves, tolerance)
    
    projections = [rg.Curve.ProjectToPlane(curve, rg.Plane.WorldXY) for curve in joined_building_curve]
    polylines = [curve.TryGetPolyline()[1] for curve in projections]
    
    if len(polylines) == 0:
        if logger:
            print("cut_polyline() was not able to extract polylines")
    
    valid_polylines = []
    for polyline in polylines:
        template = polyline.Duplicate()
        
        try:
            if polyline.IsClosed:
                area = rg.AreaMassProperties.Compute(template.ToNurbsCurve()).Area       
            
                if area > min_area:
                    valid_polylines.append(polyline)
            else:
                points = [point for point in polyline]
                points += [points[0]]
                template = rg.Polyline(System.Array[rg.Point3d](points))
                
                area = rg.AreaMassProperties.Compute(template.ToNurbsCurve()).Area  
                
                if area > min_area:
                    valid_polylines.append(polyline)
        except AttributeError:
            # Area computation returns None
            valid_polylines.append(polyline)
                
    return valid_polylines

# Translate objects to origin
def translate(outline, ground_outline, height=0):
    # Compute center of bbox
    center = ground_outline.Center
    x, y = center.X, center.Y
    translation = rg.Transform.Translation(-x, -y, height)
    outline.Transform(translation)

def compute_FSI(ground_outline, building_outlines, logger=False):
    ground_area = rg.AreaMassProperties.Compute(ground_outline.ToNurbsCurve()).Area
    
    building_areas = []
    for outlines in building_outlines:
        if len(outlines) > 0:
            try:            
                building_areas.append(
                    rg.AreaMassProperties.Compute(outlines[0].ToNurbsCurve()).Area
                    )
            except:
                if logger:
                    print("RESOLVE: Polyline was not closed so area not added to FSI")
    
    return sum(building_areas) / ground_area, ground_area, building_areas

def fix_self_intersections(polyline):
    curve = polyline.ToPolylineCurve()
    intersections = rg.Intersect.Intersection.CurveSelf(curve, 0.001)
    
    if intersections > 0:
        intersect = True
    else:
        intersect = False
    
    parameters = []
    for event in intersections:
        event_parameters = System.Array[float]([event.ParameterA, event.ParameterB])
        
        parameters += event_parameters
    
    curve_segments = curve.Split(parameters)
    
    polyline_segments = []
    polyline_lengths = []
    for curve in curve_segments:
        polyline = curve.TryGetPolyline()[1]
        polyline_segments.append(polyline)
        polyline_lengths.append(polyline.Length)
    
    polylines_sorted = [pl for _, pl  in sorted(zip(polyline_lengths, polyline_segments))]
    
    if len(intersections) == 0:
        polyline = polyline
    if len(intersections) == 1:
        polyline = polylines_sorted[-1]
        
        # CHECK IF CLOSED!!!
    elif len(intersections) > 1:
        polylines_sorted_inv = polylines_sorted[::-1]
        
        curves = System.Array[rg.Curve](
            [polylines_sorted_inv[i].ToNurbsCurve() for i in range(0, len(intersections))]
            )
        
        polyline = rg.NurbsCurve.JoinCurves(curves)[0].TryGetPolyline()[1]
    
    return polyline, intersect


def generate_building_outlines(ground_outline, all_building_outlines, heights, translate_to_origin=TRANSLATE_TO_ORIGIN, fsi=FSI, logger=False):    
    """Compute the building outlines that are inside a ground outline. If a building polyline intersects
    with the ground outline, the building outlines are splitted in multiple segmenets and then closed.

    Args:
        ground_outline (rg.Rectangle3d): ground rectangle outline
        all_building_outlines (list[list[rg,Polyline]]): all building outlines
        heights (listt[float]): all heights of the buildings
        translate_to_origin (bool, optional): indicates if outlines should be moved to origin. Defaults to TRANSLATE_TO_ORIGIN.
        fsi (bool, optional): Indicates if fsi should be computed. Defaults to FSI.

    Returns:
        building_outlines (list[list[rg.Polyline]]): outlines for the buildings (including splitted intersecting building outlines)
        courtyard_outlines (list[list[rg.Polyline]]): outlines for the courtyards, empty if no courtyard available
        building_heights (list[float]): heights of the included buildings
        FSI_score (float): fsi score, None if not computed
        envelope_area (float): area of envelope in m2, None if not computed 
        building_area (float): area of all building, courtyards included in m2, None if not computed
    """
    
    # Store the polylines and heights
    included_polylines = []
    courtyards = []
    building_heights = []
    num_of_courtyards = 0
    
    # Iterate over all outlines and the corresponding heights
    for polylines, height in zip(all_building_outlines, heights):
        # The outer polyline of the building
        outer = polylines[0]
        
        # There is only one polyline, so there is no courtyard
        if len(polylines) == 1:
            # Check if outer polyline inside the ground_outline
            relation = includes_ground_outline(ground_outline, outer)
            
            # This building is inside the ground outline
            if relation == 1:
                included_polylines.append([outer])
                building_heights.append(height)
                
                # No courtyards are available, so add empty list
                courtyards.append([])
            
            # This building is intersecting with the ground outline
            elif relation == 0:
                # Cut the polyline and add it to included_polylines
                polylines = cut_polyline(ground_outline, outer)
                               
                if len(polylines) > 0:                   
                    # if smallest_area > min_area:
                    included_polylines.append(polylines)
                    building_heights.append(height)
            
                    # No courtyards are available, so add empty list
                    courtyards.append([])
        
        # There are polylines, so there is at least one courtyard
        else:
            # First find the courtyards
            temp_courtyard_polylines = []
            
            # Check if the courtyard outlines are inside the ground outline
            for inner in polylines[1:]:
                # These are the couryards!
                relation = includes_ground_outline(ground_outline, inner)
                
                # Courtyards do NOT have to agree with minimum area
                if relation == 1:
                    temp_courtyard_polylines.append(inner)
                    num_of_courtyards += 1
                    
            # Check if the outer polyline is inside the groundline
            relation = includes_ground_outline(ground_outline, outer)
            
            # The outer polyline is inside the ground polyline
            if relation == 1:
                included_polylines.append([outer])
                building_heights.append(height)
                
                # Outer was inside ground_outline, so all courtyards too
                courtyards.append(temp_courtyard_polylines)
            
            # The outer polyline intersects the ground polyline
            elif relation == 0:
                polylines = cut_polyline(ground_outline, outer)
                included_polylines.append(polylines)
                
                building_heights.append(height)
                
                # Check if any courtyard_outlines are inside the ground outline
                if len(temp_courtyard_polylines) > 0:
                    courtyards.append(temp_courtyard_polylines)
                else:
                    # Add an empty list
                    courtyards.append([])
     
    # Translate all outlines to the origin
    if translate_to_origin:
        for outlines in included_polylines:
            for polyline in outlines:
                translate(polyline, ground_outline)
        
        for outlines in courtyards:
            for outline in outlines:
                translate(outline, ground_outline)        
        
        # Translate the ground outline to the origin    
        translate(ground_outline, ground_outline)
    
    if fsi:
        FSI_score, envelope_area, building_area = compute_FSI(ground_outline, included_polylines)
    else:
        FSI_score = None
        envelope_area = None
        building_area = None

    building_outlines = included_polylines
    courtyard_outlines = courtyards

    if logger:
        logger.debug(f'Generated {len(building_outlines)} building_outlines and {num_of_courtyards} courtyards with FSI {round(FSI_score, 2)}.')
    
    building_curves = []
    for outlines in building_outlines:
        building_curves.append([out.ToNurbsCurve() for out in outlines])
    
    # Sometimes a building outline is accidently a courtyard, this is fixed here:
    pops = []
    # Iterate over potential courtyards
    for i, potential_courtyards in enumerate(building_outlines):
        for potential_courtyard in potential_courtyards:
            
            # Iterate over buildings
            for j, curves in enumerate(building_curves):
                for curve in curves:
                    inside = []
                    
                    for point in potential_courtyard:
                        if curve.Contains(point, rg.Plane.WorldXY) == rg.PointContainment.Inside:
                            inside.append(True)
                        else:
                            #This is not a potential courtyard for this building
                            inside.append(False)
                            break
                    
                    if all(inside):
                        courtyard_outlines[j].append(potential_courtyard)
                        pops.append(i)
    
    building_outlines = [i for j, i in enumerate(building_outlines) if j not in pops]
    courtyard_outlines = [i for j, i in enumerate(courtyard_outlines) if j not in pops]
    building_heights = [i for j, i in enumerate(building_heights) if j not in pops]
    
    return building_outlines, courtyard_outlines, building_heights, FSI_score, envelope_area, building_area