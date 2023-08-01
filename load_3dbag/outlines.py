from parameters.params import LOGGER
import Rhino.Geometry as rg
import System
import sys

from parameters.params import TRANSLATE_TO_ORIGIN, FSI, _SPLIT_TOLERANCE, MIN_AREA

# Get the x,y domain of a bbox
def get_domain(bbox):
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
    min_x, max_x = domain[0]
    min_y, max_y = domain[1]
    
    x_size = abs(min_x) + abs(max_x)
    y_size = abs(min_y) + abs(max_y)
    
    step_size = size * min_coverage / 100
    x_steps = (x_size // step_size) - 1
    y_steps = (y_size // step_size) - 1
    
    sub_domains = []
    for i in range(int(x_steps)):
        for j in range(int(y_steps)):
            x_domain = (min_x + i * step_size, min_x + i * step_size + size)
            y_domain = (min_y + j * step_size, min_y + j * step_size + size)
            
            sub_domains.append((x_domain, y_domain))
    
    return sub_domains

# Generate bbox from domain and size
def create_bbox(domain, bottom_treshold):
    x_domain, y_domain = domain
    x_center = (x_domain[0] + x_domain[1]) / 2
    y_center = (y_domain[0] + y_domain[1]) / 2
    
    size = abs(x_domain[1] - x_domain[0])
    
    center = rg.Point3d(x_center, y_center, 0)
    plane = rg.Plane(center, rg.Vector3d(0, 0, 1))
    
    interv = rg.Interval(-0.5 * size, 0.5 * size)
    z_interv = rg.Interval(-0.1, size + 0.1)

    box = rg.Box(plane, interv, interv, z_interv)
    return box

def generate_outlines_from_bbox(bbox, size, min_coverage):
    main_domain = get_domain(bbox)
    sub_domains = divide_domain(main_domain, size, min_coverage)
    
    outlines = []
    for domain in sub_domains:
        corner_0 = rg.Point3d(domain[0][0], domain[1][0], 0)
        corner_1 = rg.Point3d(domain[0][1], domain[1][1], 0)
        outlines.append(rg.Rectangle3d(rg.Plane.WorldXY, corner_0, corner_1))

    return outlines






def includes_ground_outline(ground_outline, building_outline):
    outside = False
    inside = False
    
    relation = rg.PointContainment.Unset
    
    curve = ground_outline.ToNurbsCurve()
    
    for point in building_outline:
        relation = curve.Contains(point)
        
        if relation == rg.PointContainment.Inside:
            inside = True
        elif relation == rg.PointContainment.Outside:
            outside = True
        
        if outside and inside:
            return 0
    
    if outside and not inside:
        return -1
    elif inside and not outside:
        return 1
    else:
        return None

def extract_building_outlines(wall_meshes, roof_meshes, tolerance=_SPLIT_TOLERANCE):
    building_outlines = []
    building_heights = []
    
    for i, (wall, roof) in enumerate(zip(wall_meshes, roof_meshes)):
        outlines = []
        
        lines = wall.GetNakedEdges()
        
        for naked in lines:
            height = naked.CenterPoint().Z
            
            if height < tolerance:
                outlines.append(naked)
        
        for outline in outlines:
            outline.ReduceSegments(tolerance)
        
        if len(outlines) != 0:
            building_outlines.append(outlines)
            building_heights.append(roof.Faces.GetFaceCenter(0).Z)
        else:
            LOGGER.warning('Mesh ' + str(i) + ' does not have outline with height lower than tolerance ' + str(tolerance))
    
    return building_outlines, building_heights

# Find which polyline is the outer polyline of a mesh surface
def find_outer_polyline(polylines):
    # IMPROVE:
    # Rough assumption that outer polyline is always longest length
    lengths = []
    inner = []
    
    for polyline in polylines:
        lengths.append(polyline.Length)
        inner.append(polyline)
    
    idx = lengths.index(max(lengths))
    outer = [polylines[idx]]
    del inner[idx]
    
    return outer + inner

def find_segments(segments, segmented_curve, base_curve):
    """Find which segments from a segmented curve should be kept and which
    should be deleted.
    Inputs:
        segments: The segments as polylines
        segmented_curve: The The curve that was segmented
        base_curve: The curve used for containment
    Output:
        polylines: the segments that should be kept"""
        
    polylines = []
    for segment in segments:
        segment = segment.ToNurbsCurve()
        
        domain = segment.Domain
        midpoint_parameter = domain.Mid
        
        midpoint = segment.PointAt(midpoint_parameter)
        
        if base_curve.Contains(midpoint, rg.Plane.WorldXY) == rg.PointContainment.Inside:
            polylines.append(segment)
        elif base_curve.Contains(midpoint, rg.Plane.WorldXY) == rg.PointContainment.Coincident:
            polylines.append(segment)

    return polylines
    
def cut_polyline(ground_outline, building_outline, tolerance=_SPLIT_TOLERANCE, min_area=MIN_AREA):
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
    building_segments = find_segments(building_segments, building_curve, ground_curve)

    # Get the ground segment
    ground_polylines = [segment.TryGetPolyline()[1] for segment in ground_segments]
    ground_segments = find_segments(ground_polylines, ground_curve, building_curve)

    new_building_curves = System.Array[rg.NurbsCurve](building_segments + ground_segments)
    joined_building_curve = rg.Curve.JoinCurves(new_building_curves, tolerance)
    
    projections = [rg.Curve.ProjectToPlane(curve, rg.Plane.WorldXY) for curve in joined_building_curve]
    polylines = [curve.TryGetPolyline()[1] for curve in projections]
    
    if len(polylines) == 0:
        LOGGER.warning("cut_polyline() was not able to extract polylines")
    
    # ! NOT IMPLEMENTED YET
    # valid_polylines = []
    # for polyline in polylines:
    #     area = rg.AreaMassProperties.Compute(polyline.ToNurbsCurve()).Area
        
    #     if area > min_area:
    #         valid_polylines.append(polyline)
            
    return polylines

# Translate objects to origin
def translate(outline, ground_outline, height=0):
    # Compute center of bbox
    center = ground_outline.Center
    x, y = center.X, center.Y
    translation = rg.Transform.Translation(-x, -y, height)
    outline.Transform(translation)

def compute_FSI(ground_outline, building_outlines):
    ground_area = rg.AreaMassProperties.Compute(ground_outline.ToNurbsCurve()).Area
    
    building_areas = []
    for outlines in building_outlines:
        if len(outlines) > 0:
            try:            
                building_areas.append(
                    rg.AreaMassProperties.Compute(outlines[0].ToNurbsCurve()).Area
                    )
            except:
                LOGGER.warning("RESOLVE: Polyline was not closed so area not added to FSI")
    
    return sum(building_areas) / ground_area, ground_area, building_areas
    
def generate_building_outlines(ground_outline, all_building_outlines, heights, min_area, translate_to_origin=TRANSLATE_TO_ORIGIN, fsi=FSI):    
    # Compute the polylines inside the ground_outline
    included_polylines = []
    courtyards = []
    building_heights = []
    num_of_courtyards = 0
    
    for i, (polylines, height) in enumerate(zip(all_building_outlines, heights)):
        # The outer polyline of the building
        outer = polylines[0]
        
        # There is only one polyline, so there is no courtyard
        if len(polylines) == 1:
            # Check if outer polyline inside the ground_outline
            relation = includes_ground_outline(ground_outline, outer)
            
            # Inside
            if relation == 1:
                included_polylines.append([outer])
                building_heights.append(height)
                
                # No courtyards are available, so add empty list
                courtyards.append([])
            
            # Intersectin
            elif relation == 0:
                # Cut the polyline and add it to included_polylines
                polylines = cut_polyline(ground_outline, outer)
                               
                # ! Check if this works!!!
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
                # elif relation == 0:
                #     polylines = cut_polyline(ground_outline, inner)
                #     temp_courtyard_polylines += polylines
            
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
        
        translate(ground_outline, ground_outline)
    
    
    if fsi:
        FSI_score, envelope_area, building_area = compute_FSI(ground_outline, included_polylines)
    else:
        FSI_score = None
        envelope_area = None
        building_area = None

    building_outlines = included_polylines
    courtyard_outlines = courtyards

    LOGGER.debug(f'Generated {len(building_outlines)} building_outlines and {num_of_courtyards} courtyards with FSI {round(FSI_score, 2)}.')
    return building_outlines, courtyard_outlines, building_heights, FSI_score, envelope_area, building_area

# ground_outline = ground_outlines[int(idx)]
# building_outlines, courtyard_outlines, heights, FSI, envelope_area, building_area = main(ground_outlines, wall_meshes, heights, size, idx, FSI=FSI)

# courtyard_outlines = th.list_to_tree(courtyard_outlines, source=[0])
# building_outlines = th.list_to_tree(building_outlines, source=[0])