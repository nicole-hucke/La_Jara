def get_hydraulic_radius(X, Y, WSE):
    wet_areas = []
    wet_perimeters = []
    poly_x = []
    poly_y = []

    # Loop through the points of the cross-section to find wet polygon
    for i in range(len(X) - 1):
        y1, y2 = Y[i], Y[i+1]
        x1, x2 = X[i], X[i+1]

        # First, checking if we enter the water
        if (y1 >= WSE and y2 < WSE):
            # Linear interpolation to find the intersection point
            x_int1 = x1 + (WSE - y1) * (x2 - x1) / (y2 - y1)
            poly_x.append(x_int1)
            poly_y.append(WSE)
        
        # If not, we check if the first point is below the WSE
        if (y1 < WSE):
            poly_x.append(x1)
            poly_y.append(y1)
        
        # Then, we check if we go out of the water. If we did, we calculate the area and store it
        if (y1 < WSE and y2 >= WSE):
            # Linear interpolation to find the intersection point
            x_int2 = x1 + (WSE - y1) * (x2 - x1) / (y2 - y1)
            poly_x.append(x_int2)
            poly_y.append(WSE)

            # Closing the polygon
            if len(poly_x) > 0:
                poly_x.append(poly_x[0])
                poly_y.append(poly_y[0])
            
            # Calculating the area
            tempCalc1 = 0.5 * np.abs(np.dot(poly_x, np.roll(poly_y, 1)) - np.dot(poly_y, np.roll(poly_x, 1)))
            wet_areas.append(tempCalc1)

            # Calculating the wet perimeter
            distances = np.sqrt(np.diff(poly_x)**2 + np.diff(poly_y)**2)
            wet_perimeters.append(np.sum(distances[:-1]))

            # Resetting the polygon
            poly_x = []
            poly_y = []
    
    # return hydraulic radius, wet area, wet perimeter
    tempCalc1 = np.sum(wet_areas)
    wet_area = tempCalc1
    tempCalc2 = np.sum(wet_perimeters)
    wet_perimeter = tempCalc2
    # check if tempCalc2 is zero to avoid division by zero
    if tempCalc2 == 0:
        hydraulic_radius = float('nan')  
    else:
        hydraulic_radius = tempCalc1 / tempCalc2
    
    return hydraulic_radius, wet_area, wet_perimeter

def calculate_hydraulic_radius(topography_df, depth_timeseries, depth_location):
    X = topography_df['distance'].values
    Z = topography_df['elevation'].values
    depth = depth_timeseries['depth'].values
    times = depth_timeseries.index 

    #find the riverbed elevation at the depth measurement location
    riverbed_elevation_at_depth_location = np.interp(depth_location, X, Z)

    hyd_rad = [] 
    wet_area = []
    wet_perimeter = []

    # loop over each time step and calculate hydraulic radius
    for d in depth:
        wse = riverbed_elevation_at_depth_location + d  # wse at this time step
        hydraulic_radius_calc, wet_area_calc, wet_perimeter_calc = get_hydraulic_radius(X, Z, wse)  # calculate hydraulic radius
        hyd_rad.append(hydraulic_radius_calc)
        wet_area.append(wet_area_calc)
        wet_perimeter.append(wet_perimeter_calc)

    # create a DataFrame with time and hydraulic radius for each time step
    R_timeseries = pd.DataFrame({
        'time': times,
        'hydraulic_radius': hyd_rad,
        'wet_area': wet_area,
        'wet_perimeter': wet_perimeter
    }).set_index('time')
    return R_timeseries, 