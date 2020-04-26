

class Feature:
    type = 'type'
    id = 'id'
    properties = 'properties'
    geometry = 'geometry'


class Geometry:
    type = 'type'
    coordinates = 'coordinates'


class LinkProperties:
    id = 'ID_1'
    length = 'LENGTH'
    speed_ab = 'AB_SPEED'
    speed_ba = 'BA_SPEED'
    capacity_ab = 'AB_CAPACIT'
    capacity_ba = 'BA_CAPACIT'
    zone_ab = 'AB_ZONE'
    zone_ba = 'BA_ZONE'
    lanes_ab = 'AB_LANES'
    lanes_ba = 'BA_LANES'
    latitude_a = 'LAT1'
    longitude_a = 'LON1'
    latitude_b = 'LAT2'
    longitude_b = 'LON2'


class NodeProperties:
    id = 'ID'
    latitude = 'LAT'
    longitude = 'LON'


class LinkAttributes:
    free_flow = 'free_flow_travel_time'
    capacity = 'capacity'
