import math
import random
import pyproj
import mlrose

import json

from geopy.geocoders import Nominatim

from flask import Flask, render_template
from flask import request
import osmapi

app = Flask(__name__, static_folder='static')
geolocator = Nominatim(user_agent="divergente")

THICKNESS = 0.25 # thickness of box relative to distance between A and B
MIN_LENGTH = 250 # in meters

def bounds(a, b):
    lonA, latA = a
    lonB, latB = b
    xA, yA = pyproj.transform('merc', 'epsg:3857', lonA, latA)
    xB, yB = pyproj.transform('merc', 'epsg:3857', lonB, latB)
    slope = (xB - xA) / (yB - yA)
    distance = math.sqrt((xB - xA) ** 2 + (yB - yA) ** 2)
    length = max(distance * THICKNESS, MIN_LENGTH)
    dx = length / math.sqrt(1 + slope ** 2)
    dy = dx * slope
    p1 = pyproj.transform('epsg:3857', 'merc', xA + dx, yA + dy)
    p2 = pyproj.transform('epsg:3857', 'merc', xA - dx, yA - dy)
    p3 = pyproj.transform('epsg:3857', 'merc', xB + dx, yB + dy)
    p4 = pyproj.transform('epsg:3857', 'merc', xB - dx, yB - dy)
    return p1, p2, p3, p4


def get_bounding_box_data(*lat_lon):
    """
    Only works for latin america <3

    TODO: improve logic
    """
    lats = [l[0] for l in lat_lon]
    lons = [l[1] for l in lat_lon]

    min_lon = min(lons)
    min_lat = min(lats)
    max_lon = max(lons)
    max_lat = max(lats)

    api = osmapi.OsmApi()
    return api.Map(min_lon, min_lat, max_lon, max_lat)


def shortest_path(coords, start, end):
    tsp_fitness = mlrose.TravellingSales(coords = coords)
    def filter_endpoints(state):
        if state[0] == start and state[-1] == end:
            return tsp_fitness.evaluate(state)
        else:
            return math.inf
    fitness = mlrose.CustomFitness(filter_endpoints, 'tsp')
    problem = mlrose.TSPOpt(length = len(coords), fitness_fn = fitness, maximize = False)
    path, length = mlrose.genetic_alg(problem, random_state = 2)
    return [coords[i] for i in path]


def address_lat_lon(search):
    location = geolocator.geocode(search)
    return (location.latitude, location.longitude)


@app.route('/', methods = ['GET', 'POST'])
def index():
    waypoints = []
    if request.method.upper() == 'POST':
        start = request.form['start']
        end = request.form['end']

        p1 = address_lat_lon(start)
        p2 = address_lat_lon(end)

        bounding_box = bounds(p1, p2)
        data = [d for d in get_bounding_box_data(*bounding_box) if d['type'] == 'node']
        random.shuffle(data)

        intermediaries = [(d['data']['lat'], d['data']['lon']) for d in data[:5]]

        waypoints = shortest_path([p1, p2] + intermediaries, 0, 1)

    return render_template('index.html', waypoints=waypoints)


if __name__ == "__main__":
    app.run(debug=True)
