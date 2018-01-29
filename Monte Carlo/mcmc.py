import csv, math
from random import randint, random, shuffle
import matplotlib.pyplot as plt, numpy as np

# For small solution spaces it’s better to lower the starting temperature and increase the cooling rate, 
# as it will reduce the simulation time, without lose of quality.
# For bigger solution spaces choose a higher starting temperature and a small cooling rate, 
# as there will be more local minima.
def main():
	cities = get_cities()

	print("\nUsing MCMC with MAX_ITER = 10,000 and T = 0")
	route_1 = MCMC(cities, MAX_ITER=10000, T=0)
	print_total_distance(route_1)

	print("\nUsing MCMC with MAX_ITER = 10,000 and T = 1")
	route_2 = MCMC(cities, MAX_ITER=10000, T=1)
	print_total_distance(route_2)

	print("\nUsing MCMC with MAX_ITER = 10,000 and T = 10")
	route_3 = MCMC(cities, MAX_ITER=10000, T=10)
	print_total_distance(route_3)

	print("\nUsing MCMC with MAX_ITER = 10,000 and T = 100")
	route_4 = MCMC(cities, MAX_ITER=10000, T=100)
	print_total_distance(route_4)

	print("\nUsing MCMC with Simulated Annealing, MAX_ITER = 10,000 and c = 100")
	route_5 = MCMC_SA(cities, MAX_ITER=10000, c=100)
	print_total_distance(route_5)

# Markov Chain Monte Carlo method for TSP.
# MAX_ITER is the number of iterations
# T is the Markov Chain Monte Carlo temperature
# Lower temperature results in a slower mixing time, and is more likely to
# result in a local optimum, but also requires fewer iterations
def MCMC(cities, MAX_ITER=10000, T=10):
	route = list(cities.items())
	best  = list(route)
	shuffle(route)

	curr_distance  = total_distance(route)
	best_distance  = total_distance(best)

	for _ in range(MAX_ITER):
		new_route = create_new_route(route)
		new_distance   = total_distance(new_route)
		delta_distance = new_distance - curr_distance

		if (delta_distance < 0) or \
		(T > 0 and random() < math.exp(-1 * delta_distance / T)):
			route = new_route
			curr_distance = new_distance

		if curr_distance < best_distance:
			best = route
			best_distance = curr_distance

	return best

# MCMC with Simulated Annealing (T decreases over time over a cooling rate)
def MCMC_SA(cities, MAX_ITER=10000, c=100):
	route = list(cities.items())
	best  = list(route)
	shuffle(route)

	curr_distance  = total_distance(route)
	best_distance  = total_distance(best)

	for t in range(1, MAX_ITER+1):
		T = c / math.sqrt(t) # cuadratic cooling rate
		new_route = create_new_route(route)
		new_distance   = total_distance(new_route)
		delta_distance = new_distance - curr_distance

		# either is an improve
		if (delta_distance < 0) or \
		# or else Boltzmann function of probability distribution lower than random value in 0-1 range.
		# The Boltzmann distribution is a probability distribution that gives the probability of a certain state 
		# as a function of that state’s energy and temperature.
		(T > 0 and random() < math.exp(-1 * delta_distance / T)): 
			route = new_route
			curr_distance = new_distance

		if curr_distance < best_distance:
			best = route
			best_distance = curr_distance

	return best

# Load the locations of the cities into memory
def get_cities(cities_csv_file="cities.csv"):
	with open(cities_csv_file, 'rU') as cities_csv:
		city_reader = csv.reader(cities_csv, delimiter = ',')
		next(city_reader) # skip first line (column headings)

		# we'll make a dict with city names as keys and coordinates as values
		# line[0] = city name, line[1] = latitude, line[2] = longitude
		cities = {line[0]:[float(line[1]), float(line[2])] for line in city_reader}
		return cities

# route is a list of key-value tuples ('City', [latitude, longitude])
def create_new_route(route):
	new_route = list(route)

	# generate indices for two random cities
	city_1, city_2 = (randint(0, len(route) - 1) for _ in range(2))

	# swap the cities
	new_route[city_1] = route[city_2]
	new_route[city_2] = route[city_1]

	return new_route

def print_total_distance(route, verbose=1):
	dist = 0.0
	for i in range(len(route) - 1): # skip last element
		x = route[i]
		y = route[i+1]
		tmp = distance(x, y)
		if verbose:
			print("{} and {} are {} km apart.".format(x[0], y[0], int(tmp)))
		dist += distance(x, y)

	# we finish where we start, so add distance between last and first
	x = route[-1]
	y = route[0]
	tmp = distance(x, y)
	if verbose:
		print("{} and {} are {} km apart.".format(x[0], y[0], int(tmp)))
	dist += tmp

	print("The total distance is {} km.".format(int(dist)))
	return dist

# route is a list of key-value tuples, ('City', [latitude, longitude])
def total_distance(route):
	dist = 0.0
	for i in range(len(route) - 1): # skip last element
		x = route[i]
		y = route[i+1]
		dist += distance(x, y)

	# we finish where we start, so add distance between last and first
	dist += distance(route[-1], route[0])
	return dist

# haversine distance between x and y
# x and y are kv tuples of the form ('City', [latitude, longitude])
def distance(x, y):
    # convert degrees to radians
    lat_x, lon_x = map(math.radians, x[1])
    lat_y, lon_y = map(math.radians, y[1])

    # haversine of distance / radius
    h = (haversine(lat_y - lat_x) + math.cos(lat_x) * math.cos(lat_y) * haversine(lon_y - lon_x))
    r = 6371 # radius of earth in kilometers
    d = 2.0 * r * math.asin(math.sqrt(h))
    return d

# haversine function
def haversine(theta):
    return math.sin(theta / 2.0) ** 2

if __name__ == "__main__":
    # execute only if run as a script
    main()
