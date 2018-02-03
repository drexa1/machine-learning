import csv, math
from random import random, randint, shuffle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')



# For small solution spaces it’s better to lower the starting temperature and increase the cooling rate, 
# as it will reduce the simulation time, without lose of quality.
# For bigger solution spaces choose a higher starting temperature and a small cooling rate, 
# as there will be more local minima.
def main():
	cities = get_cities()

	logger.info("Using MCMC with T = 0")
	route_1 = MCMC(cities, STEPS=10_000, T=0)
	print_total_distance(route_1, False)

	logger.info("Using MCMC with T = 1")
	route_2 = MCMC(cities, STEPS=10_000, T=1)
	print_total_distance(route_2, False)

	logger.info("Using MCMC with T = 10")
	route_3 = MCMC(cities, STEPS=10_000, T=10)
	print_total_distance(route_3, False)

	logger.info("Using MCMC with T = 100")
	route_4 = MCMC(cities, STEPS=10_000, T=100)
	print_total_distance(route_4, False)

	logger.info("Using MCMC with simulated annealing, and c = 100")
	route_5 = MCMC_SA(cities, STEPS=10_000, c=100)
	print_total_distance(route_5, True)

# Markov Chain Monte Carlo method for TSP.
# STEPS is the number of iterations
# T is the Markov Chain Monte Carlo temperature
# Lower temperature results in a slower mixing time, and is more likely to
# result in a local optimum, but also requires fewer iterations
def MCMC(cities, STEPS=10_000, T=10):
	route = list(cities.items())
	best  = list(route)
	shuffle(route)

	curr_distance  = total_distance(route)
	best_distance  = total_distance(best)

	for _ in range(STEPS):
		new_route = create_new_route(route, False)
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
def MCMC_SA(cities, STEPS=10_000, c=100):
	route = list(cities.items())
	best = list(route)
	shuffle(route)

	curr_distance = total_distance(route)
	best_distance = total_distance(best)

	for t in range(1, STEPS+1):
		T = c / math.sqrt(t) # quadratic cooling rate
		new_route = create_new_route(route, True)
		new_distance = total_distance(new_route)
		delta_distance = new_distance - curr_distance

		# either is an improve
		# or else Boltzmann function of probability distribution lower than random value in 0-1 range
		# (it is not an improvement but I still grant myself to keep exploring around).
		# The Boltzmann distribution is a probability distribution that gives the probability of a certain state 
		# as a function of that state’s energy and temperature.
		# H(T) = ∑j e^(−Ej / kT)
		rand = random()
		boltzmann = (-1 * delta_distance / T)
		logger.debug("Current distance: {}, new distance: {} [{}] | Temperature: {}, random: {}, Boltzmann: {} [{}]" \
				.format(int(curr_distance), int(new_distance), (delta_distance < 0), round(T,4), rand, math.exp(boltzmann), rand < math.exp(boltzmann)))

		if (delta_distance < 0) or (T > 0 and rand < math.exp(boltzmann)): 
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
		next(city_reader) # skip header

		# make a dictionary with city names as keys and coordinates as values
		cities = {line[0]:[float(line[1]), float(line[2])] for line in city_reader}
		return cities

# route is a list of key-value tuples ('City', [latitude, longitude])
def create_new_route(route, verbose=False):
	new_route = list(route)

	# generate indexes for two random cities
	city_1, city_2 = (randint(0, len(route) - 1) for _ in range(2))
	if verbose:
		logger.debug("Swapping {} with {}.".format(route[city_1][0], route[city_2][0]))

	# swap the cities
	new_route[city_1] = route[city_2]
	new_route[city_2] = route[city_1]

	return new_route

def print_total_distance(route, verbose=False):
	dist = 0.0
	for i in range(len(route) - 1): # skip last element
		x = route[i]
		y = route[i+1]
		tmp = distance(x, y)
		dist += distance(x, y)
		if verbose:
			logger.debug("{} and {} are {} km apart.".format(x[0], y[0], int(tmp)))

	# we finish where we start, so add distance between last and first
	x = route[-1]
	y = route[0]
	tmp = distance(x, y)
	dist += tmp
	if verbose:
		logger.debug("{} and {} are {} km apart.".format(x[0], y[0], int(tmp)))

	logger.info("The total distance is {} km.\n".format(int(dist)))
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
