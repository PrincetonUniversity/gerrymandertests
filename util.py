import gerrytests as gt
import json, os
from pprint import pprint

def parse_results(filename):
	""" Read the individual election results file and store the result in memory.

	This function is called at application start.
	"""
	STATES = [
		'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
		'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
		'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
		'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
		'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
	]

	# Initialize results dict
	results = {str(y) : {} for y in range(1948, 2018, 2)};
	with open(filename) as file:
		for line in file:
			try:
				(year, state, _, res, _, _) = line.split(',')
			except:
				print(line)
				continue

			# State Postal code from alphabetical index
			state = STATES[int(state) - 1]

			if state not in results[year]:
				results[year][state] = []
			results[year][state].append(float(res))

	with open('static/data/allresults.json', 'w') as file:
		json.dump(results, file)

	return results

def run_all_tests(all_results, years=None):
	""" Run all of the tests on each of the elections present in results.

	Results should be a dict indexed by year.  Each value is another dict
	giving district-level results for that year.
	"""

	impute_val = 0.75

	with open('static/data/precomputed_tests.json', 'r') as file:
		tests = json.load(file)

	if years is None:
		years = all_results.keys()

	for year in years:
		year_results = all_results[year]
		print('Running tests for %s' % year)
		# get all national results for current year
		national_results = []
		for state in year_results:
			if year in ['2012', '2014', '2016'] and state in ['MI', 'NC', 'OH', 'PA', 'VA', 'WI']: continue
			national_results += year_results[state]

		for idx, x in enumerate(national_results):
			if x == 1: national_results[idx] = impute_val
			if x == 0: national_results[idx] = 1 - impute_val

		# dict for storing test results
		if year not in tests: tests[year] = {}
		for state, state_results in year_results.items():
			# DO IMPUTATION
			imputed = list(state_results)
			for idx, x in enumerate(state_results):
				if x == 1: imputed[idx] = impute_val
				if x == 0: imputed[idx] = 1 - impute_val

			print('\t- %s' % state)
			# run each test and save outcome
			tests[year][state] = {
				"test1"		: gt.test_lopsided_wins(imputed),
				"test2"		: gt.test_consistent_advantage(imputed),
				"test3"		: gt.test_fantasy_delegations(imputed, national_results, n_sims=1000000),
				"voteshare" : sum(state_results) / len(state_results),
				"seats"		: len([0 for r in state_results if r > 0.5]),
				"results"	: state_results,
				"ndists"	: len(state_results),
				"nall"		: len(all_results),
				"state"		: state,
				"year"		: year
			}

	with open('static/data/precomputed_tests.json', 'w') as file:
		json.dump(tests, file)

	return tests

def load_content():
	""" From the content folder, load each json file into a single dictionary indexed by page.
	"""
	print('Initializing content...\n')
	content = {}
	for f in os.listdir('content'):
		name = f[:-5]	# up until ".json"
		print('Loading %s...' % name)
		with open(os.path.join('content', f), encoding='utf-8') as file:
			content[name] = json.load(file)
			#print(content[name], '\n')

	return content

def geojson_convert(fname):
	""" Some weird stuff with the Highcharts-provided geojson files.  Let's see if we can fix it.
	"""
	with open(fname) as file:
		geo = json.load(file)

	newgeo = {'features': [], 'type': 'FeatureCollection'}

	for obj in geo['geometries']:
		newgeo['features'].append({
			'type': 'feature',
			'geometry': obj,
			'properties': {}
		})

	pprint(newgeo)

if __name__=="__main__":
	# results = parse_results('static/data/results1948.csv')
	# run_all_tests(results, [str(y) for y in range(1948, 2012, 2)])
	geojson_convert('static/data/districts/us-pa-congress-113.geo.json')
