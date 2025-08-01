import numpy as np

def genData(num_data, num_features = 10,  num_customers = 10, num_facilities = 5,
            deg=1, noise_width=0,  seed=135):
    # positive integer parameter
    if type(deg) is not int:
        raise ValueError("deg = {} should be int.".format(deg))
    if deg <= 0:
        raise ValueError("deg = {} should be positive.".format(deg))
    # set seed
    rnd = np.random.RandomState(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features

    customers = list (range (num_customers))
    facilities = list (range (num_facilities))
    edges = [(i, j) for i in customers for j in facilities]
    num_edges= len (edges)
    demands = rnd.normal(2.5, 1.25, num_customers)
    demands = np.clip(demands, 1, 6)
    demands = np.around (demands, decimals=2)

    capacities = rnd.uniform(5, 15, num_facilities)
    ## Ensure: the demands can be accomodated by the existing capacities
    if np.sum(capacities)<= np.sum (demands):
        cap_sum = np.sum(capacities)
        demand_sum = np.sum (demands)
        capacities *= (demand_sum/ cap_sum)
    capacities =  np.ceil( capacities )
    setup_costs = rnd.uniform(5, 25, num_facilities)
    ### Changing Setup costs to low values, so tha they have low impact on the solution
    # setup_costs = rnd.uniform(3, 8,  num_facilities)
    setup_costs = np.ceil (setup_costs)
    

    # random matrix parameter B
    B = rnd.binomial(1, 0.25, ( num_edges, p)) * rnd.uniform( -2, 2, ( num_edges, p))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))

    c = np.zeros((n, num_edges))
    for i in range(n):

        # noise
        noise = rnd.uniform(1 - noise_width, 1 + noise_width, num_edges)
        # from feature to edge
        values = (((np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3)
                  ** deg) / 3 ** (deg - 1)).reshape(-1) * noise
        values = np.clip(values, 1e-2, 100)

        
        # values = np.ceil(values)
        c[i, :] = values
        c = np.around(c, decimals=2)
        # float
    c = c.astype(np.float64)
    capacities = capacities.astype(np.float64)
    demands = demands.astype(np.float64)
    setup_costs = setup_costs.astype(np.float64)

    return demands, capacities, setup_costs,  x, c 