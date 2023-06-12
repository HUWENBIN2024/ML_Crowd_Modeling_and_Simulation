
def mu(b, I, mu0, mu1):
    """
    Recovery rate.
    input:
    - I: number of infective person
    - b: num of beds per 10000 people
    - mu0: min of mu
    - mu1: max of mu

    output: mu: recovery rate
    
    """
    # recovery rate, depends on mu0, mu1, b
    mu = mu0 + (mu1 - mu0) * (b/(I+b))
    return mu

def R0(beta, d, nu, mu1):
    """
    Basic reproduction number.

    input:
    - beta: contact times
    - mu1: max of mu
    - d: death rate
    - nu: disease-induced death rate

    output: Reproduction number
    """
    return beta / (d + nu + mu1)

def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcations.

    input:
    - I: number of infective person
    - beta: contact times
    - b: num of beds per 10000 people
    - mu0: min of mu
    - mu1: max of mu
    - A: birth rate
    - d: death rate
    - nu: disease-induced death rate

    output: 
    Res Indicator
    """
    c0 = b**2 * d * A
    c1 = b * ((mu0-mu1+2*d) * A + (beta-nu)*b*d)
    c2 = (mu1-mu0)*b*nu + 2*b*d*(beta-nu)+d*A
    c3 = d*(beta-nu)
    res = c0 + c1 * I + c2 * I**2 + c3 * I**3
    return res
    

def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
    SIR model including hospitalization and natural death.
    
    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        average number of adequate contacts per unit time with infectious individuals
    A
        recruitment rate of susceptibles (e.g. birth rate)
    d
        natural death rate
    nu
        disease induced death rate
    b
        hospital beds per 10,000 persons
    """
    S,I,R = y[:]
    m = mu(b, I, mu0, mu1)
    
    dSdt = A - d * S - (beta * S * I) / (S + I + R)
    dIdt = - (d + nu) * I - m * I + (beta * S * I) / (S + I + R)
    dRdt = m * I - d * R
    
    return [dSdt, dIdt, dRdt]
