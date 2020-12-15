from scipy import stats as st

class BernoulliBandit:
    #принимает на вход список из K>=2 чисел из [0,1]
    def __init__(self, means):
        self._means = means
        self._best_mean = max(means)
        self.k = len(means)
        self.current_regret = 0
        self.rewards = [0 for i in range(self.k)]
        self.num_plays = [0 for i in range(self.k)]
        self.emperic_means = [0 for i in range(self.k)]        
  
    #Функция возвращает число ручек
    def K(self):
        return self.k

    #Принимает параметр 0 <= a <= K-1 и возвращает реализацию случайной величины 
    #X c P[X=1] равной среднему значению выигрыша ручки a+1
    def pull(self, a):
        reward = st.bernoulli.rvs(p=self._means[a])
        self.current_regret += st.bernoulli.rvs(p=self._best_mean) - reward
        self.num_plays[a] += 1
        self.rewards[a] += reward
        self.emperic_means[a] = self.rewards[a] / self.num_plays[a]
        return reward

    #Возвращает текущее значение регрета
    def regret(self):
        return self.current_regret

def FollowTheLeader(bandit, T):
    """
    Follow-The-Leader algorithm
    
    Arguments
    ---------
    bandit : BernoulliBandit
    
    T : int
        Number of rounds 
        
    """
    for i in range(bandit.k):
        bandit.pull(i)
    for i in range(T - bandit.k):
        a = bandit.emperic_means.index(max(bandit.emperic_means))
        bandit.pull(a)

def ExploreFirst(bandit, T, k):
    """
    Explore-First algorithm
    
    Arguments
    ---------
    bandit : BernoulliBandit
    
    T : int
        Number of rounds 
        
    K : int
        Number of rounds in exploration phase 
    
    """
    for i in range(bandit.k):
        for j in range(k):
            bandit.pull(i)
    for i in range(T - bandit.k*k):
        a = bandit.emperic_means.index(max(bandit.emperic_means))
        bandit.pull(a)
        
def EGreedy(bandit, T, e):
    """
    Epsilon-Greedy algorithm
    
    Arguments
    ---------
    bandit : BernoulliBandit
    
    T : int
        Number of rounds 
        
    e : float
        Epsilon from [0, 1] 
        
    """
    k = bandit.k
    for t in range(T):
        if st.bernoulli.rvs(p=e):
            a = np.random.randint(0, k)
        else:
            a = bandit.emperic_means.index(max(bandit.emperic_means))
        bandit.pull(a)

def SuccessiveElimination(bandit, T):
    """
    Successive Elimination algorithm
    
    Arguments
    ---------
    bandit : BernoulliBandit
    
    T : int
        Number of rounds 
    
    """
    for i in range(bandit.k):
        bandit.pull(i)
    r = (2*np.log(T))**0.5
    T = T - bandit.k
    active = [i for i in range(bandit.k)]
    n = 1
    while T > 0:
        max_emp_mean = max([bandit.emperic_means[i] for i in active])
        active = [i for i in active if bandit.emperic_means[i] + 2*r >= max_emp_mean]
        for i in active:
            bandit.pull(i)
            T -= 1
        n += 1
        r *= 0.8*((n-1)/n)**0.5

def UCB(bandit, T):
    """
    UCB algorithm
    
    Arguments
    ---------
    bandit : BernoulliBandit
    
    T : int
        Number of rounds 
    
    """
    for i in range(bandit.k):
        bandit.pull(i)
    r = [(2*np.log(T))**0.5 for i in range(bandit.k)]
    for i in range(T - bandit.k):
        ucb = [x + y for x, y in zip(bandit.emperic_means, r)]
        a = ucb.index(max(ucb))
        bandit.pull(a)
        n = bandit.num_plays[a]
        r[a] *= 0.8*((n-1)/n)**0.5

