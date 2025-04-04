class Holder:
    factors = {}  
    attributes = () 

    '''
    Constructor of class Holder holding two parameters, self refers to the instance of the class
    '''
    def __init__(self, attr):
        self.attributes = attr
        for i in attr:
            self.factors[i] = []

    def add_values(self, factor, values):
        self.factors[factor] = values

class CandidateElimination:
    Positive = {}  # Initialize positive empty dictionary
    Negative = {}  # Initialize negative empty dictionary

    def __init__(self, data, fact):
        self.num_factors = len(data[0][0])
        self.factors = fact.factors
        self.attr = fact.attributes
        self.dataset = data

    def run_algorithm(self):
        '''
        Initialize the specific and general boundaries, and loop the dataset against the algorithm
        '''
        G = self.initializeG()
        S = self.initializeS()
        '''
        Programmatically populate list in the iterating variable trial_set
        '''
        count = 0
        for trial_set in self.dataset:
            if self.is_positive(trial_set):  # If trial set/example consists of positive examples
                G = self.remove_inconsistent_G(G, trial_set[0])  # Remove inconsistent data from the general boundary

                S_new = S[:]  # Initialize the dictionary with no key-value pair
                print(S_new)
                for s in S:
                    if not self.consistent(s, trial_set[0]):
                        S_new.remove(s)
                    generalization = self.generalize_inconsistent_S(s, trial_set[0])
                    generalization = self.get_general(generalization, G)
                    if generalization:
                        S_new.append(generalization)
                S = S_new[:]
                S = self.remove_more_general(S)
                print(S)
            else:  # If it is negative
                S = self.remove_inconsistent_S(S, trial_set[0])  # Remove inconsistent data from the specific boundary
                G_new = G[:]  # Initialize the dictionary with no key-value pair (dataset can take any value)
                print(G_new)
                for g in G:
                    if self.consistent(g, trial_set[0]):
                        G_new.remove(g)
                    specializations = self.specialize_inconsistent_G(g, trial_set[0])
                    specializationss = self.get_specific(specializations, S)
                    if specializations != []:
                        G_new += specializationss
                G = G_new[:]
                G = self.remove_more_specific(G)
                print(G)
        print(S)
        print(G)

    def initializeS(self):
        ''' Initialize the specific boundary '''
        S = tuple(['-' for factor in range(self.num_factors)])  # 6 constraints in the vector
        return [S]

    def initializeG(self):
        ''' Initialize the general boundary '''
        G = tuple(['?' for factor in range(self.num_factors)])  # 6 constraints in the vector
        return [G]

    def is_positive(self, trial_set):
        ''' Check if a given training trial_set is positive '''
        if trial_set[1] == 'Y':
            return True
        elif trial_set[1] == 'N':
            return False
        else:
            raise TypeError("invalid target value")

    def match_factor(self, value1, value2):
        ''' Check for the factors values match, necessary while checking the consistency of training trial_set with the hypothesis '''
        if value1 == '?' or value2 == '?':
            return True
        elif value1 == value2:
            return True
        return False

    def consistent(self, hypothesis, instance):
        ''' Check whether the instance is part of the hypothesis '''
        for i, factor in enumerate(hypothesis):
            if not self.match_factor(factor, instance[i]):
                return False
        return True

    def remove_inconsistent_G(self, hypotheses, instance):
        ''' For a positive trial_set, the hypotheses in G inconsistent with it should be removed '''
        G_new = hypotheses[:]
        for g in hypotheses:
            if not self.consistent(g, instance):
                G_new.remove(g)
        return G_new

    def remove_inconsistent_S(self, hypotheses, instance):
        ''' For a negative trial_set, the hypotheses in S inconsistent with it should be removed '''
        S_new = hypotheses[:]
        for s in hypotheses:
            if self.consistent(s, instance):
                S_new.remove(s)
        return S_new

    def remove_more_general(self, hypotheses):
        ''' After generalizing S for a positive trial_set, the hypothesis in S general than others in S should be removed '''
        S_new = hypotheses[:]
        for old in hypotheses:
            for new in S_new:
                if old != new and self.more_general(new, old):
                    S_new.remove(new)
        return S_new

    def remove_more_specific(self, hypotheses):
        ''' After specializing G for a negative trial_set, the hypothesis in G specific than others in G should be removed '''
        G_new = hypotheses[:]
        for old in hypotheses:
            for new in G_new:
                if old != new and self.more_specific(new, old):
                    G_new.remove(new)
        return G_new

    def generalize_inconsistent_S(self, hypothesis, instance):
        ''' When a inconsistent hypothesis for positive trial_set is seen in the specific boundary S,
        it should be generalized to be consistent with the trial_set ... we will get one hypothesis '''
        hypo = list(hypothesis)  # Convert tuple to list for mutability
        for i, factor in enumerate(hypo):
            if factor == '-':
                hypo[i] = instance[i]
            elif not self.match_factor(factor, instance[i]):
                hypo[i] = '?'
        generalization = tuple(hypo)  # Convert list back to tuple for immutability
        return generalization

    def specialize_inconsistent_G(self, hypothesis, instance):
        ''' When a inconsistent hypothesis for negative trial_set is seen in the general boundary G
        should be specialized to be consistent with the trial_set.. we will get a set of hypotheses '''
        specializations = []
        hypo = list(hypothesis)  # Convert tuple to list for mutability
        for i, factor in enumerate(hypo):
            if factor == '?':
                values = self.factors[self.attr[i]]
                for j in values:
                    if instance[i] != j:
                        hyp = hypo[:]
                        hyp[i] = j
                        hyp = tuple(hyp)  # Convert list back to tuple for immutability
                        specializations.append(hyp)
        return specializations

    def get_general(self, generalization, G):
        ''' Checks if there is more general hypothesis in G for a generalization of inconsistent hypothesis in S
        in case of positive trial_set and returns valid generalization '''
        for g in G:
            if self.more_general(g, generalization):
                return generalization
        return None

    def get_specific(self, specializations, S):
        ''' Checks if there is more specific hypothesis in S for each of hypothesis in specializations of an
        inconsistent hypothesis in G in case of negative trial_set and return the valid specializations'''
        valid_specializations = []
        for hypo in specializations:
            for s in S:
                if self.more_specific(s, hypo) or s == self.initializeS()[0]:
                    valid_specializations.append(hypo)
        return valid_specializations

    def exists_general(self, hypothesis, G):
        '''Used to check if there exists a more general hypothesis in general boundary for version space'''
        for g in G:
            if self.more_general(g, hypothesis):
                return True
        return False

    def exists_specific(self, hypothesis, S):
        '''Used to check if there exists a more specific hypothesis in general boundary for version space'''
        for s in S:
            if self.more_specific(s, hypothesis):
                return True
        return False

    def more_general(self, hyp1, hyp2):
        ''' Check whether hyp1 is more general than hyp2 '''
        hyp = zip(hyp1, hyp2)
        for i, j in hyp:
            if i == '?':
                continue
            elif j == '?':
                if i != '?':
                    return False
            elif i != j:
                return False
            else:
                continue
        return True

    def more_specific(self, hyp1, hyp2):
        ''' hyp1 more specific than hyp2 is equivalent to hyp2 being more general than hyp1 '''
        return self.more_general(hyp2, hyp1)

dataset = [
    (('sunny', 'warm', 'normal', 'strong', 'warm', 'same'), 'Y'),
    (('sunny', 'warm', 'high', 'strong', 'warm', 'same'), 'Y'),
    (('rainy', 'cold', 'high', 'strong', 'warm', 'change'), 'N'),
    (('sunny', 'warm', 'high', 'strong', 'cool', 'change'), 'Y')
]
attributes = ('Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast')
f = Holder(attributes)
f.add_values('Sky', ('sunny', 'rainy', 'cloudy'))  
f.add_values('Temp', ('cold', 'warm'))  
f.add_values('Humidity', ('normal', 'high'))  
f.add_values('Wind', ('weak', 'strong'))
f.add_values('Water', ('weak', 'strong'))
f.add_values('Forecast',('same','change'))
a = CandidateElimination(dataset,f)
a.run_algorithm()
