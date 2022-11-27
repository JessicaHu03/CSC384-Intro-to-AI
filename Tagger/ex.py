###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids:Arjun Bhavsar (arjbhavs)
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
#from ioperator import itemgetter

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    
    pos = ['adj','adv','adp','conj','det','noun','num','pron', 'prt', 'verb','x','.']
    trans_f =[[0 for x in range(12)] for y in range(12)]
    trans_p ={}
    totalCount = {}
    posProb = {}
    emit_P = {}
    startP = {}
    freqs = {}
    emit_f = [[]]


    def posteriorlogSimple(self,sentence, label):
        prob_multiplication = 1
        for i in range(len(sentence)):
            word = (sentence[i],label[i])
            
            temp = 1
            if word in self.emit_P:
                temp = self.emit_P[word]
                
            prob_multiplication *= temp
            
        try:
            return math.log10(prob_multiplication)
        except ValueError:
            return 0


    def posteriorlogViterbi(self,sentence, label):
        prob_multiplication = 1
        for i in range(len(sentence)):
            word = (sentence[i],label[i])
            
            temp = 1
            if word in self.emit_P:
                temp = self.emit_P[word]
                
            if i == 0:
                labelProbability = self.startP[label[i]]
            else:
                labelProbability = self.trans_p[(label[i],label[i-1])]
                
            prob_multiplication *= (temp * labelProbability)
            
        try:
            return math.log10(prob_multiplication)
        except ValueError:
            return 0
        
    
    def posteriorlogmcmc(self,sentence, label):
        prob_multiplication = 1
        for i in range(len(sentence)):
            word = (sentence[i],label[i])
            
            temp = 1
            if word in self.emit_P:
                temp = self.emit_P[word]
            
                       
            if i == 0:
                labelProbability = self.startP[label[i]]
            else:
                labelProbability = self.trans_p[(label[i],label[i-1])]
                
            prob_multiplication *= (temp * labelProbability* self.posProb[label[i]])
            
        try:
            return math.log10(prob_multiplication)
        except ValueError:
            return 0
    
    
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return self.posteriorlogSimple(sentence,label)
        elif model == "Complex":
            return self.posteriorlogViterbi(sentence,label)
        elif model == "HMM":
            return self.posteriorlogmcmc(sentence,label)
        else:
            print("Unknown algo!")
    
    def complex_Mcmc(self,sentence,startProbability,transitionP, emission):
        
        labels = [ "noun" ] * len(sentence)
        
        for iter in range(400):
            for i in range(len(sentence)) :
                speechProb = []
                sumProb = 0
                for j in range(len(self.pos)):
                   
                    wordPOS = (sentence[i],self.pos[j])
                    wordPOSProb = 1e-10
                    if wordPOS in emission:
                        wordPOSProb = emission[wordPOS]

                    sumProb += wordPOSProb
                    speechProb.append(wordPOSProb) 
                
                cummulativeSum = 0
                
                rand = random.random()
                
                for l in range(len(speechProb)):
                    speechProb[l] = speechProb[l]/sumProb
                    cummulativeSum += speechProb[l]
                    speechProb[l] = cummulativeSum
                    
                    if rand<speechProb[l]:
                        labels[i] = self.pos[l]
                        break
        return labels
    
    #Following function is understood and refered from https://en.wikipedia.org/wiki/Viterbi_algorithm
    def viterbi(self,sentence):
       words = []
       for w in sentence:
           words.append(w)
       Z = [{}]
       for sp in self.pos:
           
           temp = 1e-8
           if (words[0],sp) in self.emit_P:
               temp = self.emit_P[(words[0],sp)]
               
           Z[0][sp] = {"p": self.startP[sp] * temp, "pr": None}
        
       for t in range(1, len(words)):
           Z.append({})
           
           for st in self.pos:
                
               transitionMax = Z[t-1][self.pos[0]]["p"]*self.trans_p[(self.pos[0],st)]
               back = self.pos[0]
                
               for back_st in self.pos[1:]:
                   transitionP = Z[t-1][back_st]["p"]*self.trans_p[(back_st,st)]
                   
                   if transitionP > transitionMax:
                       transitionMax = transitionP
                       back = back_st
               
               prob = 1e-8
               if (words[t],st) in self.emit_P:
                   prob = self.emit_P[(words[t],st)]
               max_prob = transitionMax * prob
               Z[t][st] = {"p": max_prob, "pr": back}
        
       final = []
       max_prob = max(value["p"] for value in Z[-1].values())
        
       before = None
       listofstates = Z[-1].items()
        
       l =[s for s, data in listofstates if data["p"] == max_prob]
       final.append(l[0])
       before = l[0]
       
       ln = len(Z)
       for i in range(ln - 2, -1, -1):
           final.insert(0, Z[i + 1][before]["pr"])
           before = Z[i + 1][before]["pr"]
       return final
                        
    #
    def train(self, data):
        # Basic Data Training
        for (s,gt) in data:
            for l in range(len(s)) :
                temp = s[l]+'_'+gt[l]
                self.freqs[temp] = self.freqs.get(temp,0) + 1
                self.totalCount[gt[l]] = self.totalCount.get(gt[l],0) + 1
        
        sumpos = sum(self.totalCount.values())
        for w in self.totalCount:
            self.posProb[w] = (self.totalCount[w]/ sumpos)
    
        #Start Probability
        c =0 
        for (s,gt) in data:
            c +=1
            self.startP[gt[0]] = self.startP.get(gt[0],0) + 1
        
        for l in range(len(self.pos)):
            self.startP[self.pos[l]] = math.sqrt(self.startP[self.pos[l]]/c)
            
        #Transition Frquency Count
        counter = 0
        for (sen,ps) in data:
            for i in range(len(ps)-1):
                x = self.pos.index(ps[i])
                y = self.pos.index(ps[i+1])
                val = self.trans_f[x][y]
                counter +=1
                self.trans_f[x][y] = val + 1
        
        #Transition Probability
        for row in range(len(self.pos)):
            for col in range(len(self.pos)):
                self.trans_p[(self.pos[row],self.pos[col])] = abs(self.trans_f[row][col] / counter)
        
        #Emition Probability        
        for (words,ps) in data:
            for w in words:
                for i in range(len(self.pos)):
                    temp = w+'_'+self.pos[i]
                    if temp in self.freqs:
                        self.emit_P[(w,self.pos[i])] = self.freqs[temp] / self.totalCount[self.pos[i]]
        pass
        
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        final = []
        
        for w in sentence:
            freq = []
            for i in range(len(self.pos)):
                temp = w+'_'+self.pos[i]
                if temp in self.freqs:
                    freq.append(self.freqs[temp])
                else:
                    freq.append(0)
            final.append(self.pos[freq.index(max(freq))])
        return final
    
    #[ "noun" ] * len(sentence)

    def complex_mcmc(self, sentence):
       # return [ "noun" ] * len(sentence)
        return self.complex_Mcmc(sentence, self.startP,self.trans_p,self.emit_P)
        

    def hmm_viterbi(self, sentence):
        #return [ "noun" ] * len(sentence)
        
        #for i in pos:
          #  prob = 0
          #  if (sentence[0],i) in emit_P:
          #      prob = emit_P[(sentence[0],i)]
            #startP[i] = prob

        
        return self.viterbi(sentence)

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
