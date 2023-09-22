import pandas as pd
import numpy as np

class User(object):
    def __init__(self):
        self.roster = np.array([["Pidgey", 9], ["Weedle", 7], ["Eevee", 10], ["Pikachu", 12]])

    def getMove(self):
        move = input("Enter move:")
        move = " ".join(move)
        return move.split()
    
    def chooseTeam(self):
        # Affiche les pokemon disponibles
        print()
        print(pd.DataFrame({
                            "Pokemon": self.roster[:,0],
                            "Power": self.roster[:,1]
                            }))
        # Demande le choix de l'utilisateur
        user_in = input("Enter team choice:")
        user_in = " ".join(user_in)
        args = user_in.split()
        args = np.unique(args).astype(int)

        # Verifie que le choix est valide
        while len(args) != 4 and args.max() <= len(self.roster):
            print("The team provided had an error. Try again")
            args = input("Enter team choice:").split()

        # retourne le choix de l'utilisateur
        return [self.roster[args[0]], self.roster[args[1]], self.roster[args[2]], self.roster[args[3]]]