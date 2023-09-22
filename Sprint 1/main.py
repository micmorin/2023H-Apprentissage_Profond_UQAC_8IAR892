from Board import Board
from Levels import Levels
from User import User

##########
# Projet Final - Sprint 1
# Michael Morin MORM07039500
# 8INF892 – Apprentissage Profond
# 24 Mars 2023
##########


if __name__ == "__main__":
    levels = Levels()   # Crée un object qui maintient la liste des niveaux du jeu 

    for i in range(len(levels.level_list)):
        current_level = levels.getLevelByIndex(i)
        user = User()   # Crée un objet qui interagit avec l'utilisateur

        current_level.setCurrentTeam(user.chooseTeam())
        
        board = Board(current_level)    # Initialise la matrice de jeu
        moves_left = current_level.nbr_moves
        
        # (Debut) Boucle de jeu
        while moves_left > 0:           
            # Verifie si le temps est venu d'ajouter des obstructions et fait l'ajout au besoin
            if (current_level.nbr_moves-moves_left) % current_level.disruption_counter == 0:
                board.addDisruptions(current_level.disruption)

            print("\n"+str(moves_left)+" moves left")

            # Boucle de choix valide de movement
            move = user.getMove()
            while not board.verifyMove(move):
                print("No match. Try again\n")
                move = user.getMove()

            # Effectue le movement
            board.enactMove(move, current_level)
            moves_left -= 1 

            # Verifie si le niveau est terminer
            if current_level.hp < current_level.score: 
                break 

        # (Fin) Boucle de jeu    

        # Verifie si le joueur a reussit le niveau
        if current_level.hp < current_level.score:
            current_level.hp = 0
            board.printLayout(current_level)
            print("You win!")
            user.roster.append([current_level.name,current_level.hp/100]) # Ajoute le pokemon du nivau au choix de l'utilisateur

        else:
            print("You lose!")
            i -= 1  # Recommence le niveau