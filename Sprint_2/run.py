from flask import Flask, request, render_template, url_for,redirect, flash
from Game.Board import Board
from Game.Levels import Levels
from Game.User import User
from CNN import CNN
import numpy as np

if __name__ == "__main__":
    # Begin Flask App
    app = Flask(__name__)
    app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

    # Create Object with the list of levels
    levels = Levels()

    # Create Objectto interact with User
    user = User()

    # Get CNN for to turn graphics into Matrix for Agent
    print("Getting CNN")
    cnn = CNN()
    print(cnn.model.evaluate(cnn.x_test, cnn.y_test))

    # Initialize some variables
    current_level = levels.getLevelByIndex(0)
    board = ""
    moves_left = current_level.nbr_moves

    # Index.html
    @app.route("/", methods=['GET', 'POST'])
    def index():
        global board, current_level

        # If GET, we return index.html
        if request.method == 'GET': 
            return render_template('index.html', team=user.roster)
        
        # If POST, the team is selected so redirect to level 1
        else:
            user.setCurrentTeam(user.chooseTeam(request.form.get('selected_hidden')))
            return redirect(url_for('level', index=current_level.index))

    # Level.html
    @app.route("/level/<int:index>", methods=['GET', 'POST'])        
    def level(index):
        global board, current_level, user, moves_left, levels

        # Verify that the Team was set
        try:
            assert(user.team is not None)

            # If GET, initialize level and render it
            if request.method == 'GET': 
                current_level = levels.getLevelByIndex(index)
                board = Board(current_level) 
                moves_left = current_level.nbr_moves
                return render_template('level.html', board=board, current_level=current_level, moves_left=moves_left, team=user.team, alert="")
            
            # If POST, act move or AI acts move, then render level
            else:
                requested_move = user.getMove(request.form.get('selected_hidden'))
                move = requested_move
                #if requested_move == [00,00]: move = cnn.getMove()
                if board.verifyMove(move):
                    cnn.visualMatrixToMatrixForAgent(board, user) # Until Agent completed
                    cnn.recordVisualizedMatrix(board) # Until Agent completed
                    board.enactMove(move,current_level, user)
                    #if requested_move == [00,00]:cnn.recordVisualizedMatrix(board)
                    moves_left -= 1

                    # If succeded, add new pokemon to user, message and render index
                    if current_level.hp < current_level.score:
                        l = levels.level_list[current_level.index]
                        if not np.any(user.roster[:, 0] == str(l[0]).capitalize()):
                            user.roster = np.append(user.roster, [[str(l[0]).capitalize(),int(l[1]/100)]], axis=0) # Ajoute le pokemon du niveau au choix de l'utilisateur
                        current_level.index += 1
                        if current_level.index > len(levels.level_list)-1: current_level.index = 0
                        flash("You win")
                        return redirect(url_for('index'))
                    
                    # If failed, message and render index
                    if moves_left == 0:
                        flash("You lose")
                        return redirect(url_for('index'))

                    # If moves and hp are left, render level
                    return render_template('level.html', board=board, current_level=current_level, moves_left=moves_left, team=user.team, alert="")
                
                # If move is not valid, render level and message
                else:
                    return render_template('level.html', board=board, current_level=current_level, moves_left=moves_left, team=user.team, alert="No match. Try again.")
        
        # If team is not selected, redirect to Index.html
        except AttributeError:
            return redirect(url_for('index'))
        
        # If error raised during level, print and redirect
        except Exception as e: 
            print(e)
            return redirect(url_for('index'))

    # Begin Flask Server
    app.run(host='127.0.0.1',port=80)