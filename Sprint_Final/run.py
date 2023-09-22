from flask import Flask, request, render_template, url_for,redirect, flash, Markup
from Game.Board import Board
from Game.Levels import Levels
from Game.User import User
from CNN import cnn
from DQN import dqn
import numpy as np

if __name__ == "__main__":
    # Begin Flask App
    print("Setting up APP")
    app = Flask(__name__)
    app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
    app.jinja_env.filters['zip'] = zip

    # Create Object with the list of levels
    print("Getting Levels")
    levels = Levels()

    # Create Object to interact with User
    print("Getting User")
    user = User()

    # Get CNN for to turn graphics into Matrix for Agent
    print("Getting CNN")
    cnn = cnn()
    print(cnn.model.evaluate(cnn.x_test, cnn.y_test))

    # Get DQN for AI functionnalities
    print("Getting DQN")
    agent = ""

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
        global board, current_level, user, moves_left, levels, agent

        # Verify that the Team was set
        try:
            assert(user.team is not None)

            # If GET, initialize level and render it
            if request.method == 'GET': 
                current_level = levels.getLevelByIndex(index)
                board = Board(current_level) 
                agent = dqn(cnn, board, user)
                moves_left = current_level.nbr_moves
                return render_template('level.html', board=board, current_level=current_level, moves_left=moves_left, team=user.team, alert="")
            
            # If POST, act move or AI acts move, then render level
            else:
                requested_move = user.getMove(request.form.get('selected_hidden'))
                move = requested_move
                if requested_move == ['0','0','0','0']: 
                    move, c = agent.solve(cnn, board, user)
                    board.history[len(board.history)-1].append(cnn.matrix)
                    flash("Found move "+ str(int(move[0]))+','+str(int(move[1]))+Markup(' &rarr; ')+str(int(move[2]))+','+str(int(move[3]))
                          +" after "+str(c)+" attempts.")


                if board.verifyMove(move):
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