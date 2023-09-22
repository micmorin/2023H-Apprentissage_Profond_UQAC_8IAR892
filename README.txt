


Projet Final:
Rapport de Sprint #2


Michael Morin
D�partement d�Informatique et de Math�matique
Universit� du Qu�bec � Chicoutimi
8INF892 � Apprentissage Profond
Dr. Kevin Bouchard
7 Avril 2023


Rapport de Sprint #2

Probl�matique � adresser
	La solution logicielle jusqu�� pr�sent recr�e une mini version de Pok�mon Shuffle. Alors que l�affichage jusqu�� pr�sent s�est fait sur la console, la prochaine �tape inclus une transition vers une affichage graphique. Bien que cette partie ne soit pas obligatoire, elle permet de rendre la solution plus facile d�utilisation et r�pond mieux � sa fonction initiale de mini-jeu. Une partie de cette transition inclut l�ajout d�ic�nes repr�sentant les pok�mons et les obstacles. En fait, cette repr�sentation graphique est n�cessaire pour la deuxi�me grande partie de ce sprint, la cr�ation et l�optimisation un CNN de classification d�image. Le but de ce r�seau de neurones est de prendre la matrice cr�e pr�c�demment sous forme graphique afin de recr�er la matrice form�e par les images pour ensuite l�envoyer � l�agent de renforcement � la fin de ce sprint, le CNN pourrait �tre r�utiliser pour agir directement sur la version originale de Pok�mon Shuffle.

Travail r�alis�
	La transition graphique � une importance moyenne, telle que discut� pr�c�demment. Cependant, elle constitue une valeur ajout� int�ressante donc elle sera d�crite bri�vement. L�interface utilisateur consiste en un serveur Flask pour un affichage web sur 127.0.01:80 (localhost). Deux routes sont disponibles, soient l�index et �level/<int>�, pour les m�thodes �Get� et �Post�. Alors que l�index permet de choisir l��quipe voulu, la route du level permet l�affichage du niveau ainsi que le choix des mouvements. Pour ce qui est des fonctionnalit�s internes, les fonctions du sprint #1 ont �t� r�utilis� dans leur enti�ret�s. La principale nouveaut� provient de l�affichage de l�historique. � ce point-ci, le bouton AI ne fonctionne pas car il va faire appel � l�agent de renforcement. 





Page d�accueil
Page de jeu
      Bien que l�interface graphique permette d�acc�der aux fonctionnalit�s du jeu, elle ne permet pas de demander un nouvel entrainement du CNN. Le model utilis� par le jeu est sauver dans les fichiers model.h5/.json et ils sont import� au d�marrage. Le CNN a �t� choisi au lieu du MLP car, en g�n�ral, les performances de ce dernier sont plus basses que sa contrepartie et il tend � �tre plus long � mettre en place. Pour ce qui est du choix des couches, je suis partie du travail pr�c�dant utilisant le MINST puis j�ai fait des ajustements manuels jusqu�� l�obtention d�un mod�le avec une exactitude au-del� de 80% au moins une fois sur six. Ensuite, j�ai utilis� Keras Tuner avec des param�tres entre 32 et 128 (step=16) pour les trois filtres et le filtre dense, entre 3 ou 5 pour la taille des noyaux, entre .01 ou .001 pour le taux d�apprentissage et entre 1 et 237 (step=2) pour la taille des lots. Apr�s 348 essaies, les noyaux ont tous �t� plac� � 3, le taux d�apprentissage a �t� plac� � .001 et la taille des lots � �t� r�duit � 3-150 (step=1). En utilisant ces param�tres sur 543 essaies, les valeurs du filtre 2 ont �t� r�duit � 112 ou 128, les filtre 3 et dense ont vu leur minimum rehauss� � 64 et les tailles de lots ne comprennent que 8 valeurs possibles. Avec 280 essaies, les valeurs pour le filtre 3 sont tomb�s � 112 et 128 alors que le filtre dense 1 a eu ces valeurs en plus de 96. Le premier filtre, de son c�t� a vu ses valeurs r�duites � 64, 80 et 112 alors que la tailles des lots ont perdues les valeurs 5, 7 et 15.
	Apr�s ces essaies et quelques autres, les mod�les n��tait toujours pas consistant. C�est � ce moment que j�ai r�alis� que le dataset �tait dynamique. En fait, pour ce CNN, le dataset a �t� assembl� avec 258 images de 11 cat�gories (8 pokemons, 3 obstacles). Chaque image a �t� coup�, a vu son arri�re-plan disparaitre avent d��tre redimensionn� en 32x32px. Puisque 258 est un nombre d�image beaucoup trop petits, ils ont �t� transform�s en vecteur et pass� � un g�n�rateur d�image augment�es. Le probl�me d�inconsistance des mod�les provenait du fait que je passais ce g�n�rateur au Keras Tuner au lieu d�un dataset fixe. Pour r�gler ce probl�me, le g�n�rateur a �t� utilis� pour g�n�rer un fichier csv qui est ensuite passer au Tuner. Apr�s pr�s d�une semaine de �tuning�, les meilleurs mod�les ont enfin �t� trouv�. 

Bilan et Prochain Sprint
	�norm�ment de temps lors de ce sprint a �t� utilis� pour l�optimisation du mod�le. En fait, un mod�le avec une exactitude de >90% a �t� trouv� rapidement, mais mes efforts ont �t� principalement pour trouver les param�tres donnant les meilleur mod�les et le meilleur mod�le possible. Un effort important � aussi �t� donn� lors de la cr�ation du dataset puisqu�aucun de ce genre n�existait encore. Le jeu permet maintenant de s�lectionner l��quipe et de faire les movement graphiquement en cliquant sur les cases avant de confirmer avec un bouton. De plus, l�historique des mouvements, des combos et des interpr�tations du CNN sont disponible facilement et graphiquement. Pour confirmer le bon fonctionnement du programme, voici une trace d�ex�cution�: 


	Lors du dernier sprint, l�agent de renforcement sera cr�� et optimis�. Bien que ce type d�agent va me demander une bonne recherche pour bien comprendre le concept, son impl�mentation risque de demander un temps moindre compar� � son optimisation. Cependant, une fois l�agent entrain� sauvegard�, il sera ajout� � l�interface graphique, ce qui permettre de compl�ter le mini-jeu avec un robot pour r�soudre les casse-t�tes trop difficiles � l�aide d�un bouton.

