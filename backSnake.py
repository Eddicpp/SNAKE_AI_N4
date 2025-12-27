''' stiamo andando ad utilizzare un algoritmo genetico 
    -> non è come nella Q-Table classica che ad ogni interazione calcolo la reward e aggiorno i pesi sulla base di questa]
    ma in questo caso si va a lavorare con gli snake solamente quando sono morti tutti.
    una volte che sono morti tutti li vado ad ordinare sulla base dello score che hanno accumulato durante la loro partita (fitness)
    e il miglior 10% lo vado ad usare per creare le generazioni dopo
    il restante 90% verra generato unendo i pesi dei migliori e andando ad applicare delle mutazioni casuali per provare ad uscrie da eventuali minimi locali'''

# cercello dello snake
import numpy as np

# devo andare ad assegnare una probablità ad ogni azione
def softmax(x):
    """Calcola i valori softmax per ogni elemento di x."""
    # Sottraiamo il max per stabilità numerica (evita che exp diventi infinito)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    # ciò che sto facendo è esponenziale della singola azione diviso somma degli esponenziali assegnati ad ongni azione 

class Snake:
    # non arrivo mai a convergenza -> provo a ingrandire il network
    def __init__(self, child_w1=None, child_w2=None, child_w3=None, child_w4=None, child_w5=None, chiled_w6=None):
        if child_w1 is not None and child_w2 is not None and child_w3 is not None and child_w4 is not None and child_w5 is not None and chiled_w6 is not None:
            # nel caso non sia la prima volta che uso la rete neurale carico i pesi che mi sto portanto dietro dal giro scorso
            self.weights1 = child_w1
            self.weights2 = child_w2
            self.weights3 = child_w3
            self.weights4 = child_w4
            self.weights5 = child_w5
            self.weights6 = chiled_w6
            # rete neurale costituita da 6 ayer senza considerare quello di input
        else:
            # nel caso sia la prima volta che vado ad usare la rete neurale vado ad inizializzare i pesi in maniera casuale 
            self.weights1 = np.random.randn(24, 24)  
            self.weights2 = np.random.randn(24, 20)  
            self.weights3 = np.random.randn(20, 16)   
            self.weights4 = np.random.randn(16, 12)   
            self.weights5 = np.random.randn(12, 8)
            self.weights6 = np.random.randn(8, 4)
        
        # inizializzo le variabili che mi serviranno per l'apprendimento della rete (soprattuto nella parte di reward)
        self.fitness = 0 # punteggio che riceve la mia rete neurale 
        self.apples_eaten = 0 # numero di mele prese durante il percorso

        self.prev_distance = None # distanza precedente per capire se la posizione dello snake migliora o peggiore
        self.visited_positions = {} # mi serve per individuare se ci sono dei loop in caso fermo primo il ciclo
        self.steps_since_last_apple = 0 # mi serve per mettere pressione allo snake quando passa tanto tempo dall'ultima volta che ha preso una mela 
        self.position_history = []
    
    # quando lo snake perde ho bisogno di una funzione che azzeri tutto
    def reset(self):
        """
        Resetta lo stato dell'agente all'inizio di una nuova partita
        DEVE essere chiamato da main.py prima di ogni episodio!
        """
        self.fitness = 0
        self.apples_eaten = 0
        self.prev_distance = None
        self.visited_positions = {}
        self.steps_since_last_apple = 0
        self.loop_death = False

    # se voglio capire quale azione vuole fare lo snake devo passare all'interno della rete neurale lo stato attuale dello snake e vedere cosa la rete
    # neurale mi sputa fuori.
    def get_move(self, inputs):
        # Passaggio dei dati attraverso la rete (Feedforward)
        hidden1 = np.tanh(np.dot(inputs, self.weights1))
        hidden2 = np.tanh(np.dot(hidden1, self.weights2))
        hidden3 = np.tanh(np.dot(hidden2, self.weights3))
        hidden4 = np.tanh(np.dot(hidden3, self.weights4))
        hidden5 = np.tanh(np.dot(hidden4, self.weights5))
        output = softmax(np.dot(hidden5, self.weights6)) # -> output della rete è una lista di valori con le probabilità
        # Ritorna la direzione con il valore più alto
        return np.argmax(output) # vado a prendere quella con probabilità maggiore
    # per la rete avevo pensato se inserire dei dropout ma la rete non riesce ad imparare a memoria perchè la mela cambia sempre la sua posizione
            

    # devo andare a valutare le azioni dello snake per andare ad aggiustare i pesi che ho assegnato in maniera casuale
    def reward_function(self, state, apple_just_eaten, died, snake_head, apple_pos):
        """
        REWARD FUNCTION con DISTINZIONE CORNER vs EDGE
        
        Corner = 2 muri adiacenti (molto difficile)
        Edge = 1 muro (difficile)
        Center = nessun muro vicino (facile)
        """
        
        # ════════════════════════════════════════════════════════════
        # 1. TERMINAL STATES
        # ════════════════════════════════════════════════════════════
        # nel caso lo snake muoia vado a togliere tanti punti
        if died:
            self.fitness -= 5000
            return
        
        # se ha appena mangiato una mela 
        if apple_just_eaten:
            self.apples_eaten += 1 # vado ad aumentare il numero di mele mangiate di uno
            base_reward = 10000 # definisco una reward base che lo snake prende
            # in base a dove lo snake ha mangiato la mela gli vado a dare più punti questo perchè prendere le mele sui bordi è più complesso per lui

            # per dare i punti devo capire dove è la posizione
            # ⭐ ANALIZZA POSIZIONE MELA
            x, y = snake_head[0] # capisco dove si trova la testa dello snake
            grid_size = 20  # grandezza del piano di gioco
            
            # Distanze dai 4 bordi
            dist_left = x
            dist_right = (grid_size - 1) - x
            dist_top = y
            dist_bottom = (grid_size - 1) - y
            
            # ════════════════════════════════════════════════════════
            # CLASSIFICAZIONE POSIZIONE
            # ════════════════════════════════════════════════════════
            
            # Definisci soglia "vicino al muro"
            NEAR_WALL_THRESHOLD = 2  # Entro 2 celle = vicino
            
            # Check quali muri sono vicini
            near_left = dist_left <= NEAR_WALL_THRESHOLD
            near_right = dist_right <= NEAR_WALL_THRESHOLD
            near_top = dist_top <= NEAR_WALL_THRESHOLD
            near_bottom = dist_bottom <= NEAR_WALL_THRESHOLD
            
            walls_nearby = sum([near_left, near_right, near_top, near_bottom])
            
            # ⭐ CATEGORIA 1: ANGOLO (2 muri perpendicolari)
            is_corner = (
                (near_left and near_top) or      # Angolo top-left
                (near_left and near_bottom) or   # Angolo bottom-left
                (near_right and near_top) or     # Angolo top-right
                (near_right and near_bottom)     # Angolo bottom-right
            )
            
            # ⭐ CATEGORIA 2: BORDO (1 muro, ma non angolo)
            is_edge = (walls_nearby == 1)
            
            # ⭐ CATEGORIA 3: CENTRO (nessun muro vicino)
            is_center = (walls_nearby == 0)
            
            # ════════════════════════════════════════════════════════
            # ASSEGNA REWARD BASATO SU CATEGORIA
            # ════════════════════════════════════════════════════════
            
            if is_corner:
                # 🔥🔥🔥 ANGOLO - MASSIMO REWARD
                corner_bonus = 100000 # bonus molto elevato perché deve imparare ad andarli a prendere
                total_reward = base_reward + corner_bonus 
                
                # Identifica quale angolo
                if near_left and near_top:
                    corner_name = "TOP-LEFT"
                elif near_left and near_bottom:
                    corner_name = "BOTTOM-LEFT"
                elif near_right and near_top:
                    corner_name = "TOP-RIGHT"
                else:
                    corner_name = "BOTTOM-RIGHT"
            
            elif is_edge:
                # 🔥 BORDO - REWARD MEDIO
                
                # Calcola quanto è vicino al bordo (più vicino = più reward)
                min_dist_from_wall = min(dist_left, dist_right, dist_top, dist_bottom)
                
                if min_dist_from_wall == 0: # non è possibile come cosa ma lo implemento nel caso ci possano essere bug
                    edge_bonus = 50000  # Esattamente sul bordo 
                elif min_dist_from_wall == 1:
                    edge_bonus = 35000  # 1 cella dal bordo
                elif min_dist_from_wall == 2:
                    edge_bonus = 20000  # 2 celle dal bordo
                else:
                    edge_bonus = 0
                
                total_reward = base_reward + edge_bonus
                
                # Identifica quale bordo
                if near_left:
                    edge_name = "LEFT"
                elif near_right:
                    edge_name = "RIGHT"
                elif near_top:
                    edge_name = "TOP"
                else:
                    edge_name = "BOTTOM"
            
            else:
                # ✓ CENTRO - REWARD BASE
                total_reward = base_reward
            
            self.fitness += total_reward
            
            # Reset trackers
            self.steps_since_last_apple = 0 # pressione dovuto al tempo trascorso dall'aver preso l'ultima mela si è azzerato
            self.visited_positions = {}
            
            return
        
        # ════════════════════════════════════════════════════════════
        # 2. CALCOLA DISTANZE
        # ════════════════════════════════════════════════════════════
        
        head = np.array(snake_head)
        apple = np.array(apple_pos)
        curr_distance = np.linalg.norm(head - apple) # calcolo distanza tra mela e testa dello snake
        
        # idea: distanza dall'ultima azione diminuisce do punti sennò tolgo pochi
        if not hasattr(self, 'prev_distance'):
            self.prev_distance = curr_distance
        
        if self.prev_distance is None:
            self.prev_distance = curr_distance
        
        # ════════════════════════════════════════════════════════════
        # 3. GRADIENT TOWARDS APPLE (con moltiplicatore per corner/edge)
        # ════════════════════════════════════════════════════════════
        
        distance_delta = self.prev_distance - curr_distance
        
        # Classifica posizione MELA 
        ax, ay = apple_pos
        grid_size = 20

        # posizione nella griglia della mela 
        apple_dist_left = ax
        apple_dist_right = (grid_size - 1) - ax
        apple_dist_top = ay
        apple_dist_bottom = (grid_size - 1) - ay
        
        # limite entro il quale considero la mela "vicino al muro"
        THRESHOLD = 2
        # se la distanza dai muri è minore sono "attacato al muro"
        # con quali muri sono attacato?
        apple_near_left = apple_dist_left <= THRESHOLD
        apple_near_right = apple_dist_right <= THRESHOLD
        apple_near_top = apple_dist_top <= THRESHOLD
        apple_near_bottom = apple_dist_bottom <= THRESHOLD
        
        # quanti muri ho vicino?
        apple_walls_nearby = sum([apple_near_left, apple_near_right, 
                                apple_near_top, apple_near_bottom])
        
        # sono in un angolo?
        apple_is_corner = (
            (apple_near_left and apple_near_top) or
            (apple_near_left and apple_near_bottom) or
            (apple_near_right and apple_near_top) or
            (apple_near_right and apple_near_bottom)
        )
        
        # se ho solo uno dei muri vicini allora sono su uno dei lati 
        apple_is_edge = (apple_walls_nearby == 1)
        
        # vado a decidere quanti punti assegnare se mi avvicino ad una mela sulla base di dove si trova
        # GRADIENT con moltiplicatore
        if distance_delta > 0: # CI STIAMO AVVICINANDO
            # Ci avviciniamo
            base_approach = 100 * distance_delta
            # nel caso in cui la mela sia in un angolo e mi stia avvicinando prendo tanti punti
            if apple_is_corner:
                approach_reward = base_approach * 8  # 8× per angoli!
            # un po meno nel caso in cui la mela sia su uno dei lati
            elif apple_is_edge:
                approach_reward = base_approach * 3  # 3× per bordi
            # normale nel caso in cui la mela sia nel centro
            else:
                approach_reward = base_approach      # 1× per centro
            
            self.fitness += approach_reward
            
        else: # CI STIAMO ALLONTANANDO
            # vado a togliere molti più punti se mi allontano in modo che il modello non trovi un vantaggio nel girare in torno

            base_penalty = 50 * distance_delta  # Negativo
            
            # visto che voglio spingere sempre lo snake ad andare a mangiare le mele nei bordi tolgo tanti punti se mi allontano da li
            if apple_is_corner:
                penalty = base_penalty * 4  # Penalità 4× maggiore
            # tolgo un po meno se mi allontano da una mela sui bordi
            elif apple_is_edge:
                penalty = base_penalty * 2  # Penalità 2× maggiore
            # ancora meno se la mela è al centro
            else:
                penalty = base_penalty
            
            self.fitness += penalty  # Già negativo
        
        # ════════════════════════════════════════════════════════════
        # 4. PROXIMITY BONUS
        # ════════════════════════════════════════════════════════════
        
        # vado a premiare lo snake se si trova vicino alla mela 
        if curr_distance <= 3: # distanza all'interno della quale do questa ricompensa
            proximity_bonus = (3 - curr_distance) * 400
            
            # Extra bonus se mela è in corner/edge
            if apple_is_corner:
                proximity_bonus *= 3
            elif apple_is_edge:
                proximity_bonus *= 1.5
            
            self.fitness += proximity_bonus
        
        # ════════════════════════════════════════════════════════════
        # 5. LOOP DETECTION CON GAME OVER
        # ════════════════════════════════════════════════════════════

        x, y = snake_head[0]

        # Inizializza tracking se non esiste
        if not hasattr(self, 'visited_positions'):
            self.visited_positions = {}

        if not hasattr(self, 'loop_death'):
            self.loop_death = False

        pos_key = (x, y)

        # Controlla se abbiamo già visitato questa posizione
        if pos_key in self.visited_positions:
            self.visited_positions[pos_key] += 1
            
            # SOGLIA LOOP: 3+ visite alla stessa posizione = LOOP RILEVATO!
            if self.visited_positions[pos_key] >= 3:
                # 💀 PENALITÀ MASSIVA + GAME OVER
                self.fitness -= 20000000
                self.loop_death = True
            
            else:
                # Penalità progressiva normale
                revisit_penalty = -20 * self.visited_positions[pos_key]
                self.fitness += revisit_penalty

        else:
            # Prima visita a questa posizione
            self.visited_positions[pos_key] = 1
            self.fitness += 5  # Bonus esplorazione
        
        # ════════════════════════════════════════════════════════════
        # 6. TIME PRESSURE
        # ════════════════════════════════════════════════════════════
        
        # voglio che non perda tempo in giro ma arrivi a prendere la mela -> dopo un tot inizio a togliere punti
        if not hasattr(self, 'steps_since_last_apple'):
            self.steps_since_last_apple = 0
        
        self.steps_since_last_apple += 1
        
        # Timeout diverso per corner/edge
        if apple_is_corner:
            timeout_threshold = 150  # Più tempo per angoli
        elif apple_is_edge:
            timeout_threshold = 120
        else:
            timeout_threshold = 100
        
        # se il numero di azioni è superiori a quello che ho dato come max allora tolgo punti 
        if self.steps_since_last_apple > timeout_threshold:
            time_penalty = -0.1 * (self.steps_since_last_apple - timeout_threshold) ** 1.5
            self.fitness += time_penalty
        
        # ════════════════════════════════════════════════════════════
        # 7. SURVIVAL PENALTY
        # ════════════════════════════════════════════════════════════
        
        # di default perde punti esistendo
        self.fitness -= 0.5
        
        # ════════════════════════════════════════════════════════════
        # 8. AGGIORNA STATO PRECEDENTE
        # ════════════════════════════════════════════════════════════
        
        self.prev_distance = curr_distance
        

    @staticmethod
    # algoritmi genetici prevedono di mischiare i pesi di due elite per veere cosa si ottiene
    def crossover(parent1, parent2): 
        # Scegliamo un punto di taglio per i pesi (Crossover a punto singolo)
        # Eseguiamo l'operazione per entrambi gli strati della rete
        
        def merge_weights(w1, w2):
            mask = np.random.rand(*w1.shape) > 0.5
            return np.where(mask, w1, w2)
        
        child_w1 = merge_weights(parent1.weights1, parent2.weights1)
        child_w2 = merge_weights(parent1.weights2, parent2.weights2)
        child_w3 = merge_weights(parent1.weights3, parent2.weights3)
        child_w4 = merge_weights(parent1.weights4, parent2.weights4)
        child_w5 = merge_weights(parent1.weights5, parent2.weights5)
        child_w6 = merge_weights(parent1.weights6, parent2.weights6)
        
        return Snake(child_w1, child_w2, child_w3, child_w4, child_w5, child_w6)

    # a questi pesi può essere applicata una mutazione che varia i pesi in modo da uscire da eventuali minimo locali
    def mutate(self, rate):
        # Fondamentale: senza questo l'evoluzione si blocca!
        for w in [self.weights1, self.weights2, self.weights3, self.weights4, self.weights5, self.weights6]:
            if np.random.rand() < rate:
                w += np.random.randn(*w.shape) * 0.1