import pygame
import random
import numpy as np

# Impostazioni Gioco
CELL_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WIDTH = GRID_WIDTH * CELL_SIZE
HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 60 # Velocità di rendering (puoi aumentarla per "velocizzare" l'evoluzione)

class SnakeGame:
    def __init__(self, timeout=100000):  # ⭐ Aggiungi parametro timeout
        self.timeout = timeout  # ⭐ Salva come attributo
        self.reset()

    def reset(self):
        # Stato iniziale
        self.snake_pos = [[10, 10], [10, 11], [10, 12]]
        self.direction = "UP"
        self.food_pos = self._place_food()
        self.score = 0
        self.steps = 0
        self.steps_since_last_apple = 0  # ⭐ Aggiungi questo
        self.dead = False

    def _place_food(self):
        # Piazza il cibo in un punto casuale non occupato dal corpo
        while True:
            pos = [random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)]
            if pos not in self.snake_pos:
                return pos

    def _update_direction(self, action):
        if action == 0 and self.direction != "DOWN":   # SU
            self.direction = "UP"
        elif action == 1 and self.direction != "UP": # GIÙ
            self.direction = "DOWN"
        elif action == 2 and self.direction != "RIGHT": # SINISTRA
            self.direction = "LEFT"
        elif action == 3 and self.direction != "LEFT": # DESTRA
            self.direction = "RIGHT" 

    def _check_collision(self, head):
        # 1. Collisione con i MURI
        # head[0] è la X, head[1] è la Y
        if head[0] < 0 or head[0] >= GRID_WIDTH:
            return True # Ha colpito il muro sinistro o destro
        
        if head[1] < 0 or head[1] >= GRID_HEIGHT:
            return True # Ha colpito il muro superiore o inferiore

        # 2. Collisione con il CORPO
        # Controlliamo se la nuova testa coincide con un qualsiasi pezzo del corpo
        # Nota: usiamo snake_pos[1:] per evitare di controllare la testa contro se stessa
        for position in self.snake_pos[1:]:
            if head == position:
                return True # Si è morso la coda
                
        return False # Non ha colpito nulla, è salvo!

    # esgue fisicamente oa azione di spostare lo snake nel gioco
    def step(self, action):
        """
        Esegue un'azione e restituisce lo stato aggiornato.
        """
        ate_apple = False
        self.steps += 1
        self.steps_since_last_apple += 1  # ⭐ Incrementa anche questo
        
        # 1. Cambia direzione
        self._update_direction(action)
        
        # 2. Muovi la testa
        new_head = list(self.snake_pos[0])
        if self.direction == "UP":    new_head[1] -= 1
        if self.direction == "DOWN":  new_head[1] += 1
        if self.direction == "LEFT":  new_head[0] -= 1
        if self.direction == "RIGHT": new_head[0] += 1
        
        # 3. Controlla collisioni O TIMEOUT
        # ⭐ USA self.timeout invece di valore fisso
        if self._check_collision(new_head) or self.steps_since_last_apple > self.timeout:
            self.dead = True
            return self.dead, ate_apple

        self.snake_pos.insert(0, new_head)

        # 4. Mangia cibo?
        if new_head == self.food_pos:
            ate_apple = True
            self.score += 1
            self.food_pos = self._place_food()
            self.steps_since_last_apple = 0  # ⭐ Reset quando mangia
        else:
            self.snake_pos.pop()
            
        return self.dead, ate_apple

    # mi serve per ricavare informazioni sulla posizione nello spazio dello snake 
    # sono informazioni che poi andrò ad utilizzare all'interno della reward function per migliorare lo score dello snake nel caso in cui si trovi in una bella situazione
    # in questo modo non premio il serpente solamente per prendere la mela ma anche per l'AVVICINARSI ALLA MELA E ALLONTANARSI DAL SUO CORPO/MURO
    def getInfo(self):
        # Definizione delle 8 direzioni (dx, dy)
        # N, NE, E, SE, S, SW, W, NW
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1), 
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        
        # Inizializziamo la matrice 8x3 con zeri
        # Righe: direzioni, Colonne: [dist_mela, dist_corpo, dist_muro]
        vision_matrix = np.zeros((8, 3))
        
        head_x, head_y = self.snake_pos[0]

        for i, (dx, dy) in enumerate(directions):
            dist_to_apple = 0
            dist_to_body = 0
            dist_to_wall = 0
            
            curr_x = head_x + dx
            curr_y = head_y + dy
            distance = 1 # Partiamo dalla prima cella adiacente
            
            found_apple = False
            found_body = False
            
            # Lanciamo il "raggio" finché non usciamo dai bordi (muro)
            while 0 <= curr_x < GRID_WIDTH and 0 <= curr_y < GRID_HEIGHT:
                
                # 1. Controllo Mela
                if not found_apple and [curr_x, curr_y] == self.food_pos:
                    dist_to_apple = 1 / distance # Usiamo l'inverso per aiutare la rete
                    found_apple = True
                
                # 2. Controllo Corpo
                if not found_body and [curr_x, curr_y] in self.snake_pos:
                    dist_to_body = 1 / distance
                    found_body = True
                
                # Avanziamo nel raggio
                curr_x += dx
                curr_y += dy
                distance += 1
            
            # 3. Distanza dal Muro (calcolata alla fine del loop)
            # La distanza totale percorsa prima di uscire è il limite del muro
            dist_to_wall = 1 / distance
            
            # Riempiamo la riga della matrice
            vision_matrix[i] = [dist_to_apple, dist_to_body, dist_to_wall]
            
        return vision_matrix
    
    def render(self, screen, generation=0, fitness=0):
        # 1. Sfondo nero
        screen.fill((0, 0, 0))

        # 2. Disegna la mela (Rossa)
        # Moltiplichiamo la coordinata della cella per la dimensione in pixel
        apple_rect = pygame.Rect(
            self.food_pos[0] * CELL_SIZE, 
            self.food_pos[1] * CELL_SIZE, 
            CELL_SIZE - 1, 
            CELL_SIZE - 1
        )
        pygame.draw.rect(screen, (255, 0, 0), apple_rect)

        # 3. Disegna il corpo dello snake (Verde)
        for i, pos in enumerate(self.snake_pos):
            # La testa è di un verde leggermente diverso
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            snake_rect = pygame.Rect(
                pos[0] * CELL_SIZE, 
                pos[1] * CELL_SIZE, 
                CELL_SIZE - 1, 
                CELL_SIZE - 1
            )
            pygame.draw.rect(screen, color, snake_rect)

        # 4. Disegna le informazioni (Testo)
        # Inizializza il font se non l'hai fatto (o fallo nell'__init__)
        font = pygame.font.SysFont("Arial", 18)
        score_text = font.render(f"Gen: {generation}  Fitness: {int(fitness)}  Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
