#file main per eseguire lo snake
import random
import pygame
import numpy as np
import sys

# creo gli oggetti necessari per far funzionare il programma
from backSnake import Snake
from snakeGame import SnakeGame
from saver import SnakeSaver

# caratteristiche dell'ambiente
CELL_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WIDTH = GRID_WIDTH * CELL_SIZE
HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 60

# ⭐ CONFIGURAZIONE MODALITÀ
FAST_MODE = True
RENDER_EVERY_N_GENS = 1

# ⭐ CONFIGURAZIONE SALVATAGGIO
SAVE_BEST_EVERY = 10
SAVE_CHECKPOINT_EVERY = 50

# ⭐ CONFIGURAZIONE TRAINING
MAX_GENERATIONS = 1000000 # <- da modificare per aumentare generazioni di timeOut
POPULATION_SIZE = 1000 

# ⭐ CONFIGURAZIONE TIMEOUT DINAMICO
BASE_TIMEOUT = 2000000  # Timeout iniziale
TIMEOUT_INCREMENT = 0  # Incremento per generazione

# --- INIZIALIZZAZIONE ---
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake AI Evolution")
clock = pygame.time.Clock()

# ⭐ SAVER
saver = SnakeSaver(base_dir="./")

# ════════════════════════════════════════════════════════════
# RESUME O NUOVO TRAINING?
# ════════════════════════════════════════════════════════════

# interfaccia classica per caricare i dati salvati
latest_checkpoint = saver.get_latest_checkpoint()

if latest_checkpoint is not None:
    print("\n" + "="*70)
    print("🔍 CHECKPOINT TROVATI")
    print("="*70)
    
    checkpoints = saver.list_checkpoints()
    
    print(f"\n📦 Checkpoint disponibili:")
    for cp in checkpoints[-5:]:  # Mostra ultimi 5
        print(f"  Gen {cp['generation']:4d} | μ: {cp['mutation_rate']:.6f} | {cp['timestamp'][:19]}")
    
    print(f"\n💾 Checkpoint più recente: Gen {latest_checkpoint}")
    
    # ⭐ CHIEDI ALL'UTENTE
    print("\n" + "="*70)
    print("Vuoi continuare il training esistente?")
    print("="*70)
    print("  [R] - Resume da Gen", latest_checkpoint)
    print("  [N] - Nuovo training (cancella progressi)")
    print("  [C] - Scegli checkpoint specifico")
    print("  [ESC] - Esci")
    print("="*70)
    
    waiting_for_choice = True
    choice = None
    
    while waiting_for_choice:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    choice = 'resume'
                    waiting_for_choice = False
                elif event.key == pygame.K_n:
                    choice = 'new'
                    saver.clear_all_data()
                    waiting_for_choice = False
                elif event.key == pygame.K_c:
                    choice = 'choose'
                    waiting_for_choice = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        pygame.time.wait(100)
    
    # ⭐ GESTISCI SCELTA
    if choice == 'resume':
        print(f"\n🔄 Caricamento checkpoint Gen {latest_checkpoint}...")
        popolazione, rate = saver.load_checkpoint(latest_checkpoint)
        generazione = latest_checkpoint
        saver.load_history()
        
        if popolazione is None:
            print("❌ Errore caricamento! Riparto da zero.")
            popolazione = [Snake() for _ in range(POPULATION_SIZE)]
            generazione = 0
            rate = 1.0
        else:
            print(f"✅ Resume da Gen {generazione}")
            print(f"   Mutation rate: {rate:.6f}")
            print(f"   Popolazione: {len(popolazione)} snakes")
    
    elif choice == 'choose':
        print("\n📦 Scegli checkpoint:")
        for i, cp in enumerate(checkpoints):
            print(f"  [{i}] Gen {cp['generation']}")
        
        try:
            idx = int(input("\nInserisci numero: "))
            selected_gen = checkpoints[idx]['generation']
            
            print(f"\n🔄 Caricamento checkpoint Gen {selected_gen}...")
            popolazione, rate = saver.load_checkpoint(selected_gen)
            generazione = selected_gen
            saver.load_history()
            
            print(f"✅ Resume da Gen {generazione}")
        except:
            print("❌ Input invalido, riparto da zero")
            popolazione = [Snake() for _ in range(POPULATION_SIZE)]
            generazione = 0
            rate = 1.0
    
    else:  # new
        print("\n🆕 Nuovo training - I checkpoint esistenti verranno sovrascritti")
        popolazione = [Snake() for _ in range(POPULATION_SIZE)]
        generazione = 0
        rate = 1.0
        # Reset history
        saver.history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_score': [],
            'avg_score': [],
            'mutation_rate': [],
            'timestamp': []
        }

else:
    # Nessun checkpoint trovato
    print("\n" + "="*70)
    print("🆕 NUOVO TRAINING")
    print("="*70)
    popolazione = [Snake() for _ in range(POPULATION_SIZE)]
    generazione = 0
    rate = 1.0

# ════════════════════════════════════════════════════════════
# CALCOLA EPSILON PER DECAY CORRETTO
# ════════════════════════════════════════════════════════════

# Se abbiamo fatto resume, epsilon deve essere aggiustato
epsilon = 0.995
min_rate = 0.05

print("\n" + "="*70)
print("🐍 SNAKE AI TRAINING")
print("="*70)
print(f"Generazione start: {generazione}")
print(f"Generazione target: {MAX_GENERATIONS}")
print(f"Popolazione: {POPULATION_SIZE}")
print(f"Mutation rate: {rate:.6f}")
print(f"Modalità: {'⚡ VELOCE (no render)' if FAST_MODE else '🎬 VISUAL (con render)'}")
print("="*70)
print("\n⌨️  CONTROLLI:")
print("  V - Toggle fast/visual mode")
print("  S - Salva ora")
print("  ESC - Esci e salva")
print("="*70 + "\n")

# Aspetta un secondo per leggere
pygame.time.wait(1000)

# ⭐ MAIN LOOP
while generazione < MAX_GENERATIONS:
    generazione += 1
    print(f'\n{"="*70}')
    print(f'GENERAZIONE {generazione}/{MAX_GENERATIONS}')
    print(f'{"="*70}')
    
    # ════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ════════════════════════════════════════════════════════════
    
    for i, snake in enumerate(popolazione):
        snake.reset()
        
        # ⭐ TIMEOUT DINAMICO: aumenta con la generazione
        current_timeout = BASE_TIMEOUT + (TIMEOUT_INCREMENT * generazione)
        game = SnakeGame(timeout=current_timeout)
        
        state = game.getInfo()
        
        should_render = False
        
        if not FAST_MODE:
            should_render = (i == 0)
        else:
            if generazione % RENDER_EVERY_N_GENS == 0:
                should_render = (i == 0)
        
        while not game.dead:
            action = snake.get_move(state.flatten())
            died, apple_just_eaten = game.step(action)
            state = game.getInfo()
            snake.reward_function(state, apple_just_eaten, died, game.snake_pos, game.food_pos)

            if snake.loop_death:
                game.dead = True
            
            if should_render:
                game.render(screen, generazione, snake.fitness)
                pygame.display.flip()
                
                if not FAST_MODE:
                    clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n⏹️  Chiusura richiesta...")
                    saver.save_best_agent(popolazione[0], generazione, popolazione[0].fitness, popolazione[0].apples_eaten)
                    saver.save_checkpoint(popolazione, generazione, rate)
                    saver.save_history()
                    pygame.quit()
                    sys.exit()
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        FAST_MODE = not FAST_MODE
                        mode_name = "⚡ VELOCE" if FAST_MODE else "🎬 VISUAL"
                        print(f"\n  🔄 Modalità cambiata: {mode_name}\n")
                    
                    elif event.key == pygame.K_s:
                        print(f"\n  💾 Salvataggio manuale...")
                        saver.save_best_agent(popolazione[0], generazione, popolazione[0].fitness, popolazione[0].apples_eaten)
                        saver.save_checkpoint(popolazione, generazione, rate)
                        saver.save_history()
                        print(f"  ✅ Salvato!\n")
                    
                    elif event.key == pygame.K_ESCAPE:
                        print("\n⏹️  ESC premuto - Salvataggio finale...")
                        saver.save_best_agent(popolazione[0], generazione, popolazione[0].fitness, popolazione[0].apples_eaten)
                        saver.save_checkpoint(popolazione, generazione, rate)
                        saver.save_history()
                        pygame.quit()
                        sys.exit()
    
    # ════════════════════════════════════════════════════════════
    # SELEZIONE & STATISTICHE
    # ════════════════════════════════════════════════════════════
    
    popolazione.sort(key=lambda s: s.fitness, reverse=True)
    
    scores = [s.apples_eaten for s in popolazione]
    fitnesses = [s.fitness for s in popolazione]
    
    if not FAST_MODE or generazione % RENDER_EVERY_N_GENS == 0:
        print(f"\n📊 DISTRIBUZIONE SCORE:")
        for threshold in [0, 1, 3, 5, 8, 10, 12, 15]:
            count = sum(1 for s in scores if s >= threshold)
            percentage = (count / len(scores)) * 100
            print(f"  {threshold:2d}+ mele: {count:3d} snakes ({percentage:5.1f}%)")
        
        print(f"\n🏆 TOP 10 SNAKES:")
        for i in range(min(10, len(popolazione))):
            print(f"  #{i+1}: Score={popolazione[i].apples_eaten:2d}, Fitness={popolazione[i].fitness:8.0f}")
        
        print(f"\n📈 FITNESS STATS:")
        print(f"  Best:    {max(fitnesses):8.0f}")
        print(f"  Average: {np.mean(fitnesses):8.0f}")
        print(f"  Std Dev: {np.std(fitnesses):8.0f}")
        
        print(f"\n🧬 DIVERSITY:")
        top10_scores = [s.apples_eaten for s in popolazione[:10]]
        print(f"  Top 10 scores: {top10_scores}")
        print(f"  Unique: {len(set(top10_scores))}/10")
        
        print(f"\n⚙️  Mutation rate: {rate:.6f}")
        print(f"{'='*70}\n")
    else:
        print(f"🏆 Best: Score={max(scores)}, Fitness={max(fitnesses):.0f} | Avg: {np.mean(scores):.2f}")
    
    saver.log_generation(generazione, popolazione, rate)
    
    # ════════════════════════════════════════════════════════════
    # SALVATAGGIO AUTOMATICO
    # ════════════════════════════════════════════════════════════
    
    if generazione % SAVE_BEST_EVERY == 0:
        saver.save_best_agent(
            popolazione[0],
            generazione,
            max(fitnesses),
            max(scores)
        )
    
    if generazione % SAVE_CHECKPOINT_EVERY == 0:
        saver.save_checkpoint(popolazione, generazione, rate)
        saver.save_history()
    
    # ════════════════════════════════════════════════════════════
    # EVOLUZIONE
    # ════════════════════════════════════════════════════════════
    
    nuova_popolazione = []
    nuova_popolazione.extend(popolazione[:10])
    
    for j in range(POPULATION_SIZE - 10):
        primo = random.randint(0, 9)
        secondo = primo
        while primo == secondo:
            secondo = random.randint(0, 9)
        
        figlio = Snake.crossover(nuova_popolazione[primo], nuova_popolazione[secondo])
        figlio.mutate(rate)
        nuova_popolazione.append(figlio)
    
    popolazione = nuova_popolazione
    
    if rate > min_rate:
        rate = rate * epsilon

# ════════════════════════════════════════════════════════════
# FINE TRAINING
# ════════════════════════════════════════════════════════════

# ⭐ RICALCOLA FITNESS E SCORES (perché fuori dal loop)
popolazione.sort(key=lambda s: s.fitness, reverse=True)
final_fitnesses = [s.fitness for s in popolazione]
final_scores = [s.apples_eaten for s in popolazione]

print("\n" + "="*70)
print("🏁 TRAINING COMPLETATO!")
print("="*70)
print(f"Generazioni: {generazione}")
print(f"Best fitness: {max(final_fitnesses):.0f}")
print(f"Best score: {max(final_scores)}")
print("="*70 + "\n")

# ⭐ USA VARIABILI RICALCOLATE
saver.save_best_agent(
    popolazione[0], 
    generazione, 
    max(final_fitnesses), 
    max(final_scores)
)
saver.save_checkpoint(popolazione, generazione, rate)
saver.save_history()

pygame.quit()