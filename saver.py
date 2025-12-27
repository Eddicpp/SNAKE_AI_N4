"""
Sistema di salvataggio e caricamento per Snake GA
Compatibile con checkpoint 3-layer (vecchi) e 4-layer (nuovi)
"""
import numpy as np
import json
import os
from datetime import datetime

class SnakeSaver:
    def __init__(self, base_dir="./"):
        self.base_dir = base_dir
        
        # Crea directory
        self.models_dir = os.path.join(base_dir, "saved_models")
        self.logs_dir = os.path.join(base_dir, "logs")
        
        for d in [self.models_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Storia training
        self.history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_score': [],
            'avg_score': [],
            'mutation_rate': [],
            'timestamp': []
        }
        
        self.log_file = os.path.join(self.logs_dir, "training_log.txt")
        
        print(f"💾 SnakeSaver inizializzato in: {base_dir}")
    
    def save_best_agent(self, agent, generation, fitness, score):
        """Salva il miglior agente"""
        filename = f"best_snake_gen_{generation}.npz"
        filepath = os.path.join(self.models_dir, filename)
        
        np.savez_compressed(
            filepath,
            weights1=agent.weights1,
            weights2=agent.weights2,
            weights3=agent.weights3,
            weights4=agent.weights4,
            generation=generation,
            fitness=fitness,
            score=score,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"💾 Salvato: {filename} (Fitness: {fitness:.0f}, Score: {score})")
        
        # Salva anche come "best overall"
        best_overall_path = os.path.join(self.models_dir, "best_snake_overall.npz")
        
        should_save = True
        if os.path.exists(best_overall_path):
            prev_data = np.load(best_overall_path)
            if fitness <= float(prev_data['fitness']):
                should_save = False
        
        if should_save:
            np.savez_compressed(
                best_overall_path,
                weights1=agent.weights1,
                weights2=agent.weights2,
                weights3=agent.weights3,
                weights4=agent.weights4,
                generation=generation,
                fitness=fitness,
                score=score,
                timestamp=datetime.now().isoformat()
            )
            print(f"🏆 NUOVO RECORD! Salvato come best_overall")
    
    def load_best_agent(self, filename="best_snake_overall.npz"):
        """Carica il miglior agente (compatibile con vecchi e nuovi formati)"""
        filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ File non trovato: {filename}")
            return None, None
        
        data = np.load(filepath)
        
        from backSnake import Snake
        
        # ⭐ CHECK FORMATO
        has_4_layers = 'weights4' in data.files
        
        if has_4_layers:
            # ✅ NUOVO FORMATO (4 layer)
            agent = Snake(
                child_w1=data['weights1'],
                child_w2=data['weights2'],
                child_w3=data['weights3'],
                child_w4=data['weights4']
            )
            print(f"  📦 Formato: 4-layer network")
        else:
            # ⚠️ VECCHIO FORMATO (3 layer)
            print(f"  ⚠️  ATTENZIONE: Best snake in formato vecchio (3-layer)")
            print(f"     Non è compatibile con l'architettura corrente (4-layer)")
            print(f"     Creazione nuovo snake random")
            
            agent = Snake()  # Nuovo snake random
            
            metadata = {
                'generation': 0,
                'fitness': 0,
                'score': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            return agent, metadata
        
        metadata = {
            'generation': int(data['generation']),
            'fitness': float(data['fitness']),
            'score': int(data['score']),
            'timestamp': str(data['timestamp'])
        }
        
        print(f"✅ Caricato: {filename}")
        print(f"   Gen {metadata['generation']}, Fitness: {metadata['fitness']:.0f}, Score: {metadata['score']}")
        
        return agent, metadata
    
    def save_checkpoint(self, population, generation, mutation_rate):
        """Salva checkpoint completo"""
        filename = f"checkpoint_gen_{generation}.npz"
        filepath = os.path.join(self.models_dir, filename)
        
        weights1_list = [agent.weights1 for agent in population]
        weights2_list = [agent.weights2 for agent in population]
        weights3_list = [agent.weights3 for agent in population]
        weights4_list = [agent.weights4 for agent in population]
        fitness_list = [agent.fitness for agent in population]
        
        np.savez_compressed(
            filepath,
            weights1_array=np.array(weights1_list),
            weights2_array=np.array(weights2_list),
            weights3_array=np.array(weights3_list),
            weights4_array=np.array(weights4_list),
            fitness_array=np.array(fitness_list),
            generation=generation,
            mutation_rate=mutation_rate,
            timestamp=datetime.now().isoformat(),
            format_version='4-layer'
        )
        
        print(f"📦 Checkpoint salvato: Gen {generation}")
    
    def load_checkpoint(self, generation):
        """Carica checkpoint (compatibile con vecchi e nuovi formati)"""
        filename = f"checkpoint_gen_{generation}.npz"
        filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ Checkpoint non trovato: Gen {generation}")
            return None, None
        
        data = np.load(filepath)
        
        from backSnake import Snake
        
        # ⭐ CONTROLLA FORMATO
        has_4_layers = 'weights4_array' in data.files
        
        if has_4_layers:
            # ✅ NUOVO FORMATO (4 layer: 24→24→16→12→4)
            print(f"  📦 Checkpoint formato: 4-layer network")
            
            w1_array = data['weights1_array']
            w2_array = data['weights2_array']
            w3_array = data['weights3_array']
            w4_array = data['weights4_array']
            
            popolazione = []
            for i in range(len(w1_array)):
                agent = Snake(
                    child_w1=w1_array[i],
                    child_w2=w2_array[i],
                    child_w3=w3_array[i],
                    child_w4=w4_array[i]
                )
                popolazione.append(agent)
            
            mutation_rate = 1
            print(f"✅ Checkpoint caricato: Gen {generation}, Pop: {len(popolazione)}")
            
            return popolazione, mutation_rate
        
        else:
            # ⚠️ VECCHIO FORMATO (3 layer)
            print(f"  ⚠️  ATTENZIONE: Checkpoint in formato vecchio (3-layer)")
            
            w1_array = data['weights1_array']
            w2_array = data['weights2_array']
            w3_array = data['weights3_array']
            
            # Mostra architettura vecchia
            old_shape1 = w1_array[0].shape
            old_shape2 = w2_array[0].shape
            old_shape3 = w3_array[0].shape
            
            print(f"     Architettura vecchia: {old_shape1} → {old_shape2} → {old_shape3}")
            print(f"     Architettura nuova:   (24,24) → (24,16) → (16,12) → (12,4)")
            print(f"")
            print(f"  ❌ INCOMPATIBILITÀ: I pesi non sono trasferibili")
            print(f"  🔄 Creazione popolazione random (checkpoint ignorato)")
            print(f"")
            print(f"  💡 SUGGERIMENTO: Elimina i checkpoint vecchi con:")
            print(f"     python cleanup.py")
            print(f"     E riparti il training da zero")
            print(f"")
            
            # Crea popolazione random
            pop_size = len(w1_array)
            popolazione = [Snake() for _ in range(pop_size)]
            
            # Usa mutation rate dal checkpoint
            mutation_rate = float(data['mutation_rate'])
            
            print(f"⚠️  Checkpoint ignorato: Creati {len(popolazione)} snakes random")
            
            return popolazione, mutation_rate
    
    def log_generation(self, generation, population, mutation_rate):
        """Logga statistiche generazione"""
        fitnesses = [agent.fitness for agent in population]
        scores = [agent.apples_eaten for agent in population]
        
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        best_score = max(scores)
        avg_score = np.mean(scores)
        
        # Aggiungi a history
        self.history['generations'].append(generation)
        self.history['best_fitness'].append(best_fitness)
        self.history['avg_fitness'].append(avg_fitness)
        self.history['best_score'].append(best_score)
        self.history['avg_score'].append(avg_score)
        self.history['mutation_rate'].append(mutation_rate)
        self.history['timestamp'].append(datetime.now().isoformat())
        
        # Log su file
        log_msg = (
            f"Gen {generation:4d} | "
            f"Best Fit: {best_fitness:8.0f} | "
            f"Avg Fit: {avg_fitness:8.0f} | "
            f"Best Score: {best_score:3d} | "
            f"Avg Score: {avg_score:5.2f} | "
            f"μ: {mutation_rate:.6f}\n"
        )
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg)
    
    def save_history(self):
        """Salva history come JSON"""
        history_file = os.path.join(self.models_dir, "training_history.json")
        
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"📊 History salvata")
    
    def load_history(self):
        """Carica history da JSON"""
        history_file = os.path.join(self.models_dir, "training_history.json")
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.history = json.load(f)
            print(f"✅ History caricata: {len(self.history['generations'])} generazioni")
        else:
            print("⚠️ Nessuna history trovata")

    def get_latest_checkpoint(self):
        """Trova il checkpoint più recente"""
        if not os.path.exists(self.models_dir):
            return None
        
        checkpoint_files = [
            f for f in os.listdir(self.models_dir) 
            if f.startswith("checkpoint_gen_") and f.endswith(".npz")
        ]
        
        if not checkpoint_files:
            return None
        
        # Estrai numero generazione da ogni file
        generations = []
        for f in checkpoint_files:
            try:
                gen_num = int(f.replace("checkpoint_gen_", "").replace(".npz", ""))
                generations.append(gen_num)
            except:
                continue
        
        if not generations:
            return None
        
        latest_gen = max(generations)
        return latest_gen
    
    def list_checkpoints(self):
        """Lista tutti i checkpoint disponibili"""
        if not os.path.exists(self.models_dir):
            return []
        
        checkpoint_files = [
            f for f in os.listdir(self.models_dir) 
            if f.startswith("checkpoint_gen_") and f.endswith(".npz")
        ]
        
        checkpoints = []
        for f in checkpoint_files:
            try:
                gen_num = int(f.replace("checkpoint_gen_", "").replace(".npz", ""))
                
                # Carica metadata
                filepath = os.path.join(self.models_dir, f)
                data = np.load(filepath)
                
                # ⭐ RILEVA FORMATO
                format_type = "4-layer" if 'weights4_array' in data.files else "3-layer (OLD)"
                
                checkpoints.append({
                    'generation': gen_num,
                    'filename': f,
                    'mutation_rate': float(data['mutation_rate']),
                    'timestamp': str(data['timestamp']),
                    'format': format_type
                })
            except:
                continue
        
        # Ordina per generazione
        checkpoints.sort(key=lambda x: x['generation'])
        return checkpoints
    
    def cleanup_old_checkpoints(self):
        """Elimina checkpoint in formato vecchio (3-layer)"""
        if not os.path.exists(self.models_dir):
            print("❌ Nessuna directory saved_models trovata")
            return
        
        checkpoint_files = [
            f for f in os.listdir(self.models_dir) 
            if f.startswith("checkpoint_gen_") and f.endswith(".npz")
        ]
        
        deleted = 0
        
        for f in checkpoint_files:
            filepath = os.path.join(self.models_dir, f)
            
            try:
                data = np.load(filepath)
                
                # Se non ha weights4_array, è vecchio
                if 'weights4_array' not in data.files:
                    os.remove(filepath)
                    deleted += 1
                    print(f"  🗑️  Eliminato: {f} (formato vecchio)")
            
            except Exception as e:
                print(f"  ⚠️  Errore con {f}: {e}")
        
        if deleted > 0:
            print(f"\n✅ Eliminati {deleted} checkpoint vecchi")
        else:
            print("\n✅ Nessun checkpoint vecchio trovato")
    
    # ⭐ NUOVO METODO - Aggiungi qui sotto
    def clear_all_data(self):
        """Elimina TUTTI i checkpoint e resetta la history"""
        print("\n🗑️  CANCELLAZIONE COMPLETA IN CORSO...")
        
        deleted_checkpoints = 0
        deleted_best = 0
        
        # Elimina tutti i checkpoint
        if os.path.exists(self.models_dir):
            all_files = os.listdir(self.models_dir)
            
            for f in all_files:
                filepath = os.path.join(self.models_dir, f)
                
                try:
                    # Elimina checkpoint
                    if f.startswith("checkpoint_gen_") and f.endswith(".npz"):
                        os.remove(filepath)
                        deleted_checkpoints += 1
                        print(f"  🗑️  {f}")
                    
                    # Elimina best snakes
                    elif f.startswith("best_snake_") and f.endswith(".npz"):
                        os.remove(filepath)
                        deleted_best += 1
                        print(f"  🗑️  {f}")
                    
                    # Elimina history JSON
                    elif f == "training_history.json":
                        os.remove(filepath)
                        print(f"  🗑️  {f}")
                
                except Exception as e:
                    print(f"  ⚠️  Errore eliminazione {f}: {e}")
        
        # Elimina log file
        if os.path.exists(self.log_file):
            try:
                os.remove(self.log_file)
                print(f"  🗑️  training_log.txt")
            except Exception as e:
                print(f"  ⚠️  Errore eliminazione log: {e}")
        
        # Reset history in memoria
        self.history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_score': [],
            'avg_score': [],
            'mutation_rate': [],
            'timestamp': []
        }
        
        print(f"\n✅ CANCELLAZIONE COMPLETATA:")
        print(f"   - Checkpoint eliminati: {deleted_checkpoints}")
        print(f"   - Best snakes eliminati: {deleted_best}")
        print(f"   - History resettata")
        print(f"\n🆕 Pronto per nuovo training da zero\n")