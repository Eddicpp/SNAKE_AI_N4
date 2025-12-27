"""
Utility per pulire checkpoint incompatibili
"""
from saver import SnakeSaver

def main():
    print("\n" + "="*70)
    print("🧹 CLEANUP CHECKPOINT VECCHI")
    print("="*70)
    
    saver = SnakeSaver()
    
    # Mostra checkpoint
    checkpoints = saver.list_checkpoints()
    
    if not checkpoints:
        print("\n✅ Nessun checkpoint trovato")
        return
    
    print(f"\n📦 Checkpoint trovati:")
    old_count = 0
    new_count = 0
    
    for cp in checkpoints:
        status = "⚠️ " if cp['format'] == "3-layer (OLD)" else "✅"
        print(f"  {status} Gen {cp['generation']:4d} | {cp['format']:15s} | {cp['timestamp'][:19]}")
        
        if cp['format'] == "3-layer (OLD)":
            old_count += 1
        else:
            new_count += 1
    
    print(f"\nTotale: {len(checkpoints)} checkpoint")
    print(f"  - Vecchi (3-layer): {old_count}")
    print(f"  - Nuovi (4-layer):  {new_count}")
    
    if old_count > 0:
        print("\n" + "="*70)
        print("⚠️  ATTENZIONE: Trovati checkpoint incompatibili")
        print("="*70)
        print("I checkpoint vecchi (3-layer) non sono compatibili con")
        print("l'architettura attuale (4-layer) e causeranno errori.")
        print()
        
        choice = input("Vuoi eliminarli? [s/N]: ")
        
        if choice.lower() == 's':
            print("\n🗑️  Eliminazione in corso...")
            saver.cleanup_old_checkpoints()
        else:
            print("\n❌ Operazione annullata")
            print("\n💡 SUGGERIMENTO: Se vuoi fare resume, elimina prima i vecchi checkpoint")
    else:
        print("\n✅ Tutti i checkpoint sono compatibili!")

if __name__ == "__main__":
    main()