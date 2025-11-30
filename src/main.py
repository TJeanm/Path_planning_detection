from pathlib import Path
from utils.track_utils import load_track, compute_world_bounds
from ui.process_pygame import process_pygame

# Note : On n'importe plus de PathProcessor ici.
# La logique de calcul est maintenant locale et gérée à l'intérieur de process_pygame.

TRACKS = ["small_track.csv", "hairpins_increasing_difficulty.csv", "peanut.csv"]

if __name__ == "__main__":
    # Setup paths
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    print("Choose a track:")
    print("1 - small_track")
    print("2 - hairpins_increasing_difficulty")
    print("3 - peanut")

    try:
        choice_str = input("Your choice (1/2/3): ").strip()
        track_choice = int(choice_str)

        if 1 <= track_choice <= 3:
            selected_track = data_dir / TRACKS[track_choice - 1]

            # 1. Load Data
            print(f"Loading {selected_track}...")
            cones = load_track(selected_track)

            # On calcule juste les limites pour la caméra
            world_bounds = compute_world_bounds(cones)

            # 2. Launch Visualization directly
            # On ne calcule plus le chemin global ('path') ici.
            # On passe path=None car la voiture va le calculer elle-même image par image.
            print("Lancement de la simulation locale (RRT* temps réel)...")

            process_pygame(selected_track, cones, world_bounds, path=None)

        else:
            print("Invalid choice.")

    except ValueError:
        print("Please enter a valid number.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")