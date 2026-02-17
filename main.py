import sys

from src.collect_data import start_collection


def main():
    print("=== AI Yoga Assist ===")
    print("1. Start Data Collection")
    print("q. Quit")

    choice = input("\nSelect an option: ")

    if choice == "1":
        # This executes your collect_data.py script
        print("\nLaunching Data Collector...")
        start_collection(csv_path="data/poses_raw.csv")
    elif choice.lower() == "q":
        print("Goodbye!")
        sys.exit()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
