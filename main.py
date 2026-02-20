import sys

from src.collect_data import start_collection
from src.realtime import classify


def main():
    print("=== AI Yoga Assist ===")
    print("1. Start Data Collection")
    print("2. Start Classification and correction")
    print("q. Quit")

    choice = input("\nSelect an option: ")

    if choice == "1":
        print("\nLaunching Data Collector...")
        start_collection(csv_path="data/poses.csv")
    elif choice == "2":
        print("\nLaunching Classification...")
        classify("./models/pose_classifier.h5","./models/label_encoder.pkl")
    elif choice.lower() == "q":
        print("Goodbye!")
        sys.exit()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
