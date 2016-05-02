import sys

def main(args = None):
    """Main - serves project"""
    if args is None:
        args = sys.argv[1:]

    print("\nWe are in __main__")
    print("This file serves the project\n")

if __name__ == "__main__":
    main()
