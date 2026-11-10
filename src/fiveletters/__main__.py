from .cli import main

if __name__ == "__main__":  # guard: worker processes on Windows re-import this module
    main()
