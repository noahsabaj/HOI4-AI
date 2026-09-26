from .cli import main

# Guarded: a worker process of a command's pool (dagger-label --jobs) imports this module
# again, and must not run the command a second time.
if __name__ == "__main__":
    main()
