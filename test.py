import sys

def run():
    print("run")

def train():
    print("train")

if (sys.argv[1] == '-t'):
    train()
elif (sys.argv[1] == '-r'):
    run()
else:
    print("Aucun arguments")
