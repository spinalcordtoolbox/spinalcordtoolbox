__author__ = 'slevy_local'



class Color:
    def __init__(self):
        self.purple = '\033[95m'
        self.cyan = '\033[96m'
        self.darkcyan = '\033[36m'
        self.blue = '\033[94m'
        self.green = '\033[92m'
        self.yellow = '\033[93m'
        self.red = '\033[91m'
        self.bold = '\033[1m'
        self.underline = '\033[4m'
        self.end = '\033[0m'

def progress3d(i, j, k, ni, nj, nk):

    import sys
    color = Color()

    progress = str( 100*((k-1)*(nj-1)*(ni-1) + (j-1)*(ni-1) + +i) / ((nk-1)*(nj-1)*(ni-1)) ) +' %'

    if [i, j, k] == [0, 0, 0]:
        sys.stdout.write(progress)
    else:
        sys.stdout.write(color.bold + '\b\b\b\b\b'+ progress + color.end)

