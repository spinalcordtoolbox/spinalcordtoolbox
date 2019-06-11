
import sys, os
import imageio
import fnmatch

def main(args=None):

    if args is None:
        args = sys.argv[1:]

    path_data = args[0]

    images = []

    filenames = fnmatch.filter(os.listdir(path_data), "fig*ElliTest*")

    for filename in filenames:
        images.append(imageio.imread(path_data + "/" + filename))
    imageio.mimsave(path_data + '/Aplot.gif', images, fps=3)

def memory_limit():
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


if __name__ == '__main__':

    if sys.gettrace() is None:
        main()
    else:
        memory_limit()  # Limitates maximun memory usage to half
        try:
            main()
        except MemoryError:
            sys.stderr.write('\n\nERROR: Memory Exception\n')
            sys.exit(1)
