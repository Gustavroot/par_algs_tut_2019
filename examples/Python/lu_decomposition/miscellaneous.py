import numpy as np
import resource

def read_inp_parms():

    #Extracting params from input_params.txt
    inp_params = dict()
    inp_file   = open("input_params.txt", "r")
    for line in inp_file:
        if len(line)>1 and line[0]!='#':
            str_buf = line.split("=")
            str_buf = [x.strip() for x in str_buf]

            #Check that params have the desired input form
            assert ((len(str_buf)<3) and not any([x.count(' ')>0 for x in str_buf])), \
                    "Params must be specified in the form 'PARAM_NAME = PARAM_VALUE', no spaces"

            #Storing the input params in the <inp_params> dictionary
            inp_params[str_buf[0]] = str_buf[1]

    return inp_params


def init_random():
    #Disable the following line to have a dynamic random
    np.random.seed(123)


def resources_usage(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                (usage[2]*resource.getpagesize())/1000000.0 )
