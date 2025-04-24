import protonets.data
import protonets.data.fault

def load(opt, splits):
    ds = protonets.data.fault.load(opt)
    return ds
