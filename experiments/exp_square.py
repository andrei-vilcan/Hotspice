import os
os.environ['HOTSPICE_USE_GPU'] = 'False'

import hotspice


# si = hotspice.ASI.IP_Square(a=2e-9, n=9)
# si = hotspice.ASI.IP_Square_Open(a=2e-9, n=10)
si = hotspice.ASI.IP_Square_Open_Shifted(a=2e-9, n=3)


hotspice.gui.show(si)
