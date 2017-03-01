from phyapps import flameutils
from flame import Machine

m = Machine(open('test.lat'))

flameutils.generate_latfile(m, 'out2.lat')
