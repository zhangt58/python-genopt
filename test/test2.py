#!/usr/bin/python

from flame import Machine
from numpy import ndarray

f = open('test.lat', 'r')

m = Machine(f)

fout = open('out.lat', 'w')

mconf = m.conf()
mconf_ks = mconf.keys()

[mconf_ks.remove(i) for i in ['elements', 'name'] if i in mconf_ks]

lines = []
for k in mconf_ks:
    v = mconf[k]
    if isinstance(v, ndarray):
        v = v.tolist()
    if isinstance(v, str):
        v = '"{0}"'.format(v)
    line = '{k} = {v};'.format(k=k, v=v)
    lines.append(line)

fout.writelines('\n'.join(lines))

mconfe = mconf['elements']

eu = []
for i, e in enumerate(mconfe):
    e_n = e['name']
    if e_n not in eu:
        eu.append(e_n)
    else:
        mconfe.pop(i)

elines = []
for e in mconfe:
    p = []
    for k, v in e.items():
        if k not in ['name', 'type']:
            if isinstance(v, ndarray):
                v = v.tolist()
            if isinstance(v, str):
                v = '"{0}"'.format(v)
            p.append('{k} = {v}'.format(k=k, v=v))
    pline = ', '.join(p)

    line = '{n}: {t}, {p}'.format(n=e['name'], t=e['type'], p=pline)

    line = line.strip(', ') + ';'
    elines.append(line)

fout.writelines('\n'.join(elines))

dline = '(' + ', '.join(([e['name'] for e in mconfe])) + ')'

blname = mconf['name']
fout.write('\n{0}: LINE = {1};\n'.format(blname, dline))
fout.write('USE: {0};'.format(blname))

fout.close()
