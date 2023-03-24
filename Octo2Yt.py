#import yt
import numpy as np
def octo2yt(filename):
    import numpy as np
    import h5py
    import sys

    #filename = '/home/shiber/simulations/sphere_ref5_1grid/final.silo'
    print('\nLoading Octo-Tiger simulation into Yt - by S. Shiber')
    print('------------------------------------------------------')

    print('\nopening file ' + filename + '...\n')
    f = h5py.File(filename, "r")

    print('opened succesfully!\n')
    print('reading headers...')
    hdf5_items = list(f.items())

    hdf5_data = f['.silo']
    leaf_count = f['leaf_count'][0]

    hdf5_fields = []
    fields_order = []

    xscale = f['xscale'][0]
    sim_time = f['time'][0]

    for i in range(0, len(hdf5_items)):
        if (type(hdf5_items[i][1]) == h5py._hl.datatype.Datatype):
            attrs = f[hdf5_items[i][0]].attrs
            if (attrs['silo_type'] == 521):
                hdf5_fields.append(hdf5_items[i][0])
                pos = attrs['silo'][-2]
                pos = pos[pos.find('#')+1:]
                print(i, hdf5_fields, pos)
                fields_order.append(int(pos))

    field_round = len(hdf5_fields) + 3
    number_of_fields = len(hdf5_fields)

    hdf5_fields = np.array(hdf5_fields)[np.argsort(np.array(fields_order))]

    total_nodes = f['node_count'][0]
    num = 1
    while num*8 < total_nodes:
        num *=8
        
    grid_size = int(round(num**(1.0/3))*8)

    x0 = y0 = z0 = -xscale
    delx = dely = delz = 2 * xscale / grid_size

    phys_fields = np.zeros((number_of_fields,grid_size, grid_size, grid_size))

    print('headers have been read successfully!')
    print('\nbox size: ', xscale, ' resolution: ', grid_size)
    print('will read the following fields:')
    print(hdf5_fields,'\n')

    print('reading... (from overall', leaf_count, 'leaves)')
    for leaf in range (0, leaf_count):
        cur_field = leaf * field_round + 1
        print("\033[K", float("{0:.1f}".format(leaf * 100.0 / leaf_count)), "%\r", end=' ')
        sys.stdout.flush()
        z_vec = hdf5_data['#'+str(cur_field).zfill(6)]
        cur_field += 1
        y_vec = hdf5_data['#'+str(cur_field).zfill(6)]
        cur_field += 1
        x_vec = hdf5_data['#'+str(cur_field).zfill(6)]
        for field in range (0, number_of_fields):
            cur_field += 1
            dens = hdf5_data['#'+str(cur_field).zfill(6)]

            nx = int((x_vec[0] - x0) / delx)
            ny = int((y_vec[0] - y0) / dely)
            nz = int((z_vec[0] - z0) / delz)
            phys_fields[field][nx:nx+dens.shape[0],ny:ny+dens.shape[1],nz:nz+dens.shape[2]] = dens
        #    or i in range (0, dens.shape[0]):
        #        for j in range (0, dens.shape[1]):
        #            for k in range (0, dens.shape[2]):
        #                nx = int((x_vec[i] - x0) / delx)
        #                ny = int((y_vec[j] - y0) / dely)
        #                nz = int((z_vec[k] - z0) / delz)
            #            if (dens.shape[0] > 16):
            #                print x_vec[i], '->', nx, ',', y_vec[j], '->', ny, ',', z_vec[k], '->', nz, ',', 'dens:', dens[i][j][k]
        #                phys_fields[field][nx][ny][nz] = dens[i][j][k]
                    
    print("\033[K", float("{0:.1f}".format(leaf * 100.0 / leaf_count)), "%")
    print('fields have been read successfully!\n')

    bbox = np.array([[-xscale, xscale], [-xscale, xscale], [-xscale, xscale]])

    dens_ind = np.where(hdf5_fields == 'rho_1')[0][0]
    phi_ind  = np.where(hdf5_fields == 'phi')[0][0]
    gx_ind   = np.where(hdf5_fields == 'gx')[0][0]
    gy_ind   = np.where(hdf5_fields == 'gy')[0][0]
    gz_ind   = np.where(hdf5_fields == 'gz')[0][0]

    print('creating dictionary...')

    data = dict(density = (phys_fields[dens_ind], "g/cm**3"),
        gpot = (phys_fields[phi_ind], "erg/g"),
        gx = (phys_fields[gx_ind], "erg/(g*cm)"),
        gy = (phys_fields[gy_ind], "erg/(g*cm)"),
        gz = (phys_fields[gz_ind], "erg/(g*cm)"))

    print('loading dataset...')
    import yt
    ds = yt.load_uniform_grid(data, phys_fields[dens_ind].shape, bbox=bbox, sim_time=sim_time)
    print('dataset created successfully!')
    return ds
def deco1(f):
    import yt
    yt.add_field(('gas','velocity_x'), function=_velocity_x, units="cm/s", take_log=False,
                     display_name='x velocity', sampling_type="cell", force_override=True)
def octo2yt_amr(filename, nspecies=1, epsilon_2=0.001, gather_outflows=True, copy_path='', savefile=True):
    import numpy as np
    import h5py
    import sys
    import yt
    import os
    from tqdm import tqdm

    #filename = '/home/shiber/simulations/sphere_ref5_1grid/final.silo'
    print('\nLoading Octo-Tiger simulation into Yt - by S. Shiber')
    print('------------------------------------------------------')

    print('\nopening file ' + filename + '...\n')
    f = h5py.File(filename, "r")

    print('opened succesfully!\n')
#        print 'reading headers...'
#        hdf5_items = f.items()

#        hdf5_data = f['.silo']
#        leaf_count = f['leaf_count'][0]
#    nspecies2 = f['n_species'][0]

    adiabatic_index = 5.0 / 3.0

 #       hdf5_fields = []
  #      fields_order = []

  #      xscale = f['xscale'][0]
   #     length_to_cm = f['code_to_cm'][0]
    #    time_to_s = f['code_to_s'][0]
     #   mass_to_g = f['code_to_g'][0]
      #  sim_time = f['time'][0]
       # omega = f['omega'][0]

        
    octo_fields = ['egas', 'sx', 'sy', 'sz', 'tau', 'lz', 'pot', 'gx', 'gy', 'gz']
#    octo_fields = ['egas', 'sx', 'sy', 'sz', 'tau', 'lz', 'phi', 'gx', 'gy', 'gz']
    outflow_fields = ['egas', 'sx', 'sy', 'sz', 'tau', 'lz', 'pot']
#    outflow_conversions = [mass_to_g * length_to_cm**2 / time_to_s**2, mass_to_g * length_to_cm / time_to_s, mass_to_g * length_to_cm / time_to_s, mass_to_g * length_to_cm / time_to_s, 
#                (mass_to_g * length_to_cm**2 / time_to_s**2)**(1 / adiabatic_index), mass_to_g * length_to_cm**2 / time_to_s, 
#                mass_to_g * length_to_cm**2 / time_to_s**2 ]
# tau is the entropy tracer defined by the internal energy density in power of the inverse of the adiabtic index (e^(1/gamma))

    units = ['erg/cm**3', 'g/(s*cm**2)', 'g/(s*cm**2)', 'g/(s*cm**2)', '(erg/cm**3)**('+str(adiabatic_index)+')', 'g/(cm*s)', 'erg/cm**3', 'erg/(g*cm)', 'erg/(g*cm)', 'erg/(g*cm)']

    yt_fields = ['egas', 'sx', 'sy', 'sz', 'tau', 'lz', 'pot', 'gx', 'gy', 'gz']

        #print 'reading headers...'


#        for i in range(1, nspecies+1):
#                octo_fields.append('rho_'+str(i))
#        units.append('g/cm**3')
#        yt_fields.append('rho_'+str(i))
#        outflow_fields.append('rho_'+str(i))
 #               outflow_conversions.append(mass_to_g)

 #       atomic_numbers = []
 #       atomic_masses = []
 #      Xs = []
 #       Zs = []

#    for i in range(0, nspecies2):
#        atomic_numbers.append(f['atomic_number'][i])
#        atomic_masses.append(f['atomic_mass'][i])
#        Xs.append(f['X'][i])
#        Zs.append(f['Z'][i])

#    print 'Number of Species (by the datafile): ', nspecies2
#    print 'Atomic Numbers: ', atomic_numbers
#    print 'Atomic Masses: ', atomic_masses
#    print 'Hydrogen mass: ', Xs
#    print 'Metalicities: ', Zs
 #       print '----------'
#    print 'continue working with Number of Species of :', nspecies

#        for i in range(0, len(hdf5_items)):
 #               if (type(hdf5_items[i][1]) == h5py._hl.datatype.Datatype):
  #                      attrs = f[hdf5_items[i][0]].attrs
   #                     if (attrs['silo_type'] == 521):
    #                            hdf5_fields.append(hdf5_items[i][0])
     #                           pos = attrs['silo'][-2]
      #                          pos = pos[pos.find('#')+1:]
       #                         print i, hdf5_fields, pos
        #                        fields_order.append(int(pos))

    f2, hdf5_fields, fields_order, leaf_count, total_nodes, xscale, length_to_cm, time_to_s, mass_to_g, sim_time, omega, \
        atomic_numbers, atmoic_masses, Xs, Zs, \
        spec_octo_fields, spec_units, spec_yt_fields, spec_outflow_fields, spec_outflow_conversions = read_headers(filename, nspecies)

    outflow_conversions = [mass_to_g * length_to_cm**2 / time_to_s**2, mass_to_g * length_to_cm / time_to_s, mass_to_g * length_to_cm / time_to_s, mass_to_g * length_to_cm / time_to_s,
                                (mass_to_g * length_to_cm**2 / time_to_s**2)**(1 / adiabatic_index), mass_to_g * length_to_cm**2 / time_to_s,
                                mass_to_g * length_to_cm**2 / time_to_s**2 ]

    
#        total_nodes = f['node_count'][0]
#        print '\nnode count: ', total_nodes,'\n'
    #num = 1
        #while num*8 < total_nodes:
        #        num *=8

        #grid_size = int(round(num**(1.0/3))*8)
        #grid_size = 2

        #print 'headers have been read successfully!'

    field_round = len(hdf5_fields) + 3
    number_of_fields = len(hdf5_fields)

    hdf5_fields = np.array(hdf5_fields)[np.argsort(np.array(fields_order))]

    octo_fields = octo_fields + spec_octo_fields
    units = units + spec_units
    yt_fields = yt_fields + spec_yt_fields
    outflow_fields = outflow_fields + spec_outflow_fields
    outflow_conversions = outflow_conversions + spec_outflow_conversions

    hdf5_data = f['.silo']
    grid_data = []

    print('\nbox size: ', xscale)#, ' base resolution: ', grid_size
    print('will read the following fields:')
    print(octo_fields, '\nas:\n', yt_fields[:len(octo_fields)], '\nwith physical units:\n', units[:len(octo_fields)], '\n')

    print('reading... (from overall', leaf_count, 'leaves)')
    total_leaves = 0
    cur_leaf_count = 0
    io_file = 0
    data_file = filename
    outflows = []
    min_level = 100
    pbar = tqdm(total=leaf_count, ncols=80)
    while os.path.exists(data_file):    
        if io_file > 0:
            print('\n\nreading file', str(io_file), '...')
            f_data = h5py.File(data_file, "r")
            if gather_outflows:
                outflows_temp = get_outflows_f(f_data, outflow_fields)
                print('\ncontinuing reading fields...')
                outflows = outflows + outflows_temp
            hdf5_data = f_data['.silo']
            cur_leaf_count = len(list(hdf5_data.keys())) / field_round
        else:
            if (len(list(hdf5_data.items())) < 100):
                f_data = h5py.File(filename + '.data/0.silo', "r")
                hdf5_data = f_data['.silo']
                cur_leaf_count = len(list(hdf5_data.keys())) / field_round
                print('\n\nnew scheme of data detected. Reading using the new scheme...\n')
                if gather_outflows:
                    outflows = get_outflows_f(f_data, outflow_fields)
                    print('\ncontinuing reading fields...')
            else:
                if gather_outflows:
                    outflows = get_outflows_f(f, outflow_fields)
                    print('\ncontinuing reading fields...')
                cur_leaf_count = leaf_count
        io_file = io_file + 1
        data_file = filename + '.data/'+str(io_file)+'.silo'
#            for leaf in tqdm(range (0, cur_leaf_count), ncols=80):
        for leaf in range (0, int(cur_leaf_count)):
#    print leaf, cur_leaf_countutotal_leaves = total_leaves + 1
            total_leaves = total_leaves + 1
            cur_field = leaf * field_round + 1
           # print "\033[K", float("{0:.1f}".format(total_leaves * 100.0 / leaf_count)), "%\r",
            sys.stdout.flush()
            x_vec, x_length = get_coord(hdf5_data, cur_field)
            x_vec = make_fraction(x_vec, xscale * length_to_cm) 
            cur_field += 1
            y_vec, y_length = get_coord(hdf5_data, cur_field)
            y_vec = make_fraction(y_vec, xscale * length_to_cm)
            cur_field += 1
            z_vec, z_length = get_coord(hdf5_data, cur_field)               
            z_vec = make_fraction(z_vec, xscale * length_to_cm)
            #print 'level', -np.log2((x_vec[1]-x_vec[0]))
            cur_level = int(round(-np.log2((x_vec[1]-x_vec[0])))) - 2
            min_level = min(cur_level, min_level)
            cur_grid = dict(left_edge=[x_vec[0], y_vec[0], z_vec[0]],
            right_edge=[x_vec[-1], y_vec[-1], z_vec[-1]],
            level = cur_level,
            dimensions=[x_length-1, y_length - 1, z_length - 1])
            for field in range (0, number_of_fields):
                cur_field += 1
                cur_field_name = hdf5_fields[field]
                if octo_fields.count(cur_field_name) > 0:
                    ind = octo_fields.index(cur_field_name)
                    dens = hdf5_data['#'+str(cur_field).zfill(6)]
                    cur_grid[yt_fields[ind]] = (np.transpose(np.array(dens), (2,1,0)), units[ind])
            tmpDens = np.zeros(np.shape(cur_grid['rho_1'][0]))
            tmpn = np.zeros(np.shape(cur_grid['rho_1'][0]))
            for i in range(0, nspecies):
                atomic_numbers[i] = 1.008
                mu = atomic_numbers[i]
                tmpDens = tmpDens + cur_grid['rho_'+str(i+1)][0]
                tmpn = tmpn + cur_grid['rho_'+str(i+1)][0] / mu / yt.physical_constants.mass_hydrogen
    
            cur_grid['density'] = (tmpDens, cur_grid['rho_1'][1])
            cur_grid['n'] = (tmpn, cur_grid['rho_1'][1]+'/g')
            pbar.update(1)
            grid_data.append(cur_grid)
        
    print('\nfields have been read successfully!\n')
    pbar.close()

    grid_size = 8*2**min_level

    if copy_path == '':
            copy_filename = filename + '.yt'
    else:
            copy_filename = copy_path + '/' + f.filename[f.filename.rfind('/')+1:] + '.yt'
    print('\nsaving a copy of the grid to ' + copy_filename)
    if savefile: saveGridToNPZ(copy_filename, grid_data, grid_size, xscale, length_to_cm, mass_to_g, time_to_s, sim_time, omega, adiabatic_index, nspecies, epsilon_2, [s + '_outflow' for s in outflow_fields], [a * b for a, b in zip(outflows, outflow_conversions)])
    print('a copy was saved successfully!\n')
#        return grid_data
    bbox = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    print('\nloading dataset...')
    ds = yt.load_amr_grids(grid_data, [grid_size, grid_size, grid_size], bbox=bbox, length_unit = xscale * length_to_cm, sim_time=sim_time)
    print('dataset created successfully!')
    ds.mass_unit = mass_to_g * yt.units.g
    ds.time_unit = time_to_s * yt.units.s
    ds.omega_matter = omega / yt.units.second
    ds.fullpath = filename
    ds.gamma = adiabatic_index
    ds.directory = filename[:filename.rfind('/')]
    ds.parameters['xscale'] = xscale
    ds.parameters['n_species'] = nspecies
    ds.parameters['epsilon_2'] = epsilon_2
    ds.parameters['gamma'] = adiabatic_index
    if gather_outflows:
        print('\n total outflows values:')
        for i in range(0, len(outflow_fields)):
            ds.parameters[outflow_fields[i]+'_outflow'] = outflows[i] * outflow_conversions[i]
            print(' -', outflow_fields[i], outflows[i] * outflow_conversions[i])
    print('\nadding derived fields')
    add_octo_derived_fields(ds)
    print('fields have been added sucessfully!')
#        print '\nsaving a copy ' + copy_filename
#        SaveGridToNPZ(copy_filename, grid_data, grid_size, xscale, length_to_cm, mass_to_g, time_to_s, sim_time, omega, adiabatic_index, nspecies, epsilon_2, outflow_fields, outflows)
#        save_yt_copy(ds, yt_fields, copy_filename);
#        print 'a copy was saved successfully!\n'
    #yt.add_field(('gas','velocity_x'), function=_velocity_x, units="cm/s", take_log=False,
        #            display_name='x velocity', sampling_type="cell", force_override=True)
        #yt.add_field(('gas','velocity_y'), function=_velocity_y, units="cm/s", take_log=False,
        #            display_name='y velocity', sampling_type="cell", force_override=True)
        #yt.add_field(('gas','velocity_z'), function=_velocity_z, units="cm/s", take_log=False,
        #            display_name='z velocity', sampling_type="cell", force_override=True)
        #yt.add_field(('gas','eint'), function=_eint, units="erg/g", take_log=True,
        #            display_name='specific internal energy', sampling_type="cell", force_override=True)
        #yt.add_field(('gas','etot'), function=_etot, units="erg/g", take_log=True,
        #            display_name='total specific energy', sampling_type="cell", force_override=True)
        #yt.add_field(('gas','pressure'), function=_pressure, units="erg/cm**3", take_log=True,
        #            display_name='pressure', sampling_type="cell", force_override=True)
        #yt.add_field(('gas','temperature'), function=_temperature, units="K", take_log=True,
        #            display_name='temperature', sampling_type="cell", force_override=True)
        #yt.add_field(('gas','gpot'), function=_gpot, units="erg/g", take_log=False,
        #            display_name='Gravitational Potential', sampling_type="cell", force_override=True)
        #yt.add_field(('gas','angular_velocity'), function=_angular_velocity, units="1/s", take_log=False,
        #            display_name='Angular velocity', sampling_type="cell", force_override=True)
    #yt.add_field(('gas','velocity_x_rot'), function=_velocity_x_rot, units="cm/s", take_log=False,
        #            display_name=r'v_x^{\rm rot}', sampling_type="cell", force_override=True)
        #yt.add_field(('gas','velocity_y_rot'), function=_velocity_y_rot, units="cm/s", take_log=False,
        #            display_name=r'v_y^{\rm rot}', sampling_type="cell", force_override=True)
        #yt.add_field(('gas','velocity_z_rot'), function=_velocity_z_rot, units="cm/s", take_log=False,
        #            display_name=r'v_z^{\rm rot}', sampling_type="cell", force_override=True)
    return ds#, grid_data
def get_outflows(f):
    keys = list(f.keys())
    fields_name = []
    fields_value = []
    for i in range(0, len(keys)):
    #    print i, len(keys)
        if keys[i].isnumeric():
            tt=f[keys[i]]
            n_keys = list(tt.keys())
            for j in range(0, len(n_keys)):
                if '_outflow' in n_keys[j]:
                    fields_name.append(n_keys[j])
                    fields_value.append(tt[n_keys[j]][0])
    import numpy as np
    names = np.unique(fields_name)
    sums = []
    for i in range(0, len(names)):
    #    print i
        inds = np.where(np.array(fields_name) == names[i])
        #print i, names[i]
        #print inds, np.array(fields_value)[inds[0]]
        sums.append(np.array(fields_value)[inds[0]].sum())
    return names, np.array(sums)        
def get_outflows_f(f, fields_name):
#    import sys
    from tqdm import tqdm
    print(' reading outflows from', f.filename[f.filename.rfind('/')+1:], '...')
    keys = list(f.keys())
    import numpy as np
    fields_value = np.zeros(len(fields_name))
    for i in tqdm(list(range(0, len(keys)))):
        if keys[i].isnumeric():
            for j in range(0, len(fields_name)):
                fields_value[j] = fields_value[j] + f[keys[i]][fields_name[j]+'_outflow'][0]
    print(' outflows have been read successfully!\n')
    print(' outflows values:')
    for i in range(0, len(fields_name)):
        print(' -', fields_name[i], fields_value[i])
    return fields_value

def get_coord(hdf5_data, cur_field):
    vec = []
    vec.append(hdf5_data['#'+str(cur_field).zfill(6)][:1][0])
    vec.append(hdf5_data['#'+str(cur_field).zfill(6)][1:2][0])
    vec.append(hdf5_data['#'+str(cur_field).zfill(6)][-1:][0])
    return vec, hdf5_data['#'+str(cur_field).zfill(6)].shape[0]
def make_fraction(vec, denom):
    from fractions import Fraction
    import numpy as np
    vec_f = np.zeros(len(vec))
    for i in range(0, len(vec)):
        vec_f[i] = Fraction(vec[i] / denom).limit_denominator()
    return vec_f
def check_hdf5(filename):
    import h5py
    import gc
    f_data = h5py.File(filename + '.data/0.silo', 'r', rdcc_nbytes=256, rdcc_nslots=7, track_order=True)
    hdf5_data = f_data['.silo']
    pp = list(hdf5_data.items())
    f_data.close()
    del pp
    del hdf5_data
    del f_data
    del h5py
    gc.collect()
def _etot(field, data):
    return data["eint"] + 0.5 * (data['velocity_x']**2 + data['velocity_y']**2 + data['velocity_z']**2)
def _eint(field, data):
    adiabatic_index = data.ds.parameters['gamma']
    return data["pressure"] / (adiabatic_index-1) / data["density"]
def _velocity_x(field, data):
    return data["sx"] / data["density"]
def _velocity_y(field, data):
    return data["sy"] / data["density"]
def _velocity_z(field, data):
    return data["sz"] / data["density"]
def _velocity_x_rot(field, data):
    return data["velocity_x"] + data.ds.omega_matter * data["y"]
def _velocity_y_rot(field, data):
    return data["velocity_y"] - data.ds.omega_matter * data["x"]
def _velocity_z_rot(field, data):
    return data["velocity_z"]
def _pressure(field, data):
    import yt
    adiabatic_index = data.ds.parameters['gamma']
    epsilon_2 = data.ds.parameters['epsilon_2']
    ei = data['egas'] - 0.5 * (data['sx']**2 + data['sy']**2 + data['sz']**2) / data['density']
    ei_dual = data['tau'].v**adiabatic_index * yt.units.erg / yt.units.cm**3
    return (adiabatic_index-1) * ((ei <= epsilon_2 * data['egas']) * ei_dual + (ei > epsilon_2 * data['egas']) * ei)                
def _temperature(field, data):
    import yt
    A = 6.002565788159526e+22 * yt.units.erg * yt.units.cm**-3
    B = 1962085.0342234566 * yt.units.g * yt.units.cm**-3
    x = (data['density'] / B) ** (1/3)
    edeg = A*(8*(x)**3*(((x)**2+1)**0.5-1)-((x)*(2*(x)**2-3)*((x)**2+1)**0.5)+3*np.arcsinh(x))
    ek = 0.5 * (data['sy']**2 + data['sy']**2 + data['sz']**2) / data['density']
    epsilon_2 = data.ds.parameters['epsilon_2']
    eint = (data['egas'] - ek - edeg) * ((data['egas'] - ek - edeg) >= epsilon_2 * data['egas']) \
        + (np.array(data['tau']**(5/3))*yt.units.erg * yt.units.cm**-3) * ((data['egas'] - ek - edeg) < epsilon_2 * data['egas'])
    
    return (eint/data['density']*(4/3)*yt.physical_constants.mass_hydrogen_cgs/yt.physical_constants.boltzmann_constant_cgs) * (eint > 0)
    #return data['pressure'] / data['n'] / yt.physical_constants.boltzmann_constant_cgs
def _gpot(field, data):
    return data['pot'] / data['density']
def _angular_velocity(field, data):
    return (data['x']*data['velocity_y']-data['y']*data['velocity_x'])/(data['x']**2+data['y']**2)
def save_yt_copy(ds, field_list, save_name):
    pf = ds.all_data()
    copy_fields = []
    for i in range(0, len(field_list)):
        copy_fields.append(('stream', field_list[i]))
    copy_fields.append(('stream', 'n'))
    copy_fields.append(('stream', 'density'))
#    copy_fields.append(('gas', 'pressure'))
#    copy_fields.append(('gas', 'velocity_x'))
#    copy_fields.append(('gas', 'velocity_y'))
#    copy_fields.append(('gas', 'velocity_z'))
#    copy_fields.append(('gas', 'eint'))
#    copy_fields.append(('gas', 'etot'))
#    copy_fields.append(('gas', 'temperature'))
#    copy_fields.append(('gas', 'gpot'))
#    copy_fields.append(('gas', 'angular_velocity'))
#    copy_fields.append(('gas', 'velocity_x_rot'))
#    copy_fields.append(('gas', 'velocity_y_rot'))
#    copy_fields.append(('gas', 'velocity_z_rot'))
    return pf.save_as_dataset(save_name, fields=copy_fields)
def add_octo_derived_fields(ds=[]):
    import yt
    if ds == []:
        yt.add_field(('gas','velocity_x'), function=_velocity_x, units="cm/s", take_log=False,
                    display_name='x velocity', sampling_type="cell", force_override=True)
        yt.add_field(('gas','velocity_y'), function=_velocity_y, units="cm/s", take_log=False,
                    display_name='y velocity', sampling_type="cell", force_override=True)
        yt.add_field(('gas','velocity_z'), function=_velocity_z, units="cm/s", take_log=False,
                    display_name='z velocity', sampling_type="cell", force_override=True)
        yt.add_field(('gas','pressure'), function=_pressure, units="erg/cm**3", take_log=True,
                    display_name='pressure', sampling_type="cell", force_override=True)
        yt.add_field(('gas','eint'), function=_eint, units="erg/g", take_log=True,
                    display_name='specific internal energy', sampling_type="cell", force_override=True)
        yt.add_field(('gas','etot'), function=_etot, units="erg/g", take_log=True,
                    display_name='total specific energy', sampling_type="cell", force_override=True)
        yt.add_field(('gas','temperature'), function=_temperature, units="K", take_log=True,
                    display_name='temperature', sampling_type="cell", force_override=True)
        yt.add_field(('gas','gpot'), function=_gpot, units="erg/g", take_log=False,
                    display_name='Gravitational Potential', sampling_type="cell", force_override=True)
        yt.add_field(('gas','angular_velocity'), function=_angular_velocity, units="1/s", take_log=False,
                    display_name='Angular velocity', sampling_type="cell", force_override=True)
        yt.add_field(('gas','velocity_x_rot'), function=_velocity_x_rot, units="cm/s", take_log=False,
                    display_name=r'v_x^{\rm rot}', sampling_type="cell", force_override=True)
        yt.add_field(('gas','velocity_y_rot'), function=_velocity_y_rot, units="cm/s", take_log=False,
                    display_name=r'v_y^{\rm rot}', sampling_type="cell", force_override=True)
        yt.add_field(('gas','velocity_z_rot'), function=_velocity_z_rot, units="cm/s", take_log=False,
                    display_name=r'v_z^{\rm rot}', sampling_type="cell", force_override=True)
    else:
        ds.add_field(('gas','velocity_x'), function=_velocity_x, units="cm/s", take_log=False,
                    display_name='x velocity', sampling_type="cell", force_override=True)
        ds.add_field(('gas','velocity_y'), function=_velocity_y, units="cm/s", take_log=False,
                    display_name='y velocity', sampling_type="cell", force_override=True)
        ds.add_field(('gas','velocity_z'), function=_velocity_z, units="cm/s", take_log=False,
                    display_name='z velocity', sampling_type="cell", force_override=True)
        ds.add_field(('gas','pressure'), function=_pressure, units="erg/cm**3", take_log=True,
                    display_name='pressure', sampling_type="cell", force_override=True) 
        ds.add_field(('gas','eint'), function=_eint, units="erg/g", take_log=True,
                    display_name='specific internal energy', sampling_type="cell", force_override=True)
        ds.add_field(('gas','etot'), function=_etot, units="erg/g", take_log=True,
                    display_name='total specific energy', sampling_type="cell", force_override=True)
        ds.add_field(('gas','temperature'), function=_temperature, units="K", take_log=True,
                    display_name='temperature', sampling_type="cell", force_override=True)
        ds.add_field(('gas','gpot'), function=_gpot, units="erg/g", take_log=False,
                    display_name='Gravitational Potential', sampling_type="cell", force_override=True)
        ds.add_field(('gas','angular_velocity'), function=_angular_velocity, units="1/s", take_log=False,
                    display_name='Angular velocity', sampling_type="cell", force_override=True)
        ds.add_field(('gas','velocity_x_rot'), function=_velocity_x_rot, units="cm/s", take_log=False,
                    display_name=r'v_x^{\rm rot}', sampling_type="cell", force_override=True)
        ds.add_field(('gas','velocity_y_rot'), function=_velocity_y_rot, units="cm/s", take_log=False,
                    display_name=r'v_y^{\rm rot}', sampling_type="cell", force_override=True)
        ds.add_field(('gas','velocity_z_rot'), function=_velocity_z_rot, units="cm/s", take_log=False,
                    display_name=r'v_z^{\rm rot}', sampling_type="cell", force_override=True)
def YTDataSetToAMRGrid(ds):
    from tqdm import tqdm
    grid_data = []
    pf = ds.all_data()
    for i in tqdm(list(range(0, len(pf['x'])))):
        x_i = pf['x'][i]
        dx_i = pf['dx'][i]
        y_i = pf['y'][i]
        dy_i = pf['dy'][i]
        z_i = pf['z'][i]
        dz_i = pf['dz'][i]
        cur_grid = dict(left_edge=[x_i - 0.5 * dx_i, y_i - 0.5 * dy_i, z_i - 0.5 * dz_i],
                        right_edge=[x_i + 0.5 * dx_i, y_i + 0.5 * dy_i, z_i + 0.5 * dz_i],
                        level = pf['grid_level'][i] - 2,
                        dimensions=[1, 1, 1])
        cur_grid['density'] = pf['density'][i]
        grid_data.append(cur_grid)
def saveGridToNPZ(filename, g, grid_size, xscale, length_to_cm, mass_to_g, time_to_s, sim_time, omega, adiabatic_index, nspecies, epsilon_2, outflows_names, outflows):
    import numpy as np
    np.savez_compressed(filename, grid_data=g, grid_size=grid_size, xscale=xscale, length_to_cm=length_to_cm, mass_to_g=mass_to_g, time_to_s=time_to_s, sim_time=sim_time, omega=omega, adiabatic_index=adiabatic_index, nspecies=nspecies, epsilon_2=epsilon_2, outflows_names=outflows_names, outflows=outflows, allow_pickle=True)
def loadFromNPZ(filename, encoded_in="3"):
    import numpy as np
    import yt
    if encoded_in == "2":
        pf = np.load(filename, allow_pickle=True, encoding="latin1")
    if encoded_in == "3":
        pf = np.load(filename, allow_pickle=True)
    grid_data = pf['grid_data']
    grid_size = pf['grid_size'].item()
    xscale = pf['xscale'].item()
    length_to_cm = pf['length_to_cm'].item()
    mass_to_g = pf['mass_to_g'].item()
    time_to_s = pf['time_to_s'].item()
    sim_time = pf['sim_time'].item()
    omega = pf['omega'].item()
    adiabatic_index = pf['adiabatic_index'].item()
    nspecies = pf['nspecies'].item()
    epsilon_2 = pf['epsilon_2'].item()
    outflows_names = pf['outflows_names']
    outflows = pf['outflows']

    bbox = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    ds = yt.load_amr_grids(grid_data, [grid_size, grid_size, grid_size], bbox=bbox, length_unit = xscale * length_to_cm, sim_time=sim_time)
    ds.mass_unit = mass_to_g * yt.units.g
    ds.time_unit = time_to_s * yt.units.s
    ds.omega_matter = omega / yt.units.second
    ds.fullpath = filename
    ds.gamma = adiabatic_index
    ds.directory = filename[:filename.rfind('/')]
    ds.parameters['xscale'] = xscale
    ds.parameters['n_species'] = nspecies
    ds.parameters['epsilon_2'] = epsilon_2
    ds.parameters['gamma'] = adiabatic_index
    for i in range(0, len(outflows)):
        ds.parameters[outflows_names[i]] = outflows[i]
    add_octo_derived_fields()
    return ds
def read_headers(filename, nspecies=5):
        import h5py

        print('reading headers...')

        f = h5py.File(filename, "r")

        hdf5_items = list(f.items())

        leaf_count = f['leaf_count'][0]
        nspecies2 = f['n_species'][0]

        adiabatic_index = 5.0 / 3.0

        hdf5_fields = []
        fields_order = []
        octo_fields = []
        units = []
        yt_fields = []
        outflow_fields = []
        outflow_conversions = []

        xscale = f['xscale'][0]
        length_to_cm = f['code_to_cm'][0]
        time_to_s = f['code_to_s'][0]
        mass_to_g = f['code_to_g'][0]
        sim_time = f['time'][0]
        omega = f['omega'][0]


        for i in range(1, nspecies+1):
                octo_fields.append('rho_'+str(i))
                units.append('g/cm**3')
                yt_fields.append('rho_'+str(i))
                outflow_fields.append('rho_'+str(i))
                outflow_conversions.append(mass_to_g)

        atomic_numbers = []
        atomic_masses = []
        Xs = []
        Zs = []

        for i in range(0, nspecies2):
                atomic_numbers.append(f['atomic_number'][i])
                atomic_masses.append(f['atomic_mass'][i])
                Xs.append(f['X'][i])
                Zs.append(f['Z'][i])

        print('Number of Species (by the datafile): ', nspecies2)
        print('Atomic Numbers: ', atomic_numbers)
        print('Atomic Masses: ', atomic_masses)
        print('Hydrogen mass: ', Xs)
        print('Metalicities: ', Zs)
        print('----------')
        print('continue working with Number of Species of :', nspecies)

        for i in range(0, len(hdf5_items)):
                if (type(hdf5_items[i][1]) == h5py._hl.datatype.Datatype):
                        attrs = f[hdf5_items[i][0]].attrs
                        if (attrs['silo_type'] == 521):
                                hdf5_fields.append(hdf5_items[i][0])
                                pos = attrs['silo'][-2].tobytes()
                                pos = pos[pos.find(b'#')+1:]
                                print(i, hdf5_fields, pos)
                                fields_order.append(int(pos))

        total_nodes = f['node_count'][0]
        print('\nnode count: ', total_nodes,'\n')

        print('headers have been read successfully!')

        return f, hdf5_fields, fields_order, leaf_count, total_nodes, xscale, length_to_cm, time_to_s, mass_to_g, sim_time, omega, \
        atomic_numbers, atomic_masses, Xs, Zs, octo_fields, units, yt_fields, outflow_fields, outflow_conversions

