# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import matplotlib.pyplot as plt
import stim

def build_triangle_HeavyHEX(d):
    """building the dictionary to map coordinates (x+iy) to qubit indices a (hex_rows)x(hex_cols) lattice"""
    heavyHEX_dict = {}
    excluded_qubits = []
    i=0
    row=1
    num_cols = 0
    while row <= d//2:
        even_row = row%2==0
        num_cols += 1 + (1-even_row)
        for re in range(0,num_cols*4+even_row*2+1):
            heavyHEX_dict[re+1j*2*(row-1)] = i
            if re in [0,num_cols*4+even_row*2-2,num_cols*4+even_row*2]:
                excluded_qubits+=[i]
            i+=1
        for re in range(even_row*2,num_cols*4+even_row*2+1,4):
            heavyHEX_dict[re+1j*(2*(row-1)+1)] = i
            i+=1
        row+=1

    for re in range(0,num_cols*4+even_row*2+1):
        heavyHEX_dict[re+1j*(2*(row-1))] = i
        if re in [0]:
            excluded_qubits+=[i]
        i+=1
    num_cols += 1
    while row < d:
        even_row = row%2==0
        num_cols -= 1 + (even_row)
        for re in range(even_row*2,num_cols*4+even_row*2+1,4):
            heavyHEX_dict[re+1j*(2*(row-1)+1)] = i
            i+=1
        for re in range((row==d-1)*2,num_cols*4+even_row*2+1):
            heavyHEX_dict[re+1j*(2*(row-1)+2)] = i
            if re in [0,num_cols*4+even_row*2-2,num_cols*4+even_row*2]:
                excluded_qubits+=[i]
            i+=1
        row+=1

    return heavyHEX_dict,excluded_qubits

def draw_lattice(heavyHEX_dict, coords = [], coord_col = 'red', links = [], link_col = 'red',indices:bool = True):
    qi_list, y_list, x_list = np.transpose([[item[1],item[0].imag,item[0].real] for item in heavyHEX_dict.items()])

    plt.figure(frameon=False)

    inv_dict = {v:k for k,v in heavyHEX_dict.items()}
    for link in links:
        if len(link)==2:
            plt.plot([inv_dict[link[0]].real,inv_dict[link[1]].real],
                     [-inv_dict[link[0]].imag,-inv_dict[link[1]].imag], c=link_col)
        else:
            plt.plot([inv_dict[link[0]].real],
                     [-inv_dict[link[0]].imag],'o',c=link_col)
            
    plt.plot(x_list,-y_list,'o',c='gray',alpha = 0.65)
    for i in [int(qi) for qi in qi_list]:
        if indices:
            plt.text(x_list[i],-y_list[i],str(i),fontsize=7)

    plt.plot(np.array(coords).real,-np.array(coords).imag,'o',c=coord_col,alpha = 0.65)
    for i in [int(qi) for qi in qi_list]:
        if indices:
            plt.text(x_list[i],-y_list[i],str(i),fontsize=7)

    plt.show()
    pass

def generate_color_hex_centers(hex_dict, start_coord, boundaries = False, excluded_qubits = []):
    """
        args:
            hex_dict: dictionary of the HHX lattice with the keys being the coordinates, and values are the qubit indices
            start_coord: first center coordinate either first column of first row!
        returns: coordinate list of hex centers of a given color
    """
    col_coords = []
    HHX_coords = np.array(list(hex_dict.keys()))
    hex_around_center = np.array([-2-1j,-1j,2-1j,2+1j,1j,-2+1j])
    coord = -1j
    while coord.imag<=HHX_coords.imag.max()+2:
        while coord.real<=HHX_coords.real.max()+2:
            diff = (start_coord-coord)
            if (diff.real%12==0 and diff.imag%4==0) or (diff.real%12==6 and diff.imag%4==2):
                if all([coord+rel_pos in hex_dict for rel_pos in hex_around_center]):
                    col_coords.append(coord)    
                elif boundaries and any([coord+rel_pos in hex_dict for rel_pos in hex_around_center]) and all([hex_dict[coord+rel_pos] in excluded_qubits for rel_pos in hex_around_center if coord+rel_pos in hex_dict]):
                    col_coords.append(coord)    
            coord+=4
        coord = -(diff.imag%4) + coord.imag*1j+2j
    return col_coords


def ColorCode(d = 5, num_rounds = 1, log_obs = 'Z', error_rate = 0.001):
    """
    hex_rows: must be multiples of 2 (due to the implemented boundary conditions); X-distance (weight of log-Z) is hex_rows-1
    hex_cols: must be multiples of 3 (due to the implemented boundary conditions); Z-distance is hex_cols+1+k where k is unknown...
    num_rounds: number of full syndrome measurement rounds (6 mid-circuit measurement rounds)
    log_obs: logical observable to prepare (X or Z)
    error rate: depolarizing error, after gates and on the idling qubits during the measurements
    """
    heavyHEX_dict,excluded_qubits=build_triangle_HeavyHEX(d)
    heavyHEX_dict_inv = {v:k for k,v in heavyHEX_dict.items()}

    blue_centers_no_bounary = generate_color_hex_centers(hex_dict=heavyHEX_dict,start_coord=2+1j)
    red_centers_no_bounary = generate_color_hex_centers(hex_dict=heavyHEX_dict,start_coord=6+1j)
    green_centers_no_bounary = generate_color_hex_centers(hex_dict=heavyHEX_dict,start_coord=-2+1j)

    excluded_ancillas = [heavyHEX_dict[heavyHEX_dict_inv[q]+i] 
                         for q in excluded_qubits 
                         for i in [+1,-1,+1j,-1j] 
                         if heavyHEX_dict_inv[q]+i in heavyHEX_dict]
    heavyHEX_dict_inv = {v:k for k,v in heavyHEX_dict.items() 
                         if v not in excluded_qubits and v not in excluded_ancillas}
    heavyHEX_dict = {v:k for k,v in heavyHEX_dict_inv.items()}
    blue_centers = blue_centers_no_bounary
    red_centers = red_centers_no_bounary
    green_centers = green_centers_no_bounary


    stim_circuit = stim.Circuit()

    for q_coord, q_ind in heavyHEX_dict.items():
        stim_circuit.append("QUBIT_COORDS",q_ind,[q_coord.real,q_coord.imag])

    qubit_coords_around_center = np.array([-2-1j,-1j,2-1j,2+1j,1j,-2+1j])
    ancilla_coords_around_center = np.array([-1-1j,1-1j,2,1+1j,-1+1j,-2])


    def hex_stab_measure(basis, hex_centers):
        meas_ind = stim_circuit.num_measurements
        for center in hex_centers:
            ancillas = [heavyHEX_dict[c] 
                        for c in center+ancilla_coords_around_center 
                        if c in heavyHEX_dict]
            qubits = [heavyHEX_dict[c] 
                      for c in center+qubit_coords_around_center 
                      if c in heavyHEX_dict]
            if basis == 'Z':
                stim_circuit.append("H",qubits)
                stim_circuit.append('DEPOLARIZE1',qubits,error_rate)        
        for center in hex_centers:
            ancillas = [heavyHEX_dict[c] 
                        for c in center+ancilla_coords_around_center 
                        if c in heavyHEX_dict]
            stim_circuit.append("R",ancillas)
            stim_circuit.append('X_ERROR',ancillas,error_rate)        
        # stim_circuit.append("TICK")
        for center in hex_centers:
            qa_pairs = [[heavyHEX_dict[q],heavyHEX_dict[a]] 
                        for q,a in zip(center+qubit_coords_around_center,center+ancilla_coords_around_center) 
                        if q in heavyHEX_dict and a in heavyHEX_dict]
            stim_circuit.append("CX",[q for qa in qa_pairs for q in qa])
            stim_circuit.append('DEPOLARIZE2',[q for qa in qa_pairs for q in qa],error_rate)
        for center in hex_centers:
            qa_pairs = [[heavyHEX_dict[q],heavyHEX_dict[a]] 
                        for q,a in zip(center+qubit_coords_around_center[[1,2,3,4,5,0]],center+ancilla_coords_around_center) 
                        if q in heavyHEX_dict and a in heavyHEX_dict]
            stim_circuit.append("CX",[q for qa in qa_pairs for q in qa])
            stim_circuit.append('DEPOLARIZE2',[q for qa in qa_pairs for q in qa],error_rate)
        # stim_circuit.append("TICK")
        for center in hex_centers:
            qubits = [heavyHEX_dict[c] 
                      for c in center+qubit_coords_around_center 
                      if c in heavyHEX_dict]
            stim_circuit.append("H",qubits)
            stim_circuit.append('DEPOLARIZE1',qubits,error_rate)        
        for center in hex_centers:
            qubits = [heavyHEX_dict[c] 
                      for c in center+qubit_coords_around_center 
                      if c in heavyHEX_dict]
            stim_circuit.append('X_ERROR',qubits,error_rate)        
            stim_circuit.append("M",qubits)
            stab_dict[(center,basis,time)]={k:v for k,v in zip(qubits,range(meas_ind,meas_ind+len(qubits)))}
            meas_ind+=len(qubits)
        for center in hex_centers:
            qubits = [heavyHEX_dict[c] 
                      for c in center+qubit_coords_around_center 
                      if c in heavyHEX_dict]
            stim_circuit.append("H",qubits)
            stim_circuit.append('DEPOLARIZE1',qubits,error_rate)        
            stim_targets = []
            ###adding full plaquettes to the detectors
            if not (time == 0 and basis != log_obs):
                stim_targets += [stim.target_rec(i-meas_ind) for i in stab_dict[(center,basis,time)].values()]
            if not time == 0:
                stim_targets += [stim.target_rec(i-meas_ind) for i in stab_dict[(center,basis,time-1)].values()]
            ###adding links from plaquettes measured in the opposite basis
            stim_targets1 = []
            stim_targets2 = []
            if not (time == 0 and basis != log_obs):
                plaq_basis = (basis=='Z')*'X'+(basis=='X')*'Z'
                plaq_time = (basis=='Z')*(time+1/2)+(basis=='X')*(time-1/2)
                ###adding links from neighboring plaquettes
                for rel_pos in ancilla_coords_around_center:
                    plaq_center = center+2*rel_pos
                    if center+rel_pos in heavyHEX_dict:
                        anc_ind = heavyHEX_dict[center+rel_pos]
                        if plaq_center in red_centers+blue_centers+green_centers and plaq_time>=0:
                            stim_targets +=[stim.target_rec(stab_dict[(plaq_center,plaq_basis,plaq_time)][anc_ind]-meas_ind)] 
                ###adding links from the same plaquette
                if plaq_time>=0:
                    anc_inds1 = [heavyHEX_dict[center+rel_pos]
                                 for rel_pos in ancilla_coords_around_center[[0,2,4]] 
                                 if center+rel_pos in heavyHEX_dict]
                    if len(anc_inds1)>1:
                        stim_targets1 = stim_targets.copy()
                        for anc_ind in anc_inds1:
                            stim_targets1 +=[stim.target_rec(stab_dict[(center,plaq_basis,plaq_time)][anc_ind]-meas_ind)]
                    anc_inds2 = [heavyHEX_dict[center+rel_pos] 
                                 for rel_pos in ancilla_coords_around_center[[1,3,5]] 
                                 if center+rel_pos in heavyHEX_dict]
                    if len(anc_inds2)>1:
                        stim_targets2 = stim_targets.copy()
                        for anc_ind in anc_inds2:
                            stim_targets2 += [stim.target_rec(stab_dict[(center,plaq_basis,plaq_time)][anc_ind]-meas_ind)]
            if len(stim_targets1)>0:
                stim_circuit.append("DETECTOR", stim_targets1,[center.real,center.imag,time,basis=='Z'])
            if len(stim_targets2)>0 and stim_targets!=stim_targets2:
                stim_circuit.append("DETECTOR", stim_targets2,[center.real,center.imag,time,basis=='Z'])
            if len(stim_targets1)==0 and len(stim_targets2)==0 and len(stim_targets)>0:
                stim_circuit.append("DETECTOR", stim_targets,[center.real,center.imag,time,basis=='Z'])

        stim_circuit.append("TICK")
        for center in hex_centers:
            qa_pairs = [[heavyHEX_dict[q],heavyHEX_dict[a]] 
                        for q,a in zip(center+qubit_coords_around_center,center+ancilla_coords_around_center) 
                        if q in heavyHEX_dict and a in heavyHEX_dict]
            stim_circuit.append("CX",[q for qa in qa_pairs for q in qa])
            stim_circuit.append('DEPOLARIZE2',[q for qa in qa_pairs for q in qa],error_rate)
        for center in hex_centers:
            qa_pairs = [[heavyHEX_dict[q],heavyHEX_dict[a]] 
                        for q,a in zip(center+qubit_coords_around_center[[1,2,3,4,5,0]],center+ancilla_coords_around_center) 
                        if q in heavyHEX_dict and a in heavyHEX_dict]
            stim_circuit.append("CX",[q for qa in qa_pairs for q in qa])
            stim_circuit.append('DEPOLARIZE2',[q for qa in qa_pairs for q in qa],error_rate)
        # stim_circuit.append("TICK")
        for center in hex_centers:
            ancillas = [heavyHEX_dict[c] 
                        for c in center+ancilla_coords_around_center 
                        if c in heavyHEX_dict]
            stim_circuit.append('X_ERROR',ancillas,error_rate)        
            stim_circuit.append("M",ancillas)
            stab_dict[(center,basis,time+1/2)]={k:v for k,v in zip(ancillas,range(meas_ind,meas_ind+len(ancillas)))}
            meas_ind+=len(ancillas)
            if basis != log_obs:
                log_targets = []
                all_six_anc_coord = [c for c in center+ancilla_coords_around_center]
                log_ancillas = [heavyHEX_dict[c] 
                                for c in np.array(all_six_anc_coord)[::2] 
                                if c in heavyHEX_dict]
                if len(log_ancillas)<2:
                    log_ancillas = [heavyHEX_dict[c] 
                                    for c in np.array(all_six_anc_coord)[1::2] 
                                    if c in heavyHEX_dict]
                for a_ind in log_ancillas:
                    log_targets += [stim.target_rec(stab_dict[(center,basis,time+1/2)][a_ind]-meas_ind)]
                stim_circuit.append("OBSERVABLE_INCLUDE", log_targets,0)


        for center in hex_centers:
            qubits = [heavyHEX_dict[c] for c in center+qubit_coords_around_center if c in heavyHEX_dict]
            if basis == 'Z':
                stim_circuit.append("H",qubits)
                stim_circuit.append('DEPOLARIZE1',qubits,error_rate)        
        stim_circuit.append("TICK")

    qubit_list = list(set([heavyHEX_dict[c] for centers in [blue_centers,red_centers,green_centers] 
                                            for center in centers 
                                            for c in center+qubit_coords_around_center
                                            if c in heavyHEX_dict
                                            ]))

    if log_obs == 'X':
        stim_circuit.append('H',qubit_list)
        stim_circuit.append('DEPOLARIZE1',qubit_list,error_rate)        

    stim_circuit.append('DEPOLARIZE1',qubit_list,error_rate)

    stab_dict = {}

    for time in range(num_rounds):
        hex_stab_measure(basis='X',hex_centers=blue_centers)
        stim_circuit.append('DEPOLARIZE1',qubit_list,error_rate)
        hex_stab_measure(basis='X',hex_centers=red_centers)
        stim_circuit.append('DEPOLARIZE1',qubit_list,error_rate)
        hex_stab_measure(basis='X',hex_centers=green_centers)
        stim_circuit.append('DEPOLARIZE1',qubit_list,error_rate)

        hex_stab_measure(basis='Z',hex_centers=blue_centers)
        stim_circuit.append('DEPOLARIZE1',qubit_list,error_rate)
        hex_stab_measure(basis='Z',hex_centers=red_centers)
        stim_circuit.append('DEPOLARIZE1',qubit_list,error_rate)
        hex_stab_measure(basis='Z',hex_centers=green_centers)
        stim_circuit.append('DEPOLARIZE1',qubit_list,error_rate)

    meas_ind = stim_circuit.num_measurements
    if log_obs == 'X':
        stim_circuit.append('H',qubit_list)
        stim_circuit.append('DEPOLARIZE1',qubit_list,error_rate)
    stim_circuit.append('X_ERROR',qubit_list,error_rate)        
    stim_circuit.append('M',qubit_list)
    final_meas_ind = {}
    for i,q in enumerate(qubit_list):
        final_meas_ind[heavyHEX_dict_inv[q]] = meas_ind+i
    meas_ind = stim_circuit.num_measurements


    for center in blue_centers+green_centers+red_centers:
        stim_targets = [stim.target_rec(final_meas_ind[center+rel_pos]-meas_ind) 
                        for rel_pos in qubit_coords_around_center 
                        if center+rel_pos in final_meas_ind]
        stim_targets += [stim.target_rec(i-meas_ind) 
                         for i in stab_dict[(center,log_obs,time)].values()]
        
        if log_obs=='X':
            ###adding links from neighboring plaquettes
            for rel_pos in ancilla_coords_around_center:
                plaq_center = center+2*rel_pos
                if center+rel_pos in heavyHEX_dict:
                    anc_ind = heavyHEX_dict[center+rel_pos]
                    if plaq_center in red_centers+blue_centers+green_centers:
                        stim_targets +=[stim.target_rec(stab_dict[(plaq_center,'Z',time+1/2)][anc_ind]-meas_ind)] 
            ###adding links from the same plaquette
            stim_targets1 = []
            stim_targets2 = []
            anc_inds1 = [heavyHEX_dict[center+rel_pos] 
                         for rel_pos in ancilla_coords_around_center[[0,2,4]] 
                         if center+rel_pos in heavyHEX_dict]
            if len(anc_inds1)>1:
                stim_targets1 = stim_targets.copy()
                for anc_ind in anc_inds1:
                    stim_targets1 +=[stim.target_rec(stab_dict[(center,'Z',time+1/2)][anc_ind]-meas_ind)]
                stim_circuit.append("DETECTOR", stim_targets1,[center.real,center.imag,time+1])
            anc_inds2 = [heavyHEX_dict[center+rel_pos]
                         for rel_pos in ancilla_coords_around_center[[1,3,5]] 
                         if center+rel_pos in heavyHEX_dict]
            if len(anc_inds2)>1:
                stim_targets2 = stim_targets.copy()
                for anc_ind in anc_inds2:
                    stim_targets2 +=[stim.target_rec(stab_dict[(center,'Z',time+1/2)][anc_ind]-meas_ind)]
                stim_circuit.append("DETECTOR", stim_targets2,[center.real,center.imag,time+1])
        else:
            stim_circuit.append("DETECTOR", stim_targets,[center.real,center.imag,time+1])


    log_qubits = qubit_list
    log_targets = [qubit_list.index(q)-len(qubit_list) for q in log_qubits]
    stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(i) for i in log_targets],0)

    return stim_circuit
